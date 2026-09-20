from __future__ import annotations

import math

import torch
from torch import nn


class BiLSTMNet(nn.Module):
    def __init__(self, channels: int = 64, depth: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(1, channels, num_layers=depth, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(channels * 2, channels), nn.GELU(), nn.Linear(channels, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)
        y, _ = self.lstm(seq)
        return x + self.head(y).transpose(1, 2)


class TransformerSR(nn.Module):
    def __init__(self, channels: int = 64, depth: int = 3, nhead: int = 4, max_len: int = 1024):
        super().__init__()
        # 卷积主干 (核 > 1) 加正弦位置编码: 缺少任一项, 编码器对深度位置置换等变, 无法定位层界。
        self.inp = nn.Sequential(
            nn.Conv1d(1, channels, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, channels, 2, dtype=torch.float32) * (-math.log(10000.0) / channels))
        pe = torch.zeros(max_len, channels)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pos_encoding", pe, persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=nhead, dim_feedforward=channels * 4, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.out = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x).transpose(1, 2)
        h = h + self.pos_encoding[: h.shape[1]].unsqueeze(0).to(h.dtype)
        h = self.encoder(h).transpose(1, 2)
        return x + self.out(h)


def morphology_channels(x: torch.Tensor) -> torch.Tensor:
    """由 LR 导出的四个描述子: 原值、一阶差分、与 9 点滑动平均的偏差、二阶差分。"""
    grad = torch.zeros_like(x)
    grad[..., 1:] = x[..., 1:] - x[..., :-1]
    smooth = nn.functional.avg_pool1d(x, kernel_size=9, stride=1, padding=4, count_include_pad=False)
    rough = torch.abs(x - smooth)
    curvature = torch.zeros_like(x)
    curvature[..., 1:-1] = x[..., 2:] - 2.0 * x[..., 1:-1] + x[..., :-2]
    return torch.cat([x, grad, rough, curvature], dim=1)


class DMCNet1D(nn.Module):
    """一维测井 SR 的动态多尺度上下文网络。

    并行的局部、中程和宽上下文算子由数据相关的门控混合, 是同协议下的 DMC 实现。
    """

    def __init__(self, channels: int = 64, depth: int = 4, morphology: bool = False):
        super().__init__()
        self.morphology = morphology
        self.inp = nn.Sequential(nn.Conv1d(4 if morphology else 1, channels, 5, padding=2), nn.GELU())
        self.blocks = nn.ModuleList([DynamicMultiScaleBlock(channels, i) for i in range(depth)])
        self.out = nn.Conv1d(channels, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(morphology_channels(x) if self.morphology else x)
        for block in self.blocks:
            h = block(h)
        return x + self.out(h)


class DynamicMultiScaleBlock(nn.Module):
    def __init__(self, channels: int, layer: int):
        super().__init__()
        dilation = 2 ** (layer % 4)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1), nn.GELU()),
                nn.Sequential(nn.Conv1d(channels, channels, 5, padding=2), nn.GELU()),
                nn.Sequential(
                    nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
                    nn.GELU(),
                ),
                nn.Sequential(nn.Conv1d(channels, channels, 9, padding=4), nn.GELU()),
            ]
        )
        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GELU(),
            nn.Conv1d(channels, len(self.branches), 1),
        )
        self.mix = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(8 if channels >= 8 else 1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.gate(x), dim=1)
        ys = torch.stack([branch(x) for branch in self.branches], dim=1)
        y = torch.sum(weights.unsqueeze(2) * ys, dim=1)
        return self.norm(x + self.mix(y))


class SpikeEnhancement(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )
        self.gate = nn.Sequential(nn.Conv1d(channels * 2, channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smooth = nn.functional.avg_pool1d(x, kernel_size=9, stride=1, padding=4, count_include_pad=False)
        detail = x - smooth
        enhanced = self.net(detail)
        return x + self.gate(torch.cat([x, detail], dim=1)) * enhanced


class CascadedSRNet1D(nn.Module):
    """带尖峰增强的全局-局部级联测井 SR 网络。"""

    def __init__(self, channels: int = 64, depth: int = 4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, channels, 5, padding=2), nn.GELU())
        self.global_branch = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels, channels, 3, padding=2 ** (i % 4), dilation=2 ** (i % 4)),
                    nn.GELU(),
                    nn.Conv1d(channels, channels, 1),
                )
                for i in range(depth)
            ]
        )
        self.local_branch = nn.ModuleList([SpikeEnhancement(channels) for _ in range(depth)])
        self.fuse = nn.Sequential(nn.Conv1d(channels * 2, channels, 1), nn.GELU())
        self.out = nn.Conv1d(channels, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        g = h
        l = h
        for global_block, local_block in zip(self.global_branch, self.local_branch, strict=True):
            g = g + global_block(g)
            l = local_block(l)
        h = self.fuse(torch.cat([g, l], dim=1))
        return x + self.out(h)


class GatedLongConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.long = nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels)
        self.local = nn.Conv1d(channels, channels, 3, padding=1)
        self.gate = nn.Sequential(nn.Conv1d(channels * 2, channels, 1), nn.Sigmoid())
        self.ffn = nn.Sequential(nn.Conv1d(channels, channels * 2, 1), nn.GELU(), nn.Conv1d(channels * 2, channels, 1))
        self.norm = nn.GroupNorm(8 if channels >= 8 else 1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        long = self.long(x)
        local = self.local(x)
        y = self.gate(torch.cat([long, local], dim=1)) * long + local
        return self.norm(x + self.ffn(y))


class StateSpaceSR1D(nn.Module):
    """受状态空间模型启发的长上下文基线。

    不依赖外部 Mamba 实现, 保留门控长程序列混合加局部细节恢复的核心比较压力。
    """

    def __init__(self, channels: int = 64, depth: int = 4, morphology: bool = False):
        super().__init__()
        self.morphology = morphology
        self.inp = nn.Sequential(nn.Conv1d(4 if morphology else 1, channels, 5, padding=2), nn.GELU())
        kernels = [17, 33, 65, 33]
        self.blocks = nn.ModuleList([GatedLongConvBlock(channels, kernels[i % len(kernels)]) for i in range(depth)])
        self.out = nn.Conv1d(channels, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(morphology_channels(x) if self.morphology else x)
        for block in self.blocks:
            h = block(h)
        return x + self.out(h)


def build_model(name: str, channels: int = 64, depth: int = 4, modes: int = 64) -> nn.Module:
    """按共享协议重新实现的单曲线对比方法。"""
    name = name.lower()
    if name == "lstm":
        return BiLSTMNet(channels, depth)
    if name == "transformer":
        return TransformerSR(channels, depth)
    if name == "dmcnet":
        return DMCNet1D(channels, depth)
    if name == "cascade_sr":
        return CascadedSRNet1D(channels, depth)
    if name == "ssm":
        return StateSpaceSR1D(channels, depth)
    raise ValueError(f"unknown model: {name}")
