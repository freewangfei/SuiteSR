"""SuiteSR: 垂向退化测井曲线的引导式、无幅度重建。

输入布局见 wellsr.data.build_multicurve_windows: 第 0 行为插值目标观测, 第 1..n_aux 行为
插值引导观测 (缺失为零), 第 1+n_aux..2*n_aux 行为 0/1 存在掩码, 启用时最后一行为目标采样网格掩码。

本模块包含 SuiteSR 家族 (一种形态编码, 三种骨干) 及对比方法的引导形式:
fusion stem (MultiInput)、加宽首层 (MultiInputNative) 和接收同一编码的加宽首层
(MorphologyNative, 即 SuiteSR-SSM 与 SuiteSR-DMC 骨干)。
"""

from __future__ import annotations

import torch
from torch import nn

from .models import (
    BiLSTMNet,
    CascadedSRNet1D,
    DMCNet1D,
    GatedLongConvBlock,
    StateSpaceSR1D,
    TransformerSR,
)


def shape_channels(x: torch.Tensor, keep: tuple = ("grad", "rough", "curv"), ma: int = 9) -> torch.Tensor:
    """无幅度形态描述子: 一阶差分、|x - MA_ma|、二阶差分。

    keep 选择描述子子集 (逐描述子消融), ma 为粗糙度通道的滑动平均窗口 (窗口敏感性)。
    """
    out = []
    if "grad" in keep:
        grad = torch.zeros_like(x)
        grad[..., 1:] = x[..., 1:] - x[..., :-1]
        out.append(grad)
    if "rough" in keep:
        k = max(3, int(ma) | 1)
        smooth = nn.functional.avg_pool1d(x, kernel_size=k, stride=1, padding=k // 2, count_include_pad=False)
        out.append(torch.abs(x - smooth))
    if "curv" in keep:
        curv = torch.zeros_like(x)
        curv[..., 1:-1] = x[..., 2:] - 2.0 * x[..., 1:-1] + x[..., :-2]
        out.append(curv)
    return torch.cat(out, dim=1)


class FilterBankBlock(nn.Module):
    """局部 (空洞 3 抽头)、趋势 (9 抽头) 与薄层 (高通 3 抽头) 三个算子
    按固定权重平均, 残差连接后做组归一化。"""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation), nn.GELU(),
            nn.Conv1d(channels, channels, 1))
        self.trend = nn.Sequential(nn.Conv1d(channels, channels, 9, padding=4), nn.GELU(),
                                   nn.Conv1d(channels, channels, 1))
        self.thin = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1), nn.GELU(),
                                  nn.Conv1d(channels, channels, 1))
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        hp = h - nn.functional.avg_pool1d(h, 9, stride=1, padding=4, count_include_pad=False)
        mix = (self.local(h) + self.trend(h) + self.thin(hp)) / 3.0
        return self.norm(h + mix)


class PlainBlock(nn.Module):
    """滤波器组的对照: 同宽度的单个残差 3 抽头卷积加组归一化,
    用于将三算子块与普通块比较。"""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation), nn.GELU(),
                                  nn.Conv1d(channels, channels, 1))
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.norm(h + self.conv(h))


class GeoSRv2(nn.Module):
    def __init__(self, channels: int = 32, depth: int = 3, in_channels: int = 7,
                 morphology: bool = True, use_aux: bool = True, recurrent: bool = True,
                 deep_fusion: bool = False, guide_dropout: float = 0.0,
                 target_identity_channels: int = 0, morphology_aux: bool | None = None,
                 block: str = "filterbank", descriptors: tuple = ("grad", "rough", "curv"), ma_window: int = 9,
                 include_raw: bool = False, long_context: bool = False,
                 target_observation_mask: bool = False):
        super().__init__()
        self.deep_fusion = deep_fusion and use_aux
        # 引导丢弃: 训练时以该概率将整个窗口的引导输入 (值与掩码) 置零,
        # 使模型也能仅凭目标工作, 不在引导信息少的任务 (如 2x) 上过度依赖引导。
        self.guide_dropout = guide_dropout
        self.target_identity_channels = max(0, int(target_identity_channels))
        self.target_observation_mask = bool(target_observation_mask)
        payload = in_channels - 1 - self.target_identity_channels - int(self.target_observation_mask)
        if payload < 0 or (payload > 0 and payload % 2):
            raise ValueError(
                f"in_channels={in_channels} is incompatible with "
                f"target_identity_channels={self.target_identity_channels}"
            )
        self.n_aux = payload // 2 if payload > 0 else 0
        self.morphology = morphology
        self.include_raw = bool(include_raw)
        # 引导曲线可与目标独立选择是否编码 (消融)
        self.morphology_aux = morphology if morphology_aux is None else morphology_aux
        self.use_aux = use_aux and self.n_aux > 0
        self.recurrent = recurrent
        self.long_context = bool(long_context)
        self.descriptors = tuple(descriptors); self.ma_window = int(ma_window)
        f = (1 + len(self.descriptors)) if (morphology and self.include_raw) else (len(self.descriptors) if morphology else 1)
        fa = (1 + len(self.descriptors)) if (self.morphology_aux and self.include_raw) else (len(self.descriptors) if self.morphology_aux else 1)
        self.stem = nn.Sequential(
            nn.Conv1d(
                f + self.target_identity_channels + int(self.target_observation_mask),
                channels,
                5,
                padding=2,
            ),
            nn.GELU(),
        )
        if self.use_aux:
            # 辅助曲线经独立主干进入; 由两路特征计算的逐位置门控决定注入量, 可关掉无信息曲线。
            self.aux_stem = nn.Sequential(nn.Conv1d(fa * self.n_aux + self.n_aux, channels, 5, padding=2), nn.GELU())
            self.aux_gate = nn.Sequential(nn.Conv1d(2 * channels, channels, 1), nn.GELU(),
                                          nn.Conv1d(channels, channels, 1), nn.Sigmoid())
        Block = {"filterbank": FilterBankBlock, "plain": PlainBlock}[block]
        self.blocks = nn.ModuleList([Block(channels, 2 ** (b % 4)) for b in range(depth)])
        if self.deep_fusion:
            # 辅助流经自身滤波器组处理, 在每个块后门控注入目标流。
            self.aux_blocks = nn.ModuleList([FilterBankBlock(channels, 2 ** (b % 4)) for b in range(depth)])
            self.block_gates = nn.ModuleList([nn.Sequential(nn.Conv1d(2 * channels, channels, 1), nn.GELU(),
                                                            nn.Conv1d(channels, channels, 1), nn.Sigmoid())
                                              for _ in range(depth)])
        if self.long_context:
            # 长程分支: 目标丢失大部分长程包络时该归纳偏置最有用,
            # 作为独立分支经可学习残差门控并入主流, 而非替换局部滤波器组。
            kernels = (17, 33, 65, 33)
            self.long_blocks = nn.ModuleList(
                [GatedLongConvBlock(channels, kernels[i % len(kernels)]) for i in range(max(2, depth))]
            )
            self.long_gate = nn.Sequential(
                nn.Conv1d(2 * channels, channels, 1),
                nn.GELU(),
                nn.Conv1d(channels, 1, 1),
                nn.Sigmoid(),
            )
            self.long_scale = nn.Parameter(torch.tensor(0.10))
        else:
            self.long_blocks = None
            self.long_gate = None
            self.long_scale = None
        if recurrent:
            self.lstm = nn.LSTM(channels, channels // 2, batch_first=True, bidirectional=True)
            self.alpha = nn.Sequential(nn.Conv1d(2 * channels, channels, 1), nn.Sigmoid())
        self.head = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1), nn.GELU(),
                                  nn.Conv1d(channels, 1, 3, padding=1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target = x[:, :1]
        feats = self._curve_features(target, self.morphology) if self.morphology else target
        if self.target_observation_mask:
            observation_mask = x[:, 1 + 2 * self.n_aux:2 + 2 * self.n_aux]
            feats = torch.cat([feats, observation_mask], dim=1)
        if self.target_identity_channels:
            identity_start = 1 + 2 * self.n_aux + int(self.target_observation_mask)
            identity = x[:, identity_start:]
            feats = torch.cat([feats, identity], dim=1)
        h = self.stem(feats)
        if self.use_aux:
            aux = x[:, 1:1 + self.n_aux]
            mask = x[:, 1 + self.n_aux:1 + 2 * self.n_aux]
            if self.training and self.guide_dropout > 0:
                keep = (torch.rand(x.shape[0], 1, 1, device=x.device) >= self.guide_dropout).to(x.dtype)
                aux = aux * keep
                mask = mask * keep
            aux_feats = torch.cat([self._curve_features(aux, self.morphology_aux) if self.morphology_aux else aux, mask], dim=1)
            ha = self.aux_stem(aux_feats)
            g = self.aux_gate(torch.cat([h, ha], dim=1))
            h = h + g * ha
        for b, block in enumerate(self.blocks):
            h = block(h)
            if self.deep_fusion:
                ha = self.aux_blocks[b](ha)
                h = h + self.block_gates[b](torch.cat([h, ha], dim=1)) * ha
        if self.long_blocks is not None and self.long_gate is not None and self.long_scale is not None:
            h_long = h
            for block in self.long_blocks:
                h_long = block(h_long)
            delta = h_long - h
            h = h + torch.clamp(self.long_scale, 0.0, 0.5) * self.long_gate(
                torch.cat([h, h_long], dim=1)
            ) * delta
        if self.recurrent:
            z, _ = self.lstm(h.transpose(1, 2))
            z = z.transpose(1, 2)
            h = h + self.alpha(torch.cat([h, z], dim=1)) * z
        return target + self.head(h)

    def _curve_features(self, x: torch.Tensor, use_morphology: bool) -> torch.Tensor:
        if not use_morphology:
            return x
        morph = shape_channels(x, self.descriptors, self.ma_window)
        return torch.cat([x, morph], dim=1) if self.include_raw else morph


class MultiInputNative(nn.Module):
    """加宽首层: 让基线原生接收多曲线输入, 首层加宽以读取全部行 (目标、引导、掩码),
    残差取在目标上。没有任何一路被压缩到单通道, 是各对比方法最强的引导形式;
    支持前向路径为 stem -> blocks -> head 或 LSTM -> head 的架构。"""

    def __init__(self, base: nn.Module, in_channels: int):
        super().__init__()
        self.base = base
        if isinstance(base, BiLSTMNet):
            old = base.lstm
            base.lstm = nn.LSTM(in_channels, old.hidden_size, num_layers=old.num_layers,
                                batch_first=True, bidirectional=True)
            self.kind = "lstm"
        elif isinstance(base, (DMCNet1D, StateSpaceSR1D)):
            old = base.inp[0]
            new = nn.Conv1d(in_channels, old.out_channels, old.kernel_size, padding=old.padding)
            with torch.no_grad():
                new.weight.zero_(); new.weight[:, :1] = old.weight; new.bias.copy_(old.bias)
            base.inp[0] = new
            self.kind = "conv"
        elif isinstance(base, TransformerSR):
            old = base.inp[0]
            new = nn.Conv1d(in_channels, old.out_channels, old.kernel_size, padding=old.padding)
            with torch.no_grad():
                new.weight.zero_(); new.weight[:, :1] = old.weight; new.bias.copy_(old.bias)
            base.inp[0] = new
            self.kind = "transformer"
        elif isinstance(base, CascadedSRNet1D):
            old = base.stem[0]
            new = nn.Conv1d(in_channels, old.out_channels, old.kernel_size, padding=old.padding)
            with torch.no_grad():
                new.weight.zero_(); new.weight[:, :1] = old.weight; new.bias.copy_(old.bias)
            base.stem[0] = new
            self.kind = "cascade"
        else:
            raise ValueError("native multi-input is implemented for BiLSTM, Transformer, DMC-Net, Cascade-SR and the state-space model")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target = x[:, :1]
        if self.kind == "lstm":
            y, _ = self.base.lstm(x.transpose(1, 2))
            return target + self.base.head(y).transpose(1, 2)
        if self.kind == "transformer":
            b = self.base
            h = b.inp(x).transpose(1, 2)
            h = h + b.pos_encoding[: h.shape[1]].unsqueeze(0).to(h.dtype)
            h = b.encoder(h).transpose(1, 2)
            return target + b.out(h)
        if self.kind == "cascade":
            b = self.base
            h = b.stem(x); g = h; l = h
            for global_block, local_block in zip(b.global_branch, b.local_branch, strict=True):
                g = g + global_block(g); l = local_block(l)
            return target + b.out(b.fuse(torch.cat([g, l], dim=1)))
        h = self.base.inp(x)
        for block in self.base.blocks:
            h = block(h)
        return target + self.base.out(h)


class MorphologyNative(nn.Module):
    """接收 SuiteSR 输入编码的加宽首层对比方法: 目标与每条引导曲线以三个无幅度
    形态描述子 (加引导掩码) 输入, 首层加宽以读取全部通道, 残差取在原始目标上。
    通过把编码交给其他架构 (SuiteSR-SSM、SuiteSR-DMC), 将编码与骨干分离。"""

    def __init__(self, base: nn.Module, in_channels: int):
        super().__init__()
        self.n_aux = (in_channels - 1) // 2
        feat = 3 * (1 + self.n_aux) + self.n_aux
        self.inner = MultiInputNative(base, feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target = x[:, :1]
        if self.n_aux:
            aux = x[:, 1:1 + self.n_aux]
            mask = x[:, 1 + self.n_aux:1 + 2 * self.n_aux]
            guide_features = shape_channels(aux)
        else:
            aux = None
            mask = None
            guide_features = x.new_zeros(x.shape[0], 0, x.shape[-1])
        feats = torch.cat([shape_channels(target), guide_features, mask] if mask is not None else [shape_channels(target)], dim=1)
        inner = self.inner
        if inner.kind == "lstm":
            y, _ = inner.base.lstm(feats.transpose(1, 2))
            return target + inner.base.head(y).transpose(1, 2)
        h = inner.base.inp(feats)
        for block in inner.base.blocks:
            h = block(h)
        return target + inner.base.out(h)


class _OnesGate(nn.Module):
    """将引导门控替换为常数 1: 引导流不加权直接相加。"""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        c = z.shape[1] // 2
        return torch.ones_like(z[:, :c])


class MultiInput(nn.Module):
    """fusion stem: 单曲线基线的多曲线输入形式。

    5 抽头融合主干把目标观测、辅助观测及其掩码映射为一条通道, 加到目标观测上
    再送入未改动的基线, 残差取在目标上。
    """

    def __init__(self, base: nn.Module, in_channels: int, channels: int):
        super().__init__()
        self.base = base
        self.fuse = nn.Sequential(nn.Conv1d(in_channels, channels, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(channels, 1, 5, padding=2))
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target = x[:, :1]
        return self.base(target + self.fuse(x))


def build_model_v2(name: str, channels: int = 32, depth: int = 3, in_channels: int = 1,
                   scale: int = 4, target_identity_channels: int = 0,
                   target_observation_mask: bool = False) -> nn.Module:
    """按配置文件中的模型名构建模型。"""
    name = name.lower()
    if name == "geosr2":
        return GeoSRv2(channels, depth, in_channels, target_identity_channels=target_identity_channels,
                       target_observation_mask=target_observation_mask)
    if name == "geosr2_long":
        return GeoSRv2(channels, depth, in_channels, target_identity_channels=target_identity_channels,
                       target_observation_mask=target_observation_mask, long_context=True)
    if name == "geosr2_obs_mask":
        return GeoSRv2(channels, depth, in_channels, target_identity_channels=target_identity_channels,
                       target_observation_mask=True, long_context=True)
    if name == "geosr2_gd":
        return GeoSRv2(channels, depth, in_channels, guide_dropout=0.3)
    if name == "geosr2_gd5":
        return GeoSRv2(channels, depth, in_channels, guide_dropout=0.5)
    if name == "geosr2_deep":
        return GeoSRv2(channels, depth, in_channels, deep_fusion=True)
    if name == "geosr2_nomorph":
        return GeoSRv2(channels, depth, in_channels, morphology=False)
    if name == "geosr2_norecur":
        return GeoSRv2(channels, depth, in_channels, recurrent=False)
    if name == "geosr2_plainblock":
        return GeoSRv2(channels, depth, in_channels, block="plain")
    if name == "geosr2_morphtarget":
        return GeoSRv2(channels, depth, in_channels, morphology=True, morphology_aux=False)
    if name == "geosr2_morphguide":
        return GeoSRv2(channels, depth, in_channels, morphology=False, morphology_aux=True)
    if name == "geosr2_nogate":
        m = GeoSRv2(channels, depth, in_channels)
        m.aux_gate = _OnesGate()
        return m
    if name.startswith("geosr2_drop"):
        drop = name.split("geosr2_drop")[1]
        keep = tuple(d for d in ("grad", "rough", "curv") if d != drop)
        return GeoSRv2(channels, depth, in_channels, descriptors=keep)
    if name.startswith("geosr2_ma"):
        return GeoSRv2(channels, depth, in_channels, ma_window=int(name.split("geosr2_ma")[1]))
    bases = {
        "lstm": lambda: BiLSTMNet(channels, depth),
        "transformer": lambda: TransformerSR(channels, depth),
        "dmcnet": lambda: DMCNet1D(channels, depth),
        "cascade_sr": lambda: CascadedSRNet1D(channels, depth),
        "ssm": lambda: StateSpaceSR1D(channels, depth),
    }
    if name.endswith("_morph_native"):
        return MorphologyNative(bases[name[:-13]](), in_channels)
    if name.endswith("_native"):
        return MultiInputNative(bases[name[:-7]](), in_channels)
    if name in bases:
        base = bases[name]()
        return MultiInput(base, in_channels, channels) if in_channels > 1 else base
    raise ValueError(f"unknown model: {name}")
