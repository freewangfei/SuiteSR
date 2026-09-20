from __future__ import annotations

import torch
from torch import nn


def gradient_1d(x: torch.Tensor) -> torch.Tensor:
    return x[..., 1:] - x[..., :-1]


def curvature_1d(x: torch.Tensor) -> torch.Tensor:
    return x[..., 2:] - 2.0 * x[..., 1:-1] + x[..., :-2]


def spectral_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pf = torch.abs(torch.fft.rfft(pred, dim=-1))
    tf = torch.abs(torch.fft.rfft(target, dim=-1))
    return torch.mean(torch.abs(pf - tf))


def spectral_angle_loss(
    pred: torch.Tensor, target: torch.Tensor, high_weight: float = 0.0, exclude_dc: bool = True
) -> torch.Tensor:
    """幅度谱夹角损失, 排除 DC 分量。"""
    pf = torch.abs(torch.fft.rfft(pred, dim=-1))
    tf = torch.abs(torch.fft.rfft(target, dim=-1))
    if exclude_dc:
        pf = pf[..., 1:]
        tf = tf[..., 1:]
    if high_weight:
        freq = torch.linspace(0.0, 1.0, pf.shape[-1], device=pf.device, dtype=pf.dtype)
        weight = 1.0 + high_weight * freq[None, None, :]
        pf = pf * weight
        tf = tf * weight
    dot = torch.sum(pf * tf, dim=-1)
    denom = torch.linalg.vector_norm(pf, dim=-1) * torch.linalg.vector_norm(tf, dim=-1)
    cos = torch.clamp(dot / (denom + 1e-8), -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.mean(torch.acos(cos))


def spectral_shape_l1(pred: torch.Tensor, target: torch.Tensor, high_weight: float = 0.0) -> torch.Tensor:
    pf = torch.abs(torch.fft.rfft(pred, dim=-1))
    tf = torch.abs(torch.fft.rfft(target, dim=-1))
    if high_weight:
        freq = torch.linspace(0.0, 1.0, pf.shape[-1], device=pf.device, dtype=pf.dtype)
        weight = 1.0 + high_weight * freq[None, None, :]
        pf = pf * weight
        tf = tf * weight
    pf = pf / (torch.mean(pf, dim=-1, keepdim=True) + 1e-6)
    tf = tf / (torch.mean(tf, dim=-1, keepdim=True) + 1e-6)
    return torch.mean(torch.abs(pf - tf))


def spectral_gradient_angle_loss(pred: torch.Tensor, target: torch.Tensor, high_weight: float = 0.0) -> torch.Tensor:
    return spectral_angle_loss(gradient_1d(pred), gradient_1d(target), high_weight=high_weight)


def super_nyquist_component(x: torch.Tensor, scale: int) -> torch.Tensor:
    """返回高于目标观测奈奎斯特频率的分量。"""
    scale = max(1, int(scale))
    if scale == 1:
        return x
    spectrum = torch.fft.rfft(x, dim=-1)
    frequencies = torch.fft.rfftfreq(x.shape[-1], device=x.device, dtype=x.dtype)
    keep = frequencies > 0.5 / float(scale)
    return torch.fft.irfft(torch.where(keep, spectrum, torch.zeros_like(spectrum)), n=x.shape[-1], dim=-1)


def missing_band_component(x: torch.Tensor, scale: int) -> torch.Tensor:
    """返回 scale/2 引导曲线可观测但 scale 下缺失的频带。"""
    scale = max(1, int(scale))
    if scale <= 1:
        return torch.zeros_like(x)
    spectrum = torch.fft.rfft(x, dim=-1)
    frequencies = torch.fft.rfftfreq(x.shape[-1], device=x.device, dtype=x.dtype)
    keep = (frequencies > 0.5 / float(scale)) & (frequencies <= 1.0 / float(scale) + 1e-7)
    return torch.fft.irfft(torch.where(keep, spectrum, torch.zeros_like(spectrum)), n=x.shape[-1], dim=-1)


def super_nyquist_angle_loss(pred: torch.Tensor, target: torch.Tensor, scale: int) -> torch.Tensor:
    """只作用于 LR 缺失信息的谱角损失。"""
    pred_hf = super_nyquist_component(pred, scale)
    target_hf = super_nyquist_component(target, scale)
    return spectral_angle_loss(pred_hf, target_hf)


def vertical_response_consistency(
    pred: torch.Tensor,
    observed: torch.Tensor,
    kernel_size: int = 17,
    stride: int = 8,
) -> torch.Tensor:
    if kernel_size <= 1:
        pred_response = pred
        obs_response = observed
    else:
        pad = kernel_size // 2
        pred_response = nn.functional.avg_pool1d(
            pred, kernel_size=kernel_size, stride=1, padding=pad, count_include_pad=False
        )
        obs_response = nn.functional.avg_pool1d(
            observed, kernel_size=kernel_size, stride=1, padding=pad, count_include_pad=False
        )
    if stride > 1:
        pred_response = pred_response[..., ::stride]
        obs_response = obs_response[..., ::stride]
    return torch.mean(torch.abs(pred_response - obs_response))


def gaussian_decimation_consistency(
    pred: torch.Tensor,
    observed: torch.Tensor,
    scale: int,
    sigma_factor: float = 1.0,
) -> torch.Tensor:
    """将预测曲线经固定高斯核退化后, 与按 ::scale 采样得到的原始观测样本比较。"""
    scale = max(1, int(scale))
    if scale == 1:
        return torch.mean(torch.abs(pred - observed))
    sigma = max(0.6, scale / 2.0) * float(sigma_factor)
    radius = max(1, int(torch.ceil(torch.tensor(3.0 * sigma)).item()))
    offsets = torch.arange(-radius, radius + 1, device=pred.device, dtype=pred.dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / torch.sum(kernel)
    padded = nn.functional.pad(pred, (radius, radius), mode="reflect")
    filtered = nn.functional.conv1d(padded, kernel.view(1, 1, -1), groups=1)
    return torch.mean(torch.abs(filtered[..., ::scale] - observed[..., ::scale]))


class SRLoss(nn.Module):
    def __init__(
        self,
        l1: float = 1.0,
        mse: float = 0.0,
        gradient: float = 0.2,
        gradient_mse: float = 0.0,
        aux_gradient: float = 0.0,
        aux_integral: float = 0.0,
        flat_gradient: float = 0.0,
        flat_quantile: float = 0.55,
        curvature: float = 0.0,
        spectral: float = 0.05,
        spectral_shape: float = 0.0,
        spectral_angle: float = 0.0,
        spectral_gradient_angle: float = 0.0,
        hf_l1: float = 0.0,
        hf_spectral: float = 0.0,
        band_residual: float = 0.0,
        hf_scale: int = 1,
        spectral_angle_high_weight: float = 0.0,
        tool_response: float = 0.0,
        tool_response_kernel: int = 17,
        tool_response_stride: int = 8,
        data_consistency: float = 0.0,
        data_consistency_scale: int = 1,
        data_consistency_sigma_factor: float = 1.0,
        boundary_gradient: float = 0.0,
        boundary_power: float = 1.0,
    ):
        super().__init__()
        self.l1 = l1
        self.mse = mse
        self.gradient = gradient
        self.gradient_mse = gradient_mse
        self.aux_gradient = aux_gradient
        self.aux_integral = aux_integral
        self.flat_gradient = flat_gradient
        self.flat_quantile = flat_quantile
        self.curvature = curvature
        self.spectral = spectral
        self.spectral_shape = spectral_shape
        self.spectral_angle = spectral_angle
        self.spectral_gradient_angle = spectral_gradient_angle
        self.hf_l1 = hf_l1
        self.hf_spectral = hf_spectral
        self.band_residual = band_residual
        self.hf_scale = max(1, int(hf_scale))
        self.spectral_angle_high_weight = spectral_angle_high_weight
        self.tool_response = tool_response
        self.tool_response_kernel = int(tool_response_kernel)
        self.tool_response_stride = int(tool_response_stride)
        self.data_consistency = data_consistency
        self.data_consistency_scale = max(1, int(data_consistency_scale))
        self.data_consistency_sigma_factor = data_consistency_sigma_factor
        self.boundary_gradient = boundary_gradient
        self.boundary_power = boundary_power
        self.base = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor | dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        aux = pred if isinstance(pred, dict) else {}
        if isinstance(pred, dict):
            pred = pred["curve"]
        loss = self.l1 * self.base(pred, target)
        if self.mse:
            loss = loss + self.mse * self.mse_loss(pred, target)
        if self.gradient:
            loss = loss + self.gradient * self.base(gradient_1d(pred), gradient_1d(target))
        if self.gradient_mse:
            loss = loss + self.gradient_mse * self.mse_loss(gradient_1d(pred), gradient_1d(target))
        if self.flat_gradient:
            pred_grad = gradient_1d(pred)
            target_grad = gradient_1d(target)
            mag = torch.abs(target_grad).detach()
            threshold = torch.quantile(mag.flatten(1), self.flat_quantile, dim=1, keepdim=True).unsqueeze(1)
            flat_weight = (mag <= threshold).to(pred.dtype)
            loss = loss + self.flat_gradient * torch.mean(flat_weight * torch.abs(pred_grad - target_grad))
        if self.aux_gradient and "grad_residual" in aux:
            target_residual_grad = gradient_1d(target) - gradient_1d(aux.get("input", torch.zeros_like(target)))
            loss = loss + self.aux_gradient * self.base(aux["grad_residual"][..., 1:], target_residual_grad)
        if self.aux_integral and "grad_integral" in aux:
            target_residual = target - aux.get("input", torch.zeros_like(target))
            target_residual = target_residual - torch.mean(target_residual, dim=-1, keepdim=True)
            loss = loss + self.aux_integral * self.base(aux["grad_integral"], target_residual)
        if self.curvature:
            loss = loss + self.curvature * self.base(curvature_1d(pred), curvature_1d(target))
        if self.boundary_gradient:
            pred_grad = gradient_1d(pred)
            target_grad = gradient_1d(target)
            weights = torch.abs(target_grad)
            weights = weights / (torch.mean(weights, dim=-1, keepdim=True) + 1e-6)
            weights = torch.clamp(weights, 0.25, 6.0).pow(self.boundary_power)
            loss = loss + self.boundary_gradient * torch.mean(weights * torch.abs(pred_grad - target_grad))
        if self.spectral:
            loss = loss + self.spectral * spectral_l1(pred, target)
        if self.spectral_shape:
            loss = loss + self.spectral_shape * spectral_shape_l1(
                pred, target, high_weight=self.spectral_angle_high_weight
            )
        if self.spectral_angle:
            loss = loss + self.spectral_angle * spectral_angle_loss(
                pred, target, high_weight=self.spectral_angle_high_weight
            )
        if self.spectral_gradient_angle:
            loss = loss + self.spectral_gradient_angle * spectral_gradient_angle_loss(
                pred, target, high_weight=self.spectral_angle_high_weight
            )
        if self.hf_l1:
            loss = loss + self.hf_l1 * self.base(
                super_nyquist_component(pred, self.hf_scale),
                super_nyquist_component(target, self.hf_scale),
            )
        if self.hf_spectral:
            loss = loss + self.hf_spectral * super_nyquist_angle_loss(
                pred, target, self.hf_scale
            )
        if self.band_residual and "band_residual" in aux:
            observed = aux.get("input", None)
            if observed is not None:
                # 提议分支以归一化目标残差的缺失分量为监督; 除以目标频带尺度使该辅助目标跨曲线可比。
                residual_target = missing_band_component(target - observed[:, :1], self.hf_scale)
                scale = residual_target.std(dim=-1, keepdim=True).detach() + 1e-4
                proposal = aux["band_residual"]
                loss = loss + self.band_residual * self.base(proposal / scale, residual_target / scale)
        if self.tool_response:
            observed = aux.get("input", None)
            if observed is not None:
                loss = loss + self.tool_response * vertical_response_consistency(
                    pred,
                    observed,
                    kernel_size=self.tool_response_kernel,
                    stride=self.tool_response_stride,
                )
        if self.data_consistency:
            observed = aux.get("input", None)
            if observed is not None:
                loss = loss + self.data_consistency * gaussian_decimation_consistency(
                    pred,
                    observed[:, :1],
                    scale=self.data_consistency_scale,
                    sigma_factor=self.data_consistency_sigma_factor,
                )
        return loss
