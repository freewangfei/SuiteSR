from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import SRWindow
from .metrics import compute_metrics
from .models import build_model
from .losses import (
    SRLoss,
    curvature_1d,
    gradient_1d,
    spectral_angle_loss,
    spectral_gradient_angle_loss,
    spectral_l1,
    spectral_shape_l1,
    vertical_response_consistency,
    gaussian_decimation_consistency,
    missing_band_component,
)


@dataclass
class TrainResult:
    model_name: str
    best_val: float
    history: list[dict[str, float]]
    val_predictions: list[np.ndarray]
    predictions: list[np.ndarray]
    train_predictions: list[np.ndarray] | None = None
    model: object | None = None  # 训练后的模块, 用于对其他长度的窗口推理
    normalization_stats: dict[tuple[str, str], tuple[float, float]] | None = None
    normalization: str = "window"
    normalization_local_weight: float = 0.75


def resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def fit_normalization_stats(
    windows: list[SRWindow],
    *,
    robust: bool = False,
) -> dict[tuple[str, str], tuple[float, float]]:
    """仅从训练窗口估计固定的 dataset/curve 统计量; robust 模式先按井汇总再取跨井中位数。"""
    values: dict[tuple[str, str], list[np.ndarray]] = {}
    for w in windows:
        values.setdefault((w.dataset, w.curve), []).append(w.hr.astype(np.float32))
        if w.aux_hr is not None and w.aux_curves is not None:
            for curve, arr in zip(w.aux_curves, w.aux_hr, strict=True):
                values.setdefault((w.dataset, curve), []).append(arr.astype(np.float32))
    if not robust:
        stats = {}
        for key, chunks in values.items():
            flat = np.concatenate(chunks)
            stats[key] = (float(np.mean(flat)), float(np.std(flat) + 1e-6))
        return stats

    by_well: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    for w in windows:
        key = (w.dataset, w.curve)
        by_well.setdefault(key, {}).setdefault(w.well, []).append(w.hr.astype(np.float32))
        if w.aux_hr is not None and w.aux_curves is not None:
            for curve, arr in zip(w.aux_curves, w.aux_hr, strict=True):
                aux_key = (w.dataset, curve)
                by_well.setdefault(aux_key, {}).setdefault(w.well, []).append(arr.astype(np.float32))
    stats = {}
    for key, wells in by_well.items():
        locations = [float(np.mean(np.concatenate(chunks))) for chunks in wells.values()]
        scales = [float(np.std(np.concatenate(chunks))) for chunks in wells.values()]
        stats[key] = (float(np.median(locations)), float(np.median(scales) + 1e-6))
    return stats


def windows_to_tensors(
    windows: list[SRWindow],
    device,
    normalization_stats: dict[tuple[str, str], tuple[float, float]] | None = None,
    normalization: str = "window",
    normalization_local_weight: float = 0.75,
):
    import torch

    if normalization not in {"window", "target_shared", "global_train", "robust_train", "hybrid_train"}:
        raise ValueError(f"unknown normalization mode: {normalization}")
    if not 0.0 <= normalization_local_weight <= 1.0:
        raise ValueError("normalization_local_weight must be in [0, 1]")
    # 传入固定统计量但未指定模式时视为 global_train。
    effective_normalization = (
        "global_train"
        if normalization == "window" and normalization_stats is not None
        else normalization
    )

    def stats_for_row(
        w: SRWindow,
        curve: str | None,
        row: np.ndarray,
    ) -> tuple[float, float]:
        local = (float(np.mean(row)), float(np.std(row) + 1e-6))
        if curve is None or effective_normalization not in {"global_train", "robust_train", "hybrid_train"}:
            return local
        prior = (normalization_stats or {}).get((w.dataset, curve), local)
        if effective_normalization == "hybrid_train":
            alpha = float(normalization_local_weight)
            return (
                alpha * local[0] + (1.0 - alpha) * float(prior[0]),
                alpha * local[1] + (1.0 - alpha) * float(prior[1]),
            )
        return float(prior[0]), float(prior[1])

    lr_rows = []
    hr_rows = []
    means = []
    stds = []
    for w in windows:
        lr_np = w.lr.astype(np.float32)
        hr_np = w.hr.astype(np.float32)
        if lr_np.ndim == 2:
            # 多曲线窗口: 每条真实曲线行按自身统计量归一化, 0/1 掩码行保持不变;
            # 目标为第 0 行, 其统计量用于反变换预测。
            k = lr_np.shape[0]
            n_identity = len(w.target_identity or ())
            has_observation_mask = w.target_observation_mask is not None
            extra_mask = 1 if has_observation_mask else 0
            if n_identity or has_observation_mask:
                payload = k - 1 - n_identity - extra_mask
                if payload < 0 or payload % 2:
                    raise ValueError(
                        f"invalid multi-curve layout for {w.dataset}/{w.curve}: "
                        f"shape={lr_np.shape}, target_identity={w.target_identity!r}"
                    )
                n_aux = payload // 2
            else:
                # 无元数据的多行输入按默认布局解释。
                n_aux = (k - 1) // 2
            out = np.zeros_like(lr_np)
            target_mean = float(np.mean(lr_np[0]))
            target_std = float(np.std(lr_np[0]) + 1e-6)
            curve_names = (w.curve,) + tuple(w.aux_curves or ())
            for r in range(1 + n_aux):
                if r > 0 and lr_np[1 + n_aux + r - 1].max() == 0:
                    continue
                curve = curve_names[r] if r < len(curve_names) else None
                if effective_normalization == "target_shared":
                    # 保留引导与目标间的观测幅度关系, 同时维持仅依赖输入的逐窗口反变换;
                    # 目标统计量在推理时可得, 不使用 HR 值或划分标签。
                    m_r, s_r = target_mean, target_std
                else:
                    m_r, s_r = stats_for_row(w, curve, lr_np[r])
                out[r] = (lr_np[r] - m_r) / s_r
            mask_start = 1 + n_aux
            mask_stop = 1 + 2 * n_aux
            out[mask_start:mask_stop] = lr_np[mask_start:mask_stop]
            if has_observation_mask:
                out[mask_stop] = lr_np[mask_stop]
            if n_identity:
                identity_start = mask_stop + extra_mask
                out[identity_start:] = lr_np[identity_start:]
            if effective_normalization == "target_shared":
                mean, std = target_mean, target_std
            else:
                mean, std = stats_for_row(w, w.curve, lr_np[0])
            lr_rows.append(out)
        else:
            mean, std = stats_for_row(w, w.curve, lr_np)
            lr_rows.append(((lr_np - mean) / std)[None, :].astype(np.float32))
        hr_rows.append(((hr_np - mean) / std)[None, :].astype(np.float32))
        means.append(mean)
        stds.append(std)
    lr = torch.from_numpy(np.stack(lr_rows)).to(device, non_blocking=True)
    hr = torch.from_numpy(np.stack(hr_rows)).to(device, non_blocking=True)
    mean = torch.tensor(means, dtype=torch.float32, device=device)
    std = torch.tensor(stds, dtype=torch.float32, device=device)
    return lr, hr, mean, std


def set_trainable_by_keywords(model, keywords: list[str] | None) -> None:
    if not keywords:
        for param in model.parameters():
            param.requires_grad = True
        return
    lowered = [kw.lower() for kw in keywords]
    for name, param in model.named_parameters():
        lname = name.lower()
        param.requires_grad = any(keyword in lname for keyword in lowered)


def make_optimizer(model, learning_rate: float, device) -> object:
    import torch

    params = [param for param in model.parameters() if param.requires_grad]
    adamw_kwargs = {"lr": learning_rate, "weight_decay": 1e-4}
    if device.type == "cuda":
        adamw_kwargs["fused"] = True
    try:
        return torch.optim.AdamW(params, **adamw_kwargs)
    except TypeError:
        adamw_kwargs.pop("fused", None)
        return torch.optim.AdamW(params, **adamw_kwargs)


def main_prediction(output):
    if isinstance(output, dict):
        return output["curve"]
    return output


def with_input_for_loss(output, lr, loss_weights: dict | None):
    weights = loss_weights or {}
    needs_input = (
        weights.get("tool_response", 0.0)
        or weights.get("data_consistency", 0.0)
        or weights.get("band_residual", 0.0)
    )
    if not needs_input:
        return output
    if isinstance(output, dict):
        return {**output, "input": lr}
    return {"curve": output, "input": lr}


def _crop_eval_tensor(value, windows: list[SRWindow]):
    """将批量张量/字典裁剪到显式评分区间。

    启用 loss_on_scored_interval 时, 所有损失项 (含 FFT 项和响应一致性项) 只在
    评分区间上计算。混合批次由 scored_loss 逐样本处理。
    """
    if isinstance(value, dict):
        return {
            key: _crop_eval_tensor(item, windows)
            if hasattr(item, "shape") and item.ndim >= 3 and item.shape[0] == len(windows)
            else item
            for key, item in value.items()
        }
    if not hasattr(value, "shape") or value.ndim < 3:
        return value
    starts = [int(getattr(w, "eval_start", 0) or 0) for w in windows]
    lengths = [int(getattr(w, "eval_length", value.shape[-1]) or value.shape[-1]) for w in windows]
    if len(set(starts)) != 1 or len(set(lengths)) != 1:
        raise ValueError("mixed scored intervals require per-sample loss evaluation")
    return value[..., starts[0]: starts[0] + lengths[0]]


def scored_loss(
    criterion: SRLoss,
    output,
    target,
    observed,
    windows: list[SRWindow],
    loss_weights: dict | None,
) -> object:
    """在每个窗口声明的评分区间上计算损失。"""
    if not windows:
        raise ValueError("scored_loss requires at least one window")
    starts = [int(getattr(w, "eval_start", 0) or 0) for w in windows]
    lengths = [int(getattr(w, "eval_length", target.shape[-1]) or target.shape[-1]) for w in windows]
    if len(set(starts)) == 1 and len(set(lengths)) == 1:
        cropped_output = _crop_eval_tensor(output, windows)
        cropped_target = target[..., starts[0]: starts[0] + lengths[0]]
        cropped_observed = observed[..., starts[0]: starts[0] + lengths[0]]
        return criterion(
            with_input_for_loss(cropped_output, cropped_observed, loss_weights),
            cropped_target,
        )
    losses = []
    for index, (window, start, length) in enumerate(zip(windows, starts, lengths, strict=True)):
        output_i = output[index:index + 1] if not isinstance(output, dict) else {
            key: item[index:index + 1] if hasattr(item, "shape") and item.ndim >= 3 else item
            for key, item in output.items()
        }
        target_i = target[index:index + 1, ..., start:start + length]
        observed_i = observed[index:index + 1, ..., start:start + length]
        losses.append(
            criterion(
                with_input_for_loss(output_i, observed_i, loss_weights),
                target_i,
            )
        )
    import torch
    return torch.stack(losses).mean()


def multi_objective_losses(
    pred: torch.Tensor | dict[str, torch.Tensor],
    target: torch.Tensor,
    observed: torch.Tensor,
    loss_weights: dict | None,
    scale: int,
) -> list[torch.Tensor]:
    """返回分别加权的各目标项 (供 PCGrad 使用), 权重为零的项省略。"""
    import torch

    weights = loss_weights or {}
    aux_output = pred if isinstance(pred, dict) else {}
    if isinstance(pred, dict):
        pred = pred["curve"]
    objectives: list[torch.Tensor] = []
    point = float(weights.get("l1", 1.0)) * torch.mean(torch.abs(pred - target))
    if weights.get("mse", 0.0):
        point = point + float(weights["mse"]) * torch.mean((pred - target) ** 2)
    if point.requires_grad:
        objectives.append(point)

    shape = torch.zeros((), device=pred.device, dtype=pred.dtype)
    if weights.get("gradient", 0.0):
        shape = shape + float(weights["gradient"]) * torch.mean(
            torch.abs(gradient_1d(pred) - gradient_1d(target))
        )
    if weights.get("gradient_mse", 0.0):
        shape = shape + float(weights["gradient_mse"]) * torch.mean(
            (gradient_1d(pred) - gradient_1d(target)) ** 2
        )
    if weights.get("curvature", 0.0):
        shape = shape + float(weights["curvature"]) * torch.mean(
            torch.abs(curvature_1d(pred) - curvature_1d(target))
        )
    if weights.get("boundary_gradient", 0.0):
        target_grad = gradient_1d(target)
        pred_grad = gradient_1d(pred)
        boundary_weight = torch.abs(target_grad)
        boundary_weight = boundary_weight / (
            torch.mean(boundary_weight, dim=-1, keepdim=True) + 1e-6
        )
        boundary_weight = torch.clamp(
            boundary_weight, 0.25, 6.0
        ).pow(float(weights.get("boundary_power", 1.0)))
        shape = shape + float(weights["boundary_gradient"]) * torch.mean(
            boundary_weight * torch.abs(pred_grad - target_grad)
        )
    if shape.detach().abs().item() > 0.0:
        objectives.append(shape)

    spectrum = torch.zeros((), device=pred.device, dtype=pred.dtype)
    high_weight = float(weights.get("spectral_angle_high_weight", 0.0))
    if weights.get("spectral", 0.0):
        spectrum = spectrum + float(weights["spectral"]) * spectral_l1(pred, target)
    if weights.get("spectral_shape", 0.0):
        spectrum = spectrum + float(weights["spectral_shape"]) * spectral_shape_l1(
            pred, target, high_weight=high_weight
        )
    if weights.get("spectral_angle", 0.0):
        spectrum = spectrum + float(weights["spectral_angle"]) * spectral_angle_loss(
            pred, target, high_weight=high_weight
        )
    if weights.get("spectral_gradient_angle", 0.0):
        spectrum = spectrum + float(weights["spectral_gradient_angle"]) * spectral_gradient_angle_loss(
            pred, target, high_weight=high_weight
        )
    if spectrum.detach().abs().item() > 0.0:
        objectives.append(spectrum)

    # 响应一致性项单独成组。
    consistency = torch.zeros((), device=pred.device, dtype=pred.dtype)
    if weights.get("tool_response", 0.0):
        consistency = consistency + float(weights["tool_response"]) * vertical_response_consistency(
            pred,
            observed[:, :1],
            kernel_size=int(weights.get("tool_response_kernel", 17)),
            stride=int(weights.get("tool_response_stride", 8)),
        )
    if weights.get("data_consistency", 0.0):
        consistency = consistency + float(weights["data_consistency"]) * gaussian_decimation_consistency(
            pred,
            observed[:, :1],
            scale=max(1, int(scale)),
            sigma_factor=float(weights.get("data_consistency_sigma_factor", 1.0)),
        )
    if consistency.detach().abs().item() > 0.0:
        objectives.append(consistency)
    if weights.get("band_residual", 0.0) and "band_residual" in aux_output:
        residual_target = missing_band_component(target - observed[:, :1], max(1, int(scale)))
        target_scale = residual_target.std(dim=-1, keepdim=True).detach() + 1e-4
        band = float(weights["band_residual"]) * torch.mean(
            torch.abs(aux_output["band_residual"] / target_scale - residual_target / target_scale)
        )
        objectives.append(band)
    return objectives


def pcgrad_step(
    objectives: list[torch.Tensor],
    parameters: list[torch.nn.Parameter],
    optimizer,
    *,
    normalize: bool = True,
    priority: bool = False,
    auxiliary_scale: float = 0.20,
) -> torch.Tensor:
    """执行一次确定性 PCGrad 步并返回目标项之和。"""
    import torch

    if not objectives:
        raise ValueError("PCGrad received no objectives")
    gradients: list[list[torch.Tensor | None]] = []
    for objective in objectives:
        grads = torch.autograd.grad(
            objective,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        copied = [None if grad is None else grad.detach().clone() for grad in grads]
        if normalize:
            norm = torch.sqrt(
                sum((grad.float().pow(2).sum() for grad in copied if grad is not None), torch.zeros((), device=objective.device))
                + 1e-12
            )
            copied = [None if grad is None else grad / norm.to(grad.dtype) for grad in copied]
        gradients.append(copied)

    # 按固定顺序投影: 每个目标只保留与冲突目标正交的分量后再平均。
    # priority 模式下第一个目标为锚: 其余目标对其投影、按其范数缩放后注入。
    if priority:
        anchor = gradients[0]
        anchor_norm = torch.sqrt(
            sum((grad.float().pow(2).sum() for grad in anchor if grad is not None), torch.zeros((), device=objectives[0].device))
            + 1e-12
        )
        aggregate: list[torch.Tensor | None] = [
            None if grad is None else grad.clone() for grad in anchor
        ]
        for row in gradients[1:]:
            current = [None if grad is None else grad.clone() for grad in row]
            dot = sum(
                (a * b).sum() for a, b in zip(current, anchor, strict=True) if a is not None and b is not None
            )
            anchor_sq = sum(b.pow(2).sum() for b in anchor if b is not None) + 1e-12
            if bool(dot < 0):
                for k, (a, b) in enumerate(zip(current, anchor, strict=True)):
                    if a is not None and b is not None:
                        current[k] = a - dot / anchor_sq * b
            current_norm = torch.sqrt(
                sum((grad.float().pow(2).sum() for grad in current if grad is not None), torch.zeros((), device=objectives[0].device))
                + 1e-12
            )
            factor = float(auxiliary_scale) * anchor_norm / current_norm
            for k, grad in enumerate(current):
                if grad is not None:
                    contribution = grad * factor.to(grad.dtype)
                    aggregate[k] = contribution if aggregate[k] is None else aggregate[k] + contribution
        optimizer.zero_grad(set_to_none=True)
        for parameter, grad in zip(parameters, aggregate, strict=True):
            if grad is not None:
                parameter.grad = grad
        total = torch.stack([objective.detach() for objective in objectives]).sum()
        return total

    projected: list[list[torch.Tensor | None]] = []
    for i, current in enumerate(gradients):
        current = [None if grad is None else grad.clone() for grad in current]
        for j, other in enumerate(gradients):
            if i == j:
                continue
            dot = sum(
                (a * b).sum() for a, b in zip(current, other, strict=True) if a is not None and b is not None
            )
            other_norm = sum(
                b.pow(2).sum() for b in other if b is not None
            ) + 1e-12
            if bool(dot < 0):
                for k, (a, b) in enumerate(zip(current, other, strict=True)):
                    if a is not None and b is not None:
                        current[k] = a - dot / other_norm * b
        projected.append(current)

    optimizer.zero_grad(set_to_none=True)
    for index, parameter in enumerate(parameters):
        grads = [row[index] for row in projected if row[index] is not None]
        if grads:
            parameter.grad = torch.stack(grads, dim=0).mean(dim=0)
    total = torch.stack([objective.detach() for objective in objectives]).sum()
    return total


V2_NAMES = {"geosr2", "geosr2_long", "geosr2_obs_mask", "geosr2_gd", "geosr2_gd5", "geosr2_deep", "geosr2_nomorph",
            "geosr2_norecur", "geosr2_plainblock", "geosr2_morphtarget", "geosr2_morphguide", "geosr2_nogate",
            "geosr2_dropgrad", "geosr2_droprough", "geosr2_dropcurv", "geosr2_ma5", "geosr2_ma15"}


def train_model(
    name: str,
    train_windows: list[SRWindow],
    val_windows: list[SRWindow],
    test_windows: list[SRWindow],
    *,
    channels: int,
    depth: int,
    modes: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    min_epochs: int = 0,
    min_train_epochs: int = 0,
    patience: int = 10,
    min_delta: float = 1e-5,
    selection_metric: str = "loss",
    selection_weights: dict[str, float] | None = None,
    pareto_checkpoints: int = 0,
    lr_milestones: list[int] | None = None,
    lr_gamma: float = 0.5,
    finetune: dict | None = None,
    device: str = "auto",
    loss_weights: dict | None = None,
    optimization_mode: str = "sum",
    pcgrad_priority: bool = False,
    pcgrad_auxiliary_scale: float = 0.20,
    nan_guard: bool = False,
    deterministic: bool = False,
    normalization: str = "window",
    normalization_local_weight: float = 0.75,
    loss_on_scored_interval: bool = False,
    model_override: object | None = None,
) -> TrainResult:
    import torch

    dev = resolve_device(device)
    if dev.type == "cuda" and deterministic:
        # 位级确定性模式, 需要 CUBLAS_WORKSPACE_CONFIG。
        import os

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True, warn_only=True)
    elif dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    in_channels = int(train_windows[0].lr.shape[0]) if train_windows and train_windows[0].lr.ndim == 2 else 1
    target_identity_channels = (
        len(train_windows[0].target_identity or ()) if train_windows and train_windows[0].lr.ndim == 2 else 0
    )
    target_observation_mask = bool(
        train_windows
        and train_windows[0].lr.ndim == 2
        and train_windows[0].target_observation_mask is not None
    )
    if target_identity_channels and name.lower() not in {
        "geosr2_tc",
        "geosr2_target_conditioned",
        "geosr2_tc_deep",
        "geosr2_target_conditioned_deep",
        "geosr2_tc_raw",
        "geosr2_target_conditioned_raw",
        "geosr2_tc_plain",
        "geosr2_target_conditioned_plain",
    }:
        raise ValueError(
            f"target identity channels require a target-conditioned model, got {name!r}"
        )
    sc = int(train_windows[0].scale) if train_windows else 4
    if model_override is not None:
        model = model_override.to(dev)
    elif in_channels > 1 or name.lower() in V2_NAMES:
        from .models_v2 import build_model_v2

        model = build_model_v2(name, channels=channels, depth=depth, in_channels=in_channels,
                               scale=sc, target_identity_channels=target_identity_channels,
                               target_observation_mask=target_observation_mask).to(dev)
    else:
        model = build_model(name, channels=channels, depth=depth, modes=modes).to(dev)
    if normalization not in {"window", "target_shared", "global_train", "robust_train", "hybrid_train"}:
        raise ValueError(f"unknown normalization mode: {normalization}")
    if normalization in {"global_train", "hybrid_train"}:
        normalization_stats = fit_normalization_stats(train_windows)
    elif normalization == "robust_train":
        normalization_stats = fit_normalization_stats(train_windows, robust=True)
    else:
        normalization_stats = None
    loss_cfg = dict(loss_weights or {})
    if "data_consistency" in loss_cfg and "data_consistency_scale" not in loss_cfg:
        loss_cfg["data_consistency_scale"] = sc
    if ("hf_l1" in loss_cfg or "hf_spectral" in loss_cfg) and "hf_scale" not in loss_cfg:
        loss_cfg["hf_scale"] = sc
    criterion = SRLoss(**loss_cfg)
    opt = make_optimizer(model, learning_rate, dev)
    if optimization_mode not in {"sum", "pcgrad"}:
        raise ValueError(f"unknown optimization mode: {optimization_mode}")
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    train_lr, train_hr, _train_mean, _train_std = windows_to_tensors(
        train_windows, dev, normalization_stats, normalization, normalization_local_weight
    )
    val_lr, val_hr, val_mean, val_std = windows_to_tensors(
        val_windows, dev, normalization_stats, normalization, normalization_local_weight
    )
    discriminator = None
    disc_opt = None
    adversarial_loss = None

    best_state = None
    best_epoch = None
    best_val = float("inf")
    pareto_states: list[tuple[float, int, dict[str, float], dict[str, object]]] = []
    stale = 0
    nonfinite_batches = 0
    n_train = train_lr.shape[0]
    history: list[dict[str, float]] = []
    lr_milestones = sorted(lr_milestones or [])
    finetune = finetune or {}
    finetune_start = int(finetune.get("start_epoch", 0) or 0)
    finetune_started = False
    for _epoch in range(epochs):
        if finetune_start and not finetune_started and (_epoch + 1) == finetune_start:
            set_trainable_by_keywords(model, list(finetune.get("trainable_keywords", [])))
            opt = make_optimizer(model, float(finetune.get("learning_rate", learning_rate * 0.2)), dev)
            if "loss" in finetune:
                criterion = SRLoss(**{**loss_cfg, **dict(finetune["loss"])})
            # 切换后的目标和可训练参数集构成新的优化阶段, 不让切换前累积的 patience 立即终止它。
            stale = 0
            finetune_started = True
        if _epoch + 1 in lr_milestones:
            for group in opt.param_groups:
                group["lr"] *= lr_gamma
        model.train()
        if discriminator is not None:
            discriminator.train()
        train_loss_sum = torch.zeros((), device=dev)
        train_items = 0
        order = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            lr = train_lr[idx]
            hr = train_hr[idx]
            batch_windows = [train_windows[int(i)] for i in idx.detach().cpu().tolist()]
            if discriminator is not None and disc_opt is not None and adversarial_loss is not None:
                with torch.no_grad():
                    fake = main_prediction(model(lr)).detach()
                disc_opt.zero_grad(set_to_none=True)
                real_logits = discriminator(hr)
                fake_logits = discriminator(fake)
                real_targets = torch.ones_like(real_logits)
                fake_targets = torch.zeros_like(fake_logits)
                disc_loss = 0.5 * (
                    adversarial_loss(real_logits, real_targets) + adversarial_loss(fake_logits, fake_targets)
                )
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                disc_opt.step()

            opt.zero_grad(set_to_none=True)
            output = model(lr)
            pred = main_prediction(output)
            if optimization_mode == "pcgrad" and discriminator is None:
                objectives = multi_objective_losses(output, hr, lr, loss_weights, sc)
                loss = pcgrad_step(
                    objectives,
                    trainable_parameters,
                    opt,
                    priority=pcgrad_priority,
                    auxiliary_scale=pcgrad_auxiliary_scale,
                )
            else:
                loss = (
                    scored_loss(criterion, output, hr, lr, batch_windows, loss_weights)
                    if loss_on_scored_interval
                    else criterion(with_input_for_loss(output, lr, loss_weights), hr)
                )
                if discriminator is not None and adversarial_loss is not None:
                    logits = discriminator(pred)
                    loss = loss + 0.005 * adversarial_loss(logits, torch.ones_like(logits))
            # 数值保护: 梯度裁剪救不了损失已非有限的步 (NaN 梯度裁剪后仍是 NaN, 会污染全部权重),
            # 启用时直接丢弃该批次; 训练保持有限时该保护无作用。
            if nan_guard and not bool(torch.isfinite(loss)):
                nonfinite_batches += 1
                opt.zero_grad(set_to_none=True)
                continue
            if optimization_mode != "pcgrad" or discriminator is not None:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if nan_guard and not all(
                bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
            ):
                nonfinite_batches += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            train_loss_sum = train_loss_sum + loss.detach() * lr.shape[0]
            train_items += int(lr.shape[0])

        model.eval()
        val_loss_sum = torch.zeros((), device=dev)
        val_items = 0
        val_pred_chunks = []
        with torch.no_grad():
            for start in range(0, val_lr.shape[0], batch_size):
                lr = val_lr[start : start + batch_size]
                hr = val_hr[start : start + batch_size]
                batch_windows = val_windows[start : start + batch_size]
                output = model(lr)
                pred = main_prediction(output)
                val_batch_loss = (
                    scored_loss(criterion, output, hr, lr, batch_windows, loss_weights)
                    if loss_on_scored_interval
                    else criterion(with_input_for_loss(output, lr, loss_weights), hr)
                )
                val_loss_sum = val_loss_sum + val_batch_loss.detach() * lr.shape[0]
                if selection_metric != "loss":
                    val_pred_chunks.append(pred.detach().cpu().numpy()[:, 0, :])
                val_items += int(lr.shape[0])
        train_loss = float((train_loss_sum / max(1, train_items)).detach().cpu())
        val_loss = float((val_loss_sum / max(1, val_items)).detach().cpu()) if val_items else float("inf")
        selection_score = val_loss
        metric_summary: dict[str, float] = {}
        if selection_metric in {
            "composite_metrics",
            "constrained_metrics",
            "weighted_metrics",
            "thresholded_metrics",
            "normalized_metrics",
        } and val_pred_chunks:
            val_pred_norm = np.concatenate(val_pred_chunks, axis=0)
            val_pred_np = (
                val_pred_norm * val_std.detach().cpu().numpy()[:, None] + val_mean.detach().cpu().numpy()[:, None]
            )
            val_hr_np = np.stack([w.hr.astype(np.float32) for w in val_windows])
            metric_rows = [
                compute_metrics(
                    hr[int(getattr(w, "eval_start", 0) or 0):
                       int(getattr(w, "eval_start", 0) or 0) + int(getattr(w, "eval_length", len(hr)) or len(hr))],
                    pred[int(getattr(w, "eval_start", 0) or 0):
                         int(getattr(w, "eval_start", 0) or 0) + int(getattr(w, "eval_length", len(hr)) or len(hr))],
                    scale=sc,
                ).__dict__
                for w, hr, pred in zip(val_windows, val_hr_np, val_pred_np, strict=True)
            ]
            metric_summary = {k: float(np.mean([row[k] for row in metric_rows])) for k in metric_rows[0]}
            if selection_metric == "composite_metrics":
                selection_score = (
                    metric_summary["mae"]
                    + metric_summary["rmse"]
                    + 0.50 * metric_summary["grad_mae"]
                    + 0.50 * metric_summary["hf_rmse"]
                    + 0.25 * metric_summary["spectral_angle"]
                    - 0.05 * metric_summary["pearson"]
                )
            else:
                selection_score = (
                    2.0 * metric_summary["mae"]
                    + 2.0 * metric_summary["rmse"]
                    + metric_summary["grad_mae"]
                    + metric_summary["hf_rmse"]
                    + 0.05 * metric_summary["spectral_angle"]
                    - 0.02 * metric_summary["pearson"]
                )
            if selection_metric == "weighted_metrics":
                weights = selection_weights or {}
                selection_score = 0.0
                for metric, value in metric_summary.items():
                    selection_score += float(weights.get(metric, 0.0)) * value
            if selection_metric == "normalized_metrics":
                # 以插值目标观测 (仅来自验证输入) 为参考, 在评分区间上按无量纲的相对指标比较检查点。
                ref_rows = []
                for w in val_windows:
                    start = int(getattr(w, "eval_start", 0) or 0)
                    length = int(getattr(w, "eval_length", len(w.hr)) or len(w.hr))
                    stop = start + length
                    ref_rows.append(
                        compute_metrics(w.hr[start:stop], w.lr[0][start:stop], scale=w.scale).__dict__
                    )
                ref_summary = {
                    key: float(np.mean([row[key] for row in ref_rows]))
                    for key in metric_summary
                }
                weights = selection_weights or {}
                selection_score = 0.0
                for metric, value in metric_summary.items():
                    weight = float(weights.get(metric, 1.0))
                    if not weight or metric == "thresholds":
                        continue
                    ref = ref_summary[metric]
                    if metric in {"r2", "pearson", "psnr"}:
                        ratio = (ref + 1e-6) / (value + 1e-6)
                    else:
                        ratio = value / (ref + 1e-6)
                    selection_score += weight * float(ratio)
            if selection_metric == "thresholded_metrics":
                weights = selection_weights or {}
                thresholds = dict(weights.get("thresholds", {})) if isinstance(weights.get("thresholds", {}), dict) else {}
                threshold_penalty = 0.0
                for metric, threshold in thresholds.items():
                    if metric not in metric_summary:
                        continue
                    value = metric_summary[metric]
                    if metric in {"r2", "pearson", "psnr"}:
                        regret = max(0.0, (float(threshold) - value) / (abs(float(threshold)) + 1e-12))
                    else:
                        regret = max(0.0, (value - float(threshold)) / (abs(float(threshold)) + 1e-12))
                    threshold_penalty += float(weights.get(f"{metric}_penalty", 1.0)) * regret
                quality = 0.0
                for metric, value in metric_summary.items():
                    if metric == "thresholds" or metric.endswith("_penalty"):
                        continue
                    quality += float(weights.get(metric, 0.0)) * value
                selection_score = float(weights.get("threshold_scale", 1000.0)) * threshold_penalty + quality
        history.append(
            {
                "epoch": float(_epoch + 1),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "selection_score": float(selection_score),
                "nonfinite_batches": float(nonfinite_batches),
                **{f"val_{k}": v for k, v in metric_summary.items()},
            }
        )
        # 检查点选择遵守最小 epoch 预算。
        can_select = (_epoch + 1) >= max(1, min_epochs)
        if can_select and selection_score < best_val - min_delta:
            best_val = float(selection_score)
            best_epoch = int(_epoch + 1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            if can_select:
                stale += 1
        if can_select and pareto_checkpoints > 0 and metric_summary:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pareto_states.append((float(selection_score), int(_epoch + 1), dict(metric_summary), state))
            pareto_states = sorted(pareto_states, key=lambda item: item[0])[: int(pareto_checkpoints)]
        # min_epochs 约束检查点选择, min_train_epochs 与微调起点约束最早停止 epoch。
        stop_floor = max(min_epochs, min_train_epochs, finetune_start if finetune_start else 0)
        if (_epoch + 1) >= stop_floor and stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    def _predict(windows: list[SRWindow]) -> list[np.ndarray]:
        lr_all, _hr_all, mean_all, std_all = windows_to_tensors(
            windows, dev, normalization_stats, normalization, normalization_local_weight
        )
        preds = []
        with torch.no_grad():
            for start in range(0, lr_all.shape[0], batch_size):
                pred = main_prediction(model(lr_all[start : start + batch_size]))
                pred = pred * std_all[start : start + batch_size, None, None] + mean_all[start : start + batch_size, None, None]
                preds.append(pred.detach().cpu().numpy()[:, 0, :])
        return [row for chunk in preds for row in chunk]

    val_preds = _predict(val_windows)
    preds = _predict(test_windows)
    if pareto_states:
        chosen_state = best_state
        chosen_score = best_val
        chosen_epoch = best_epoch
        if val_windows:
            # 用带保护的谱准则重新评分保留的检查点: 逐点和层界指标须接近最佳验证点, 再优先选谱角更低者。
            current_metrics = mean_metric_dict(val_windows, val_preds)
            best_metrics = current_metrics
            best_guard_score = guarded_checkpoint_score(current_metrics, best_metrics, selection_weights)
            for score, epoch, _summary, state in pareto_states:
                model.load_state_dict(state)
                cand_val_preds = _predict(val_windows)
                cand_metrics = mean_metric_dict(val_windows, cand_val_preds)
                guard_score = guarded_checkpoint_score(cand_metrics, best_metrics, selection_weights)
                if guard_score < best_guard_score - 1e-12:
                    best_guard_score = guard_score
                    chosen_state = state
                    chosen_score = float(score)
                    chosen_epoch = int(epoch)
            if chosen_state is not None:
                model.load_state_dict(chosen_state)
                val_preds = _predict(val_windows)
                preds = _predict(test_windows)
    train_preds = _predict(train_windows)
    if history and best_epoch is not None:
        for row in history:
            row["selected_epoch"] = float(chosen_epoch if pareto_states else best_epoch)
            row["selected_score"] = float(chosen_score if pareto_states else best_val)
    return TrainResult(
        name,
        best_val,
        history,
        val_preds,
        preds,
        train_preds,
        model,
        normalization_stats,
        normalization,
        normalization_local_weight,
    )


def mean_metric_dict(windows: list[SRWindow], preds: list[np.ndarray]) -> dict[str, float]:
    rows = [compute_metrics(w.hr, pred).__dict__ for w, pred in zip(windows, preds, strict=True)]
    return {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}


def guarded_checkpoint_score(
    candidate: dict[str, float],
    reference: dict[str, float],
    selection_weights: dict[str, float] | None,
) -> float:
    weights = selection_weights or {}
    tolerances = dict(weights.get("pareto_tolerances", {})) if isinstance(weights.get("pareto_tolerances", {}), dict) else {}
    default_tol = float(weights.get("pareto_tolerance", 0.002))
    penalty = 0.0
    for metric in ["mae", "rmse", "grad_mae", "hf_rmse"]:
        tol = float(tolerances.get(metric, default_tol))
        allowed = reference[metric] * (1.0 + tol)
        penalty += max(0.0, (candidate[metric] - allowed) / (abs(reference[metric]) + 1e-12))
    pearson_tol = float(tolerances.get("pearson", default_tol))
    allowed_pearson = reference["pearson"] - pearson_tol
    penalty += max(0.0, (allowed_pearson - candidate["pearson"]) / (abs(reference["pearson"]) + 1e-12))
    return 1000.0 * penalty + candidate["spectral_angle"]


def summarize_predictions(windows: list[SRWindow], preds: list[np.ndarray], method: str) -> list[dict]:
    from .metrics import metric_row

    return [metric_row(w, pred, method) for w, pred in zip(windows, preds, strict=True)]
