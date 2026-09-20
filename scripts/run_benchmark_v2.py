"""SuiteSR (滤波器组骨干 GeoSRv2) 及其多输入对比方法的多曲线基准运行器。
每个任务为 (panel, target curve, scale), 窗口携带面板所有曲线的低分辨率观测; 方法通过 multi: true 声明是否使用引导曲线,
支持多退化算子训练、深度翻转 tta 以及失配算子测试 (以 condition 列区分)。输出 window_metrics.csv, summary_metrics.csv 等。

用法:
    PYTHONPATH=src python scripts/run_benchmark_v2.py --config configs/v2_pilot.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from wellsr.data import SRWindow, apply_operator, build_multicurve_windows
from wellsr.degradation import anti_alias_downsample, upsample_to_length
from wellsr.metrics import metric_row, register_data_ranges
from wellsr.models_v2 import build_model_v2
from wellsr.train import main_prediction, resolve_device, train_model

INTERP = {"linear", "pchip"}
LINEAR_TAPS = (-2, -1, 0, 1, 2)


def _n_aux(w: SRWindow) -> int:
    """解析多曲线行布局 (含可选元数据行), 返回引导曲线数量。"""
    n_identity = len(w.target_identity or ())
    observation_mask = int(w.target_observation_mask is not None)
    payload = int(w.lr.shape[0]) - 1 - n_identity - observation_mask
    if payload < 0 or payload % 2:
        raise ValueError(
            f"invalid row layout for {w.dataset}/{w.curve}: "
            f"shape={w.lr.shape}, target_identity={w.target_identity!r}, "
            f"observation_mask={bool(observation_mask)}"
        )
    return payload // 2


def _highpass(a: np.ndarray, k: int = 9) -> np.ndarray:
    pad = k // 2
    padded = np.pad(a, ((0, 0), (pad, pad)), mode="reflect")
    ma = np.stack([padded[:, i:i + a.shape[1]] for i in range(k)], 0).mean(0)
    return a - ma


def linear_guided_features(windows: list[SRWindow]) -> tuple[np.ndarray, np.ndarray]:
    """构造线性跨曲线滤波器的设计矩阵: 目标插值观测和各引导观测的五抽头高通 (引导缺失处为零) 加常数项,
    响应为目标相对插值观测的残差。返回在窗口和深度上展平的 (X, y)。"""
    lr = np.stack([w.lr for w in windows]); hr = np.stack([w.hr for w in windows])
    n_aux = (lr.shape[1] - 1) // 2
    cols = []
    for j in range(1 + n_aux):
        hp = _highpass(lr[:, j])
        if j > 0:
            hp = hp * lr[:, n_aux + j][:, :1]  # 掩码行在窗口内为常数
        for t in LINEAR_TAPS:
            cols.append(np.roll(hp, t, axis=1))
    X = np.stack(cols, -1).reshape(-1, len(cols))
    X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], 1)
    y = (hr - lr[:, 0]).reshape(-1)
    return X.astype(np.float64), y.astype(np.float64)


def linear_guided_fit(windows: list[SRWindow], ridge: float = 1e-6) -> np.ndarray:
    X, y = linear_guided_features(windows)
    return np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ y)


def linear_guided_predict(beta: np.ndarray, windows: list[SRWindow]) -> list[np.ndarray]:
    X, _ = linear_guided_features(windows)
    corr = (X @ beta).reshape(len(windows), -1)
    return [w.lr[0] + c.astype(np.float32) for w, c in zip(windows, corr)]


def _window_zscore(row: np.ndarray) -> tuple[np.ndarray, float, float]:
    """返回局部标准化后的行及用于反变换的均值和标准差。"""
    mean = float(np.mean(row))
    std = float(np.std(row) + 1e-6)
    return ((row - mean) / std).astype(np.float64), mean, std


def wiener_guided_fit(windows: list[SRWindow], ridge: float = 1e-2) -> np.ndarray:
    """仅在训练窗口上拟合多通道复 Wiener 映射。
    该映射在 Fourier 基下由目标插值观测和引导曲线预测目标残差; ridge 相对每频率平均特征能量设定。
    返回形状 (n_freq, n_rows) 的复系数。
    """
    if not windows:
        raise ValueError("cannot fit Wiener map without windows")
    first = windows[0]
    if first.lr.ndim != 2:
        raise ValueError("wiener_guided requires multi-curve windows")
    n_rows = 1 + len(first.aux_curves or ())
    length = int(first.hr.shape[-1])
    spectra_x = []
    spectra_y = []
    for w in windows:
        if w.lr.shape[0] < n_rows:
            raise ValueError("inconsistent guide count in Wiener windows")
        target_z, mean, std = _window_zscore(w.lr[0])
        target_hr = (w.hr.astype(np.float64) - mean) / std
        rows = [target_z]
        for j in range(1, n_rows):
            row = w.lr[j].astype(np.float64)
            mask = w.lr[n_rows + j - 1]
            if float(np.max(mask)) <= 0.0:
                row = np.zeros_like(row)
            row_z, _, _ = _window_zscore(row)
            rows.append(row_z)
        x = np.fft.rfft(np.stack(rows, axis=0), axis=-1).T
        y = np.fft.rfft(target_hr - target_z)
        spectra_x.append(x)
        spectra_y.append(y)
    x_all = np.stack(spectra_x, axis=0)  # 形状 (N, F, C)
    y_all = np.stack(spectra_y, axis=0)  # 形状 (N, F)
    n_freq = x_all.shape[1]
    beta = np.empty((n_freq, n_rows), dtype=np.complex128)
    for freq in range(n_freq):
        x = x_all[:, freq, :]
        y = y_all[:, freq]
        gram = x.conj().T @ x
        scale = float(np.trace(gram).real / max(1, n_rows))
        gram = gram + float(ridge) * max(scale, 1e-8) * np.eye(n_rows)
        beta[freq] = np.linalg.solve(gram, x.conj().T @ y)
    return beta


def wiener_guided_predict(beta: np.ndarray, windows: list[SRWindow]) -> list[np.ndarray]:
    """应用冻结的 Wiener 映射并反变换目标窗口归一化, 返回预测列表。"""
    preds = []
    n_rows = beta.shape[1]
    for w in windows:
        target_z, mean, std = _window_zscore(w.lr[0])
        rows = [target_z]
        for j in range(1, n_rows):
            row = w.lr[j].astype(np.float64)
            mask = w.lr[n_rows + j - 1]
            if float(np.max(mask)) <= 0.0:
                row = np.zeros_like(row)
            row_z, _, _ = _window_zscore(row)
            rows.append(row_z)
        x = np.fft.rfft(np.stack(rows, axis=0), axis=-1).T
        residual = np.fft.irfft(np.sum(x * beta, axis=1), n=w.hr.shape[-1])
        preds.append((w.lr[0] + std * residual).astype(np.float32))
    return preds


def split_internal_holdout(windows: list[SRWindow], seed: int, val_fraction: float = 0.20) -> tuple[list[SRWindow], list[SRWindow]]:
    """从每口训练井留出最深的一部分窗口用于检查点选择, 不使用 val/test 井。
    返回 (train, val) 两个窗口列表。
    """
    if not windows:
        return [], []
    by_well: dict[str, list[SRWindow]] = {}
    for window in windows:
        by_well.setdefault(window.well, []).append(window)
    train, val = [], []
    rng = np.random.default_rng(seed)
    for well, items in by_well.items():
        items = sorted(items, key=lambda w: (float(w.depth_start), w.curve, w.scale))
        n_val = max(1, int(round(len(items) * float(val_fraction)))) if len(items) > 4 else max(1, len(items) // 5 or 1)
        n_val = min(n_val, max(1, len(items) - 1)) if len(items) > 1 else 0
        if n_val == 0:
            train.extend(items)
            continue
        cut = len(items) - n_val
        train.extend(items[:cut])
        val.extend(items[cut:])
    if not val:
        # 回退: 随机窗口, 仍仅来自训练井
        idx = rng.choice(len(windows), size=max(1, len(windows) // 5), replace=False)
        hold = {int(i) for i in idx}
        train = [w for i, w in enumerate(windows) if i not in hold]
        val = [w for i, w in enumerate(windows) if i in hold]
    return train, val


def subsample_groups(windows: list[SRWindow], n_variants: int, cap: int | None, seed: int) -> list[SRWindow]:
    """按 seed 随机抽取基础窗口子集, 保留其全部退化变体。"""
    n_base = len(windows) // n_variants
    if cap is None or n_base <= cap:
        return windows
    rng = np.random.default_rng(seed)
    keep = sorted(rng.choice(n_base, size=cap, replace=False))
    out = []
    for b in keep:
        out.extend(windows[b * n_variants:(b + 1) * n_variants])
    return out


def single_curve(windows: list[SRWindow]) -> list[SRWindow]:
    return [
        SRWindow(
            w.dataset, w.well, w.curve, w.split, w.scale, w.depth_start,
            w.lr[0].copy(), w.hr, eval_start=w.eval_start, eval_length=w.eval_length,
            # 单曲线视图不含辅助行和目标观测掩码
            target_observation_mask=None,
        )
        for w in windows
    ]


def predict(
    model,
    windows: list[SRWindow],
    device,
    batch: int,
    tta: bool,
    normalization_stats=None,
    normalization: str = "window",
    normalization_local_weight: float = 0.75,
) -> list[np.ndarray]:
    from wellsr.train import windows_to_tensors

    lr, _hr, mean, std = windows_to_tensors(
        windows,
        device,
        normalization_stats,
        normalization,
        normalization_local_weight,
    )
    preds = []
    with torch.no_grad():
        for s in range(0, lr.shape[0], batch):
            x = lr[s:s + batch]
            y = main_prediction(model(x))
            if tta:
                yf = main_prediction(model(torch.flip(x, dims=[-1])))
                y = 0.5 * (y + torch.flip(yf, dims=[-1]))
            y = y * std[s:s + batch, None, None] + mean[s:s + batch, None, None]
            preds.append(y.cpu().numpy()[:, 0, :])
    return [r for c in preds for r in c]


def mismatched_copy(windows: list[SRWindow], scale: int, spec: dict, aux_div: int = 1) -> list[SRWindow]:
    """用另一退化算子重新退化测试窗口的目标行和引导行, 返回新窗口列表。"""
    out = []
    for w in windows:
        lr = w.lr.copy()
        n_aux = _n_aux(w)
        rows = [w.hr] + [w.aux_hr[j] if w.aux_hr is not None else None for j in range(n_aux)]
        for j, src in enumerate(rows):
            if src is None or (j > 0 and lr[n_aux + j].max() == 0):
                continue
            sc = w.scale if j == 0 else max(1, scale // aux_div)
            if sc == 1:
                continue
            lo = apply_operator(src, spec, sc)[::sc]
            lr[j] = upsample_to_length(lo.astype(np.float32), len(w.hr), kind="linear", scale=sc)
        out.append(
            SRWindow(
                w.dataset,
                w.well,
                w.curve,
                w.split,
                w.scale,
                w.depth_start,
                lr,
                w.hr,
                w.aux_hr,
                w.aux_curves,
                w.target_identity,
                w.eval_start,
                w.eval_length,
                w.target_observation_mask,
            )
        )
    return out


def shifted_guides(windows: list[SRWindow], k: int) -> list[SRWindow]:
    """模拟测试时深度失配: 将每个引导行平移 k 个样本并填充边缘, 返回新窗口列表。"""
    out = []
    for w in windows:
        lr = w.lr.copy()
        n_aux = _n_aux(w)
        for j in range(1, 1 + n_aux):
            lr[j] = np.roll(lr[j], k)
            if k > 0:
                lr[j, :k] = lr[j, k]
            elif k < 0:
                lr[j, k:] = lr[j, k - 1]
        out.append(
            SRWindow(
                w.dataset,
                w.well,
                w.curve,
                w.split,
                w.scale,
                w.depth_start,
                lr,
                w.hr,
                w.aux_hr,
                w.aux_curves,
                w.target_identity,
                w.eval_start,
                w.eval_length,
                w.target_observation_mask,
            )
        )
    return out


def save_prediction_bundle(
    path: Path,
    *,
    val_windows: list[SRWindow],
    val_preds: list[np.ndarray],
    test_windows: list[SRWindow],
    test_preds: list[np.ndarray],
) -> None:
    """将验证和测试预测连同窗口标识保存为 npz。"""
    np.savez(
        path,
        val_pred=np.stack(val_preds),
        val_hr=np.stack([w.hr for w in val_windows]),
        val_lr=np.stack([w.lr for w in val_windows]),
        val_dataset=np.array([w.dataset for w in val_windows]),
        val_curve=np.array([w.curve for w in val_windows]),
        val_scale=np.array([w.scale for w in val_windows]),
        val_wells=np.array([w.well for w in val_windows]),
        val_depth=np.array([w.depth_start for w in val_windows]),
        val_eval_start=np.array([w.eval_start for w in val_windows]),
        val_eval_length=np.array([w.eval_length if w.eval_length is not None else -1 for w in val_windows]),
        test_pred=np.stack(test_preds),
        test_hr=np.stack([w.hr for w in test_windows]),
        test_lr=np.stack([w.lr for w in test_windows]),
        test_dataset=np.array([w.dataset for w in test_windows]),
        test_curve=np.array([w.curve for w in test_windows]),
        test_scale=np.array([w.scale for w in test_windows]),
        test_wells=np.array([w.well for w in test_windows]),
        test_depth=np.array([w.depth_start for w in test_windows]),
        test_eval_start=np.array([w.eval_start for w in test_windows]),
        test_eval_length=np.array([w.eval_length if w.eval_length is not None else -1 for w in test_windows]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed = int(cfg.get("seed", 2026))
    # model_seed 将参数初始化和批次顺序与窗口子采样 (seed) 解耦
    model_seed = int(cfg.get("model_seed", seed))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(model_seed)
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    data_root = cfg.get("data_root", "data/processed")
    device = resolve_device(cfg.get("device", "auto"))
    caps = cfg.get("max_windows_per_split", {"train": 1024, "val": 256, "test": 512})
    W, S = int(cfg.get("window", 128)), int(cfg.get("stride", 128))
    train_stride = int(cfg.get("train_stride", S))
    val_stride = int(cfg.get("val_stride", S))
    test_stride = int(cfg.get("test_stride", S))
    min_finite_fraction = float(cfg.get("min_finite_fraction", 0.95))
    require_complete_aux = bool(cfg.get("require_complete_aux", True))
    require_complete_aux_train = bool(cfg.get("require_complete_aux_train", require_complete_aux))
    require_complete_aux_val = bool(cfg.get("require_complete_aux_val", require_complete_aux))
    require_complete_aux_test = bool(cfg.get("require_complete_aux_test", require_complete_aux))
    internal_holdout = bool(cfg.get("internal_train_holdout", False))
    internal_holdout_fraction = float(cfg.get("internal_holdout_fraction", 0.20))
    train_degs = cfg.get("train_degradations") or []
    test_degs = cfg.get("test_degradations") or []
    n_var = 1 + len(train_degs)
    aux_div = int(cfg.get("aux_scale_divisor", 1))
    upsample = str(cfg.get("upsample", "linear"))
    response_scope = str(cfg.get("response_scope", "window"))
    context_window = cfg.get("context_window")
    include_target_observation_mask = bool(cfg.get("include_target_observation_mask", False))
    csf = cfg.get("curve_scale_factor") or None
    rows, hist = [], []
    t0 = time.time()
    heldout = cfg.get("heldout")          # 设置时为留一面板迁移
    sources = [d for d in cfg["datasets"] if d != heldout] if heldout else None
    pool_scales = bool(cfg.get("pool_scales", False))
    pool = bool(cfg.get("pool_curves", False) or pool_scales)
    identity_curves = cfg.get("target_identity_curves")
    identity_curve_pool = []
    for panel in cfg["datasets"]:
        for curve in cfg["curves_by_dataset"][panel]:
            if curve not in identity_curve_pool:
                identity_curve_pool.append(curve)
    if pool_scales and not heldout and identity_curves is None:
        identity_curves = [
            f"{curve}@{scale}"
            for curve in identity_curve_pool
            for scale in cfg["scales"]
        ]
    elif pool and not heldout and identity_curves is None:
        identity_curves = identity_curve_pool
    pooled_models = {}

    # 汇聚训练时引导槽位需有面板级统一含义
    canonical_aux_curves = None
    if pool and not heldout:
        canonical_aux_curves = tuple(identity_curve_pool)

    def identity_key(curve: str, scale: int) -> str:
        return f"{curve}@{scale}" if pool_scales else curve

    for ds in ([heldout] if heldout else cfg["datasets"]):
        curves = cfg["curves_by_dataset"][ds]
        if heldout:
            # 仅所有源面板共有的曲线可作目标或引导
            shared = set(curves)
            for s_ds in sources:
                shared &= set(cfg["curves_by_dataset"][s_ds])
            curves = [c for c in curves if c in shared]
        targets = cfg.get("targets_by_dataset", {}).get(ds, curves)
        targets = [c for c in targets if c in curves]
        pooled_train_val = {}
        if pool and not heldout:
            pooled_scales = cfg["scales"] if pool_scales else [cfg["scales"][0]]
            pooled_tr, pooled_va = [], []
            for t2 in curves:
                a2 = [c for c in curves if c != t2]
                for pooled_scale in pooled_scales:
                    pooled_tr += subsample_groups(
                        build_multicurve_windows(
                            ds, t2, a2, pooled_scale, "train", W, train_stride,
                            data_root=data_root,
                            degradations=train_degs, aux_scale_divisor=aux_div, curve_scale_factor=csf,
                            target_identity_curves=identity_curves,
                            target_identity_key=identity_key(t2, pooled_scale),
                            canonical_aux_curves=canonical_aux_curves,
                            upsample=upsample,
                            response_scope=response_scope,
                            context_window=context_window,
                            include_target_observation_mask=include_target_observation_mask,
                            min_finite_fraction=min_finite_fraction,
                            require_complete_aux=require_complete_aux_train,
                        ),
                        n_var, caps.get("train"), seed)
                    pooled_va += build_multicurve_windows(
                        ds, t2, a2, pooled_scale, "val", W, S,
                        data_root=data_root,
                        max_windows=caps.get("val"), subsample_seed=seed,
                        aux_scale_divisor=aux_div, curve_scale_factor=csf,
                        target_identity_curves=identity_curves,
                        target_identity_key=identity_key(t2, pooled_scale),
                        canonical_aux_curves=canonical_aux_curves,
                        upsample=upsample,
                        response_scope=response_scope,
                        context_window=context_window,
                        include_target_observation_mask=include_target_observation_mask,
                        min_finite_fraction=min_finite_fraction,
                        require_complete_aux=require_complete_aux_val,
                    )
            pooled_train_val["all" if pool_scales else "scale"] = (pooled_tr, pooled_va)
        for target in targets:
            aux = [c for c in curves if c != target]
            for scale in cfg["scales"]:
                if heldout:
                    tr, va = [], []
                    for s_ds in sources:
                        tr += subsample_groups(
                            build_multicurve_windows(s_ds, target, aux, scale, "train", W, train_stride,
                                                     data_root=data_root,
                                                     degradations=train_degs, aux_scale_divisor=aux_div, curve_scale_factor=csf,
                                                     target_identity_curves=identity_curves,
                                                     target_identity_key=identity_key(target, scale),
                                                     canonical_aux_curves=canonical_aux_curves,
                                                     upsample=upsample,
                                                     response_scope=response_scope,
                                                     context_window=context_window,
                                                     include_target_observation_mask=include_target_observation_mask,
                                                     min_finite_fraction=min_finite_fraction,
                                                     require_complete_aux=require_complete_aux_train),
                            n_var, caps.get("train"), seed)
                        va += build_multicurve_windows(s_ds, target, aux, scale, "val", W, S,
                                                       data_root=data_root,
                                                       max_windows=caps.get("val"), subsample_seed=seed,
                                                       aux_scale_divisor=aux_div, curve_scale_factor=csf,
                                                       target_identity_curves=identity_curves,
                                                       target_identity_key=identity_key(target, scale),
                                                       canonical_aux_curves=canonical_aux_curves,
                                                       upsample=upsample,
                                                       response_scope=response_scope,
                                                       context_window=context_window,
                                                       include_target_observation_mask=include_target_observation_mask,
                                                       min_finite_fraction=min_finite_fraction,
                                                       require_complete_aux=require_complete_aux_val)
                elif pool and not heldout:
                    tr, va = [], []
                else:
                    tr_all = build_multicurve_windows(
                        ds, target, aux, scale, "train", W, train_stride,
                        data_root=data_root,
                        degradations=train_degs, aux_scale_divisor=aux_div, curve_scale_factor=csf,
                        target_identity_curves=identity_curves,
                        target_identity_key=identity_key(target, scale),
                        upsample=upsample,
                        response_scope=response_scope,
                        context_window=context_window,
                        include_target_observation_mask=include_target_observation_mask,
                        min_finite_fraction=min_finite_fraction,
                        require_complete_aux=require_complete_aux_train,
                    )
                    if internal_holdout:
                        tr_base, va = split_internal_holdout(tr_all, seed, internal_holdout_fraction)
                        tr = subsample_groups(tr_base, n_var, caps.get("train"), seed)
                        if caps.get("val") is not None and len(va) > int(caps["val"]):
                            rng = np.random.default_rng(seed + 17)
                            keep = sorted(rng.choice(len(va), size=int(caps["val"]), replace=False))
                            va = [va[i] for i in keep]
                    else:
                        tr = subsample_groups(tr_all, n_var, caps.get("train"), seed)
                        va = build_multicurve_windows(
                            ds, target, aux, scale, "val", W, val_stride,
                            data_root=data_root,
                            max_windows=caps.get("val"), subsample_seed=seed,
                            aux_scale_divisor=aux_div, curve_scale_factor=csf,
                            target_identity_curves=identity_curves,
                            target_identity_key=identity_key(target, scale),
                            upsample=upsample,
                            response_scope=response_scope,
                            context_window=context_window,
                            include_target_observation_mask=include_target_observation_mask,
                            require_complete_aux=require_complete_aux_val,
                            min_finite_fraction=min_finite_fraction,
                        )
                if pool and not heldout:
                    # 每面板每尺度一个模型 (pool_scales 时每面板一个); 测试集仍按任务划分
                    tr, va = pooled_train_val["all" if pool_scales else "scale"]
                # 保存的预测文件使用任务专属验证窗口, 而非汇聚训练的联合验证池
                task_val = build_multicurve_windows(
                    ds, target, aux, scale, "val", W, val_stride,
                    data_root=data_root,
                    max_windows=caps.get("val"), subsample_seed=seed,
                    aux_scale_divisor=aux_div, curve_scale_factor=csf,
                    target_identity_curves=identity_curves,
                    target_identity_key=identity_key(target, scale),
                    canonical_aux_curves=canonical_aux_curves,
                    upsample=upsample,
                    response_scope=response_scope,
                    context_window=context_window,
                    include_target_observation_mask=include_target_observation_mask,
                    require_complete_aux=require_complete_aux_val,
                    min_finite_fraction=min_finite_fraction,
                ) if pool and not heldout else va
                te = build_multicurve_windows(
                    ds, target, aux, scale, "test", W, test_stride,
                    data_root=data_root,
                    max_windows=caps.get("test"), subsample_seed=seed,
                    aux_scale_divisor=aux_div, curve_scale_factor=csf,
                    target_identity_curves=identity_curves,
                    target_identity_key=identity_key(target, scale),
                    canonical_aux_curves=canonical_aux_curves,
                    upsample=upsample,
                    response_scope=response_scope,
                    context_window=context_window,
                    include_target_observation_mask=include_target_observation_mask,
                    require_complete_aux=require_complete_aux_test,
                    min_finite_fraction=min_finite_fraction,
                )
                if not tr or not te:
                    print(f"[skip] {ds} {target} {scale}x: no windows"); continue
                register_data_ranges(single_curve(tr))
                for method, mcfg in cfg["methods"].items():
                    pred_path = out_dir / "predictions" / f"{ds}_{target}_{scale}_{method}.npz"
                    if (
                        cfg.get("save_predictions")
                        and pred_path.exists()
                        and not bool(cfg.get("overwrite_predictions", False))
                    ):
                        print(
                            f"[skip-existing] {ds} {target} {scale}x {method} ({pred_path})",
                            flush=True,
                        )
                        continue
                    print(f"[run] {ds} {target} {scale}x {method} (train {len(tr)}, test {len(te)})", flush=True)
                    if method in INTERP:
                        for w in te:
                            if response_scope == "continuous":
                                # 窗口已携带连续切片退化后的观测
                                lo = w.lr[0][::scale]
                                p = upsample_to_length(lo, len(w.hr), kind=method, scale=scale)
                            else:
                                lo = anti_alias_downsample(w.hr, scale)
                                p = upsample_to_length(lo, len(w.hr), kind=method, scale=scale)
                            rows.append(metric_row(w, p, method, extra={"condition": "matched"}))
                        continue
                    if method == "linear_guided":
                        # 非学习引导基线: 在训练窗口上最小二乘拟合的跨曲线滤波器
                        beta = linear_guided_fit(tr)
                        preds = linear_guided_predict(beta, te)
                        rows.extend(metric_row(w, p, method, extra={"condition": "matched"}) for w, p in zip(te, preds))
                        if cfg.get("save_predictions"):
                            pdir = out_dir / "predictions"; pdir.mkdir(exist_ok=True)
                            save_prediction_bundle(
                                pdir / f"{ds}_{target}_{scale}_{method}.npz",
                                val_windows=va,
                                val_preds=linear_guided_predict(beta, va),
                                test_windows=te,
                                test_preds=preds,
                            )
                        continue
                    if method == "wiener_guided":
                        # 仅在训练窗口上拟合频率响应; 验证预测一并保存
                        beta = wiener_guided_fit(tr, ridge=float(mcfg.get("ridge", 1e-2)))
                        preds = wiener_guided_predict(beta, te)
                        rows.extend(metric_row(w, p, method, extra={"condition": "matched"})
                                    for w, p in zip(te, preds))
                        if cfg.get("save_predictions"):
                            pdir = out_dir / "predictions"; pdir.mkdir(exist_ok=True)
                            save_prediction_bundle(
                                pdir / f"{ds}_{target}_{scale}_{method}.npz",
                                val_windows=va,
                                val_preds=wiener_guided_predict(beta, va),
                                test_windows=te,
                                test_preds=preds,
                            )
                        continue
                    multi = bool(mcfg.get("multi", False))
                    tr_m, va_m, te_m = (tr, va, te) if multi else (single_curve(tr), single_curve(va), single_curve(te))
                    model_name = mcfg.get("arch", method)
                    # 深度抖动增强: 复制训练窗口并平移引导曲线
                    jit = mcfg.get("guide_shift_jitter") or cfg.get("train_guide_shifts") or []
                    if multi and jit:
                        tr_m = list(tr_m) + [w for k in jit for w in shifted_guides(tr, int(k))]
                        va_m = list(va_m) + [w for k in jit for w in shifted_guides(va, int(k))]
                    cache_key = (ds, method) if pool_scales and not heldout else None
                    if cache_key is not None and cache_key in pooled_models:
                        r = pooled_models[cache_key]
                    else:
                        torch.manual_seed(model_seed)
                        common_train_kwargs = dict(
                            channels=int(mcfg.get("channels", 32)),
                            depth=int(mcfg.get("depth", 3)),
                            modes=int(mcfg.get("modes", 24)),
                            batch_size=int(cfg.get("batch_size", 256)),
                            epochs=int(cfg.get("epochs", 300)),
                            learning_rate=float(cfg.get("learning_rate", 6e-4)),
                            min_epochs=int(mcfg.get("min_epochs", cfg.get("min_epochs", 0))),
                            min_train_epochs=int(cfg.get("min_train_epochs", 100)),
                            patience=int(cfg.get("patience", 30)),
                            min_delta=float(mcfg.get("min_delta", cfg.get("min_delta", 1e-5))),
                            selection_metric=str(
                                mcfg.get("selection_metric", cfg.get("selection_metric", "loss"))
                            ),
                            selection_weights=dict(
                                mcfg.get("selection_weights", cfg.get("selection_weights", {}))
                            ),
                            pareto_checkpoints=int(
                                mcfg.get("pareto_checkpoints", cfg.get("pareto_checkpoints", 0))
                            ),
                            lr_milestones=list(
                                mcfg.get("lr_milestones", cfg.get("lr_milestones", [])) or []
                            ),
                            lr_gamma=float(mcfg.get("lr_gamma", cfg.get("lr_gamma", 0.5))),
                            finetune=mcfg.get("finetune"),
                            loss_weights=mcfg.get("loss"),
                            optimization_mode=str(
                                mcfg.get("optimization_mode", cfg.get("optimization_mode", "sum"))
                            ),
                            pcgrad_priority=bool(
                                mcfg.get("pcgrad_priority", cfg.get("pcgrad_priority", False))
                            ),
                            pcgrad_auxiliary_scale=float(
                                mcfg.get(
                                    "pcgrad_auxiliary_scale",
                                    cfg.get("pcgrad_auxiliary_scale", 0.20),
                                )
                            ),
                            device=cfg.get("device", "auto"),
                            normalization=mcfg.get("normalization", cfg.get("normalization", "window")),
                            normalization_local_weight=float(
                                mcfg.get("normalization_local_weight", cfg.get("normalization_local_weight", 0.75))
                            ),
                            loss_on_scored_interval=bool(
                                mcfg.get(
                                    "loss_on_scored_interval",
                                    cfg.get("loss_on_scored_interval", False),
                                )
                            ),
                        )
                        if model_name.lower() in {"dmc_guide_residual", "dmcnet_guide_residual", "dmc_residual"}:
                            # 第一阶段为编码 DMC 对比模型, 其验证选定状态载入 refiner 并冻结后进入第二阶段
                            stage1_name = mcfg.get("stage1_arch", "dmcnet_morph_native")
                            stage1 = train_model(stage1_name, tr_m, va_m, te_m, **common_train_kwargs)
                            refiner = build_model_v2(
                                model_name,
                                channels=common_train_kwargs["channels"],
                                depth=common_train_kwargs["depth"],
                                in_channels=int(tr_m[0].lr.shape[0]),
                                scale=int(tr_m[0].scale),
                                target_identity_channels=len(tr_m[0].target_identity or ()),
                            )
                            refiner.load_base_state_dict(stage1.model.state_dict())
                            r = train_model(
                                model_name,
                                tr_m,
                                va_m,
                                te_m,
                                model_override=refiner,
                                **common_train_kwargs,
                            )
                            for h in stage1.history:
                                hist.append({"dataset": ds, "curve": target, "scale": scale,
                                             "method": method, "stage": "stage1_dmc", **h})
                            for h in r.history:
                                hist.append({"dataset": ds, "curve": target, "scale": scale,
                                             "method": method, "stage": "stage2_residual", **h})
                        else:
                            r = train_model(model_name, tr_m, va_m, te_m, **common_train_kwargs)
                        if cache_key is not None:
                            pooled_models[cache_key] = r
                    model = r.model; model.eval()
                    tta = bool(mcfg.get("tta", False))
                    preds = predict(
                        model,
                        te_m,
                        device,
                        256,
                        tta,
                        r.normalization_stats,
                        r.normalization,
                        r.normalization_local_weight,
                    )
                    rows.extend(metric_row(w, p, method, extra={"condition": "matched"}) for w, p in zip(te, preds))
                    if cfg.get("save_predictions"):
                        pdir = out_dir / "predictions"; pdir.mkdir(exist_ok=True)
                        # 仅评估基础验证集; va_m 可能含增强副本, 不能当作 va 保存
                        va_eval_m = task_val if multi else single_curve(task_val)
                        val_preds = predict(
                            model,
                            va_eval_m,
                            device,
                            256,
                            tta,
                            r.normalization_stats,
                            r.normalization,
                            r.normalization_local_weight,
                        )
                        save_prediction_bundle(
                            pdir / f"{ds}_{target}_{scale}_{method}.npz",
                            val_windows=task_val,
                            val_preds=val_preds,
                            test_windows=te,
                            test_preds=preds,
                        )
                    for k in cfg.get("test_guide_shifts", []) or []:
                        if not multi:
                            continue
                        preds = predict(
                            model,
                            shifted_guides(te, int(k)),
                            device,
                            256,
                            tta,
                            r.normalization_stats,
                            r.normalization,
                            r.normalization_local_weight,
                        )
                        rows.extend(metric_row(w, p, method, extra={"condition": f"guide_shift{k:+d}"})
                                    for w, p in zip(te, preds))
                    for spec in test_degs:
                        te_mm = mismatched_copy(te, scale, spec, aux_div)
                        te_mm_m = te_mm if multi else single_curve(te_mm)
                        preds = predict(
                            model,
                            te_mm_m,
                            device,
                            256,
                            tta,
                            r.normalization_stats,
                            r.normalization,
                            r.normalization_local_weight,
                        )
                        label = spec.get("label") or f"{spec.get('kind','gaussian')}_{spec.get('sigma_factor',1.0)}"
                        rows.extend(metric_row(w, p, method, extra={"condition": label}) for w, p in zip(te, preds))
                    if model_name.lower() not in {"dmc_guide_residual", "dmcnet_guide_residual", "dmc_residual"}:
                        for h in r.history:
                            hist.append({"dataset": ds, "curve": target, "scale": scale,
                                         "method": method, "stage": "single_stage", **h})
                    pd.DataFrame(rows).to_csv(out_dir / "window_metrics.csv", index=False)
    wm = pd.DataFrame(rows)
    wm.to_csv(out_dir / "window_metrics.csv", index=False)
    if wm.empty:
        mets = []
        summ = pd.DataFrame(columns=["dataset", "curve", "scale", "method", "condition"])
    else:
        mets = [c for c in wm.columns if c not in {"dataset", "curve", "scale", "well", "depth_start", "method", "condition"}]
        summ = wm.groupby(["dataset", "curve", "scale", "method", "condition"], as_index=False)[mets].mean()
    summ.to_csv(out_dir / "summary_metrics.csv", index=False)
    pd.DataFrame(hist).to_csv(out_dir / "training_history.csv", index=False)
    json.dump({"config": cfg, "seconds": time.time() - t0}, open(out_dir / "manifest.json", "w"), indent=2)
    print("wrote", out_dir, f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
