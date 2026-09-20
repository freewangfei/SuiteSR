from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .degradation import make_lr_hr_pair, upsample_to_length


@dataclass
class SRWindow:
    dataset: str
    well: str
    curve: str
    split: str
    scale: int
    depth_start: float
    lr: np.ndarray
    hr: np.ndarray
    aux_hr: np.ndarray | None = None  # 多曲线窗口的 HR 辅助曲线, 形状 (n_aux, L)
    aux_curves: tuple[str, ...] | None = None
    target_identity: tuple[str, ...] | None = None
    # 模型输入上下文宽于评分区间时, 显式记录目标裁剪位置,
    # 使验证和测试指标仍与原基准窗口配对。
    eval_start: int = 0
    eval_length: int | None = None
    # 可选的仅输入侧掩码, 标记抽取后保留的观测位置;
    # 与引导曲线存在掩码分开, 使模型能区分实测 LR 样本与插值。
    target_observation_mask: np.ndarray | None = None


def synthetic_panel(n_wells: int = 24, length: int = 512, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    curves = ["GR", "RHOB", "DTC"]
    for i in range(n_wells):
        split = "train" if i < int(0.7 * n_wells) else "val" if i < int(0.85 * n_wells) else "test"
        depth = np.arange(length, dtype=np.float32)
        base = rng.normal(0.0, 0.08, size=length).cumsum()
        boundaries = np.zeros(length, dtype=np.float32)
        for pos in rng.choice(np.arange(24, length - 24), size=8, replace=False):
            boundaries[pos:] += rng.normal(0.0, 0.7)
        thin_beds = sum(
            rng.normal(0.0, 0.45) * np.exp(-0.5 * ((depth - rng.integers(20, length - 20)) / rng.uniform(2, 8)) ** 2)
            for _ in range(8)
        )
        gr = 0.7 * base + boundaries + thin_beds + rng.normal(0, 0.05, length)
        rhob = -0.45 * gr + rng.normal(0, 0.08, length)
        dtc = 0.35 * gr + 0.25 * np.sin(depth / 19.0) + rng.normal(0, 0.06, length)
        values = {"GR": gr, "RHOB": rhob, "DTC": dtc}
        for j in range(length):
            rows.append({"WELL": f"SYN_{i:03d}", "DEPTH": float(j), "SPLIT": split, **{c: float(values[c][j]) for c in curves}})
    return pd.DataFrame(rows)


def load_panel(dataset: str, data_root: str | Path = "data/processed") -> pd.DataFrame:
    if dataset == "synthetic":
        return synthetic_panel()
    path = Path(data_root) / dataset / "panel.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def available_curves(df: pd.DataFrame) -> list[str]:
    excluded = {"WELL", "DEPTH", "SPLIT", "SLICE_ID"}
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def build_windows(
    dataset: str,
    curves: list[str] | None,
    scales: list[int],
    split: str,
    window: int,
    stride: int,
    data_root: str | Path = "data/processed",
    max_windows: int | None = None,
    upsample: str = "linear",
    response_scope: str = "window",
    min_finite_fraction: float = 0.95,
    subsample_seed: int = 2026,
) -> list[SRWindow]:
    if response_scope not in {"window", "continuous"}:
        raise ValueError(f"unknown response_scope {response_scope!r}")
    df = load_panel(dataset, data_root)
    curves = curves or available_curves(df)
    missing = [c for c in curves if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset} missing curves: {missing}")
    df = df[df["SPLIT"].astype(str).str.lower() == split.lower()].copy()
    windows: list[SRWindow] = []
    # 有 SLICE_ID 时按切片分组: 同一井的切片在深度上不一定连续, 切窗不能跨切片边界;
    # 每个窗口记录井名。
    group_cols = ["WELL", "SLICE_ID"] if "SLICE_ID" in df.columns else ["WELL"]
    for key, g in df.groupby(group_cols, sort=False):
        well = key[0] if isinstance(key, tuple) else key
        g = g.sort_values("DEPTH")
        for curve in curves:
            arr = g[curve].to_numpy(dtype=np.float32)
            depth = g["DEPTH"].to_numpy(dtype=np.float32)
            good = np.isfinite(arr)
            if good.sum() < window:
                continue
            arr = pd.Series(arr).interpolate(limit_direction="both").to_numpy(dtype=np.float32)
            continuous_filtered = {
                scale: apply_operator(arr, None, scale) for scale in scales
            } if response_scope == "continuous" else {}
            for start in range(0, len(arr) - window + 1, stride):
                hr = arr[start : start + window]
                # 窗口内实测样本比例须达到阈值。
                finite_fraction = float(good[start : start + window].mean())
                if finite_fraction < min_finite_fraction:
                    continue
                if not np.all(np.isfinite(hr)) or np.std(hr) < 1e-6:
                    continue
                for scale in scales:
                    if response_scope == "continuous":
                        low = continuous_filtered[scale][start : start + window : scale]
                        lr = upsample_to_length(low, len(hr), kind=upsample, scale=scale)
                        target = hr
                    else:
                        lr, target = make_lr_hr_pair(hr, scale=scale, upsample=upsample)
                    windows.append(
                        SRWindow(dataset, str(well), curve, split, scale, float(depth[start]), lr, target)
                    )
    if max_windows is not None and len(windows) > max_windows:
        # 带种子随机抽样, 而非只保留文件顺序靠前井的截断。
        rng = np.random.default_rng(subsample_seed)
        keep = rng.choice(len(windows), size=max_windows, replace=False)
        windows = [windows[i] for i in sorted(keep)]
    return windows




def apply_operator(hr: np.ndarray, spec: dict | None, sc: int) -> np.ndarray:
    """在按 sc 抽取前用指定的垂向响应形状对曲线低通滤波。
    kind 可为 gaussian (sigma_factor 缩放匹配宽度 sc/2)、box (width_factor 缩放宽度 sc)、
    triangle (宽 width_factor*2*sc 的 Bartlett 窗) 或 exp (尺度 sigma_factor*sc 的单边指数,
    即非对称响应); 只有因子为 1 的 gaussian 是基准算子, 其余为留出测试响应。"""
    from scipy.ndimage import convolve1d, gaussian_filter1d, uniform_filter1d

    kind = (spec or {}).get("kind", "gaussian")
    if kind == "gaussian":
        sigma = max(0.6, sc / 2.0) * float((spec or {}).get("sigma_factor", 1.0))
        return gaussian_filter1d(hr, sigma=sigma, mode="reflect")
    if kind == "box":
        w = max(2, int(round(sc * float((spec or {}).get("width_factor", 1.0)))))
        return uniform_filter1d(hr, size=w, mode="reflect")
    if kind == "triangle":
        half = max(2, int(round(sc * float((spec or {}).get("width_factor", 1.0)))))
        k = np.bartlett(2 * half + 1); k = k / k.sum()
        return convolve1d(hr, k, mode="reflect")
    if kind == "exp":
        tau = max(0.6, sc * float((spec or {}).get("sigma_factor", 0.5)))
        n = int(np.ceil(4 * tau)); k = np.exp(-np.arange(n + 1) / tau); k = k / k.sum()
        return convolve1d(hr, k, mode="reflect", origin=-(n // 2))
    raise ValueError(f"unknown operator kind {kind}")


def build_multicurve_windows(
    dataset: str,
    target: str,
    aux: list[str],
    scale: int,
    split: str,
    window: int,
    stride: int,
    data_root: str | Path = "data/processed",
    max_windows: int | None = None,
    min_finite_fraction: float = 0.95,
    subsample_seed: int = 2026,
    n_aux_slots: int = 3,
    degradations: list[dict] | None = None,
    aux_scale_divisor: int = 1,
    target_identity_curves: list[str] | tuple[str, ...] | None = None,
    target_identity_key: str | None = None,
    curve_scale_factor: dict | None = None,
    canonical_aux_curves: list[str] | tuple[str, ...] | None = None,
    upsample: str = "linear",
    response_scope: str = "window",
    context_window: int | None = None,
    include_target_observation_mask: bool = False,
    require_complete_aux: bool = True,
) -> list[SRWindow]:
    """构建 lr 中包含同一深度区间多条曲线的窗口。

    lr 形状为 (1 + 2 * n_aux_slots + n_identity, window): 第 0 行是目标曲线的插值
    低分辨率观测, 第 1..n_aux_slots 行是辅助曲线的插值观测 (曲线不足时补零), 随后
    n_aux_slots 行是 0/1 掩码, 可选的末尾行是沿深度重复的目标曲线 one-hot 标识;
    hr 为目标曲线, curve 为目标助记符。aux_scale_divisor > 1 时辅助曲线退化更轻
    (抽取 scale // aux_scale_divisor, 至少为 1), 即用同一仪器串上更高分辨率的曲线引导
    低分辨率曲线; degradations 列出训练窗口额外施加的替代算子, 每项复制一份窗口 (盲退化训练)。
    """
    from scipy.ndimage import gaussian_filter1d, uniform_filter1d

    if upsample not in {"linear", "pchip"}:
        raise ValueError(f"unknown upsample kind: {upsample}")
    if response_scope not in {"window", "continuous"}:
        raise ValueError(f"unknown response_scope {response_scope!r}")

    from .degradation import upsample_to_length

    target_window = int(window)
    context_window = int(context_window or target_window)
    if context_window < target_window:
        raise ValueError("context_window must be >= window")
    if context_window % 2 != target_window % 2:
        raise ValueError("context_window and window must have the same parity")
    context_offset = (context_window - target_window) // 2

    def extract(arr: np.ndarray, start: int, length: int) -> np.ndarray:
        """提取上下文区间, 物理边界处做反射填充。"""
        left = max(0, -start)
        right = max(0, start + length - len(arr))
        if left or right:
            padded = np.pad(arr, (left, right), mode="reflect")
            start += left
            return padded[start:start + length].astype(np.float32, copy=True)
        return arr[start:start + length].astype(np.float32, copy=True)

    df = load_panel(dataset, data_root)
    if canonical_aux_curves is not None:
        # 按规范顺序固定每个引导槽位对应的曲线。
        slot_curves = [c for c in canonical_aux_curves if c in df.columns]
        n_aux_slots = len(slot_curves)
        curves = [target] + slot_curves
    else:
        slot_curves = [c for c in aux if c in df.columns and c != target][:n_aux_slots]
        curves = [target] + slot_curves
    identity_curves = tuple(target_identity_curves or ())
    identity_key = target if target_identity_key is None else str(target_identity_key)
    if identity_curves and identity_key not in identity_curves:
        raise ValueError(
            f"{identity_key!r} is not present in target_identity_curves={identity_curves!r}"
        )
    df = df[df["SPLIT"].astype(str).str.lower() == split.lower()].copy()
    group_cols = ["WELL", "SLICE_ID"] if "SLICE_ID" in df.columns else ["WELL"]

    aux_scale = max(1, scale // max(1, int(aux_scale_divisor)))
    # 物理有序测井组合: 每条曲线按自身仪器尺度 scale * factor (取整, 至少 1) 退化, 与是否为目标无关。
    csf = curve_scale_factor or {}
    tgt_scale = max(1, int(round(scale * float(csf.get(target, 1.0))))) if csf else scale
    aux_scales = [max(1, int(round(scale * float(csf.get(c, 1.0))))) if csf else aux_scale for c in slot_curves]

    # 匹配响应在切窗前于完整连续切片上生成; 替代算子按窗口生成。
    continuous_response: dict[int, np.ndarray] = {}

    def degrade(
        hr: np.ndarray,
        spec: dict | None,
        sc: int = scale,
        *,
        start: int | None = None,
        row: int | None = None,
    ) -> np.ndarray:
        if sc == 1:
            return hr.astype(np.float32).copy()
        if response_scope == "continuous" and spec is None and start is not None and row is not None:
            observed = extract(continuous_response[row], start, len(hr))
            low = observed[::sc]
            return upsample_to_length(low.astype(np.float32), len(hr), kind=upsample, scale=sc)
        lo = apply_operator(hr, spec, sc)[::sc]
        return upsample_to_length(lo.astype(np.float32), len(hr), kind=upsample, scale=sc)

    windows: list[SRWindow] = []
    for key, g in df.groupby(group_cols, sort=False):
        well = key[0] if isinstance(key, tuple) else key
        g = g.sort_values("DEPTH")
        depth = g["DEPTH"].to_numpy(dtype=np.float32)
        arrs, goods = [], []
        for c in curves:
            a = g[c].to_numpy(dtype=np.float32)
            goods.append(np.isfinite(a))
            arrs.append(pd.Series(a).interpolate(limit_direction="both").to_numpy(dtype=np.float32))
        if response_scope == "continuous":
            continuous_response = {
                0: apply_operator(arrs[0], None, tgt_scale),
                **{
                    j + 1: apply_operator(arrs[j + 1], None, aux_scales[j])
                    for j in range(len(slot_curves))
                },
            }
        if goods[0].sum() < window:
            continue
        for target_start in range(0, len(arrs[0]) - target_window + 1, stride):
            context_start = target_start - context_offset
            context_end = context_start + context_window
            eval_start = target_start - context_start
            target_good = extract(goods[0].astype(np.float32), target_start, target_window)
            if any(float(target_good.mean()) < min_finite_fraction for _ in [0]):
                continue
            aux_ok = True
            for gd in goods[1:]:
                frac = float(extract(gd.astype(np.float32), target_start, target_window).mean())
                if require_complete_aux:
                    if frac < min_finite_fraction:
                        aux_ok = False
                        break
                elif 0.0 < frac < min_finite_fraction:
                    # 部分缺失仍不可靠; 完全缺失的引导曲线保持掩码并允许通过。
                    aux_ok = False
                    break
            if not aux_ok:
                continue
            hr = extract(arrs[0], context_start, context_window)
            if not np.all(np.isfinite(hr)) or np.std(hr) < 1e-6:
                continue
            aux_hr = np.zeros((n_aux_slots, context_window), dtype=np.float32)
            for j, (slot_curve, a) in enumerate(zip(slot_curves, arrs[1:], strict=True)):
                aux_hr[j] = extract(a, context_start, context_window)
            target_observation_mask = None
            if include_target_observation_mask:
                # 掩码只由已知采样网格导出; 物理切片边界处的反射上下文不计为实测观测。
                positions = np.arange(context_start, context_end, dtype=np.int64)
                target_observation_mask = (
                    (positions >= 0)
                    & (positions < len(arrs[0]))
                    # 掩码相位以 context_start 为零点, 与 degrade 的抽取相位一致。
                    & ((positions - int(context_start)) % int(tgt_scale) == 0)
                ).astype(np.float32)
            for spec in [None] + list(degradations or []):
                observation_mask_index = 1 + 2 * n_aux_slots
                rows = np.zeros(
                    (
                        1
                        + 2 * n_aux_slots
                        + int(include_target_observation_mask)
                        + len(identity_curves),
                        context_window,
                    ),
                    dtype=np.float32,
                )
                rows[0] = degrade(hr, spec, tgt_scale, start=context_start, row=0)
                for j, (slot_curve, a) in enumerate(zip(slot_curves, arrs[1:], strict=True)):
                    # 目标曲线自身的槽位留空 (目标已在第 0 行), 掩码为 0。
                    if slot_curve == target:
                        continue
                    seg = extract(a, context_start, context_window)
                    if not np.all(np.isfinite(seg)):
                        # 完全缺失的辅助曲线用零输入和 mask=0 表示;
                        # 部分观测的引导曲线同样排除, 以免插值引入未追踪的目标信号。
                        if not np.isfinite(seg).any():
                            continue
                        continue
                    rows[1 + j] = degrade(seg, spec, aux_scales[j], start=context_start, row=j + 1)
                    rows[1 + n_aux_slots + j] = 1.0
                if target_observation_mask is not None:
                    rows[observation_mask_index] = target_observation_mask
                if identity_curves:
                    rows[
                        observation_mask_index
                        + int(include_target_observation_mask)
                        + identity_curves.index(identity_key)
                    ] = 1.0
                windows.append(
                    SRWindow(
                        dataset,
                        str(well),
                        target,
                        split,
                        tgt_scale,
                        float(depth[target_start]),
                        rows,
                        hr.copy(),
                        aux_hr,
                        tuple(slot_curves),
                        identity_curves or None,
                        eval_start,
                        target_window,
                        target_observation_mask.copy()
                        if target_observation_mask is not None
                        else None,
                    )
                )
    if max_windows is not None and len(windows) > max_windows:
        rng = np.random.default_rng(subsample_seed)
        keep = rng.choice(len(windows), size=max_windows, replace=False)
        windows = [windows[i] for i in sorted(keep)]
    return windows
