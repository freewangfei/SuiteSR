from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    mae: float
    rmse: float
    r2: float
    pearson: float
    psnr: float
    grad_mae: float
    hf_rmse: float
    spectral_angle: float


# 按 (dataset, curve) 固定的 PSNR 参考范围, 由基准驱动脚本从训练集注册,
# 使 PSNR 在不同窗口/方法间可比。
DATA_RANGES: dict[tuple[str, str], float] = {}


def register_data_ranges(train_windows) -> None:
    grouped: dict[tuple[str, str], list[float]] = {}
    for w in train_windows:
        grouped.setdefault((w.dataset, w.curve), []).extend(
            [float(np.min(w.hr)), float(np.max(w.hr))]
        )
    for key, vals in grouped.items():
        DATA_RANGES[key] = float(max(vals) - min(vals))


def metric_row(w, pred: np.ndarray, method: str, extra: dict | None = None) -> dict:
    """生成一行 window_metrics, 带显式标识键以支持配对检验。"""
    start = int(getattr(w, "eval_start", 0) or 0)
    length = getattr(w, "eval_length", None)
    if length is not None:
        stop = start + int(length)
        target = np.asarray(w.hr)[start:stop]
        prediction = np.asarray(pred)[start:stop]
    else:
        target = w.hr
        prediction = pred
    m = compute_metrics(target, prediction, scale=w.scale, data_range=DATA_RANGES.get((w.dataset, w.curve)))
    row = dict(extra) if extra else {}
    row.update(
        {
            "dataset": w.dataset,
            "curve": w.curve,
            "scale": w.scale,
            "well": w.well,
            "depth_start": w.depth_start,
            "method": method,
            **m.__dict__,
        }
    )
    return row


def _as_float(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(-1)


def highpass(x: np.ndarray) -> np.ndarray:
    """一阶差分; 用于梯度指标, 不用于高频带指标。"""
    x = _as_float(x)
    return np.diff(x, prepend=x[0])


def super_nyquist_component(x: np.ndarray, scale: int) -> np.ndarray:
    """提取高于抽取信号奈奎斯特频率 1 / (2 * scale) 的分量。"""
    x = _as_float(x)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0)
    spec[freqs <= 0.5 / float(scale)] = 0.0
    return np.fft.irfft(spec, n=len(x))


def spectral_angle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算排除 DC 分量后幅度谱之间的夹角。

    若包含 DC, 各方法共享的窗口均值会主导内积, 使所有方法的夹角都压向零。
    """
    a = np.abs(np.fft.rfft(_as_float(y_true)))[1:]
    b = np.abs(np.fft.rfft(_as_float(y_pred)))[1:]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return float("nan")
    cos = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(math.acos(cos))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scale: int | None = None,
    data_range: float | None = None,
) -> MetricResult:
    """计算窗口级重建指标。

    scale 决定 hf_rmse 的超奈奎斯特频带, 缺省时用一阶差分定义; data_range 为 PSNR
    的参考范围, 缺省时用窗口目标范围。退化窗口返回 NaN。
    """
    yt = _as_float(y_true)
    yp = _as_float(y_pred)
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - float(np.sum(err**2)) / denom if denom > 1e-12 else float("nan")
    pearson = (
        float(np.corrcoef(yt, yp)[0, 1]) if np.std(yt) > 1e-12 and np.std(yp) > 1e-12 else float("nan")
    )
    if data_range is None:
        data_range = float(np.max(yt) - np.min(yt))
    psnr = float(20 * np.log10(data_range / (rmse + 1e-12))) if data_range > 1e-12 else float("nan")
    grad_mae = float(np.mean(np.abs(highpass(yt) - highpass(yp))))
    if scale is not None and scale > 1:
        hf_true = super_nyquist_component(yt, scale)
        hf_pred = super_nyquist_component(yp, scale)
        hf_rmse = float(np.sqrt(np.mean((hf_true - hf_pred) ** 2)))
    else:
        hf_rmse = float(np.sqrt(np.mean((highpass(yt) - highpass(yp)) ** 2)))
    return MetricResult(mae, rmse, r2, pearson, psnr, grad_mae, hf_rmse, spectral_angle(yt, yp))
