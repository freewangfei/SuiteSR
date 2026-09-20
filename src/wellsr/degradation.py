from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.ndimage import gaussian_filter1d


def anti_alias_downsample(x: np.ndarray, scale: int, sigma: float | None = None) -> np.ndarray:
    """对一维曲线先低通滤波再抽取。"""
    if scale < 1:
        raise ValueError("scale must be >= 1")
    x = np.asarray(x, dtype=np.float32)
    if scale == 1:
        return x.copy()
    if sigma is None:
        sigma = max(0.6, scale / 2.0)
    filtered = gaussian_filter1d(x, sigma=sigma, mode="reflect")
    return filtered[::scale].astype(np.float32)


def upsample_to_length(
    x_lr: np.ndarray, length: int, kind: str = "linear", scale: int | None = None
) -> np.ndarray:
    """将低分辨率曲线插值回高分辨率网格: LR 样本 i 置于 HR 位置 i * s, (n - 1) * s 之后外推。"""
    x_lr = np.asarray(x_lr, dtype=np.float32)
    n = len(x_lr)
    if n == length:
        return x_lr.copy()
    if scale is None:
        # 推断 anti_alias_downsample 使用的抽取步长。
        scale = int(round(length / n)) if n > 0 else 1
        if scale < 1 or len(range(0, length, scale)) != n:
            raise ValueError(
                f"cannot infer decimation stride for LR length {n} and HR length {length}; pass scale explicitly"
            )
    src = (np.arange(n, dtype=np.float32)) * float(scale)
    dst = np.arange(length, dtype=np.float32)
    if kind == "pchip":
        return PchipInterpolator(src, x_lr, extrapolate=True)(dst).astype(np.float32)
    f = interp1d(src, x_lr, kind=kind, fill_value="extrapolate", assume_sorted=True)
    return f(dst).astype(np.float32)


def make_lr_hr_pair(x_hr: np.ndarray, scale: int, upsample: str = "linear") -> tuple[np.ndarray, np.ndarray]:
    """生成插值后的 LR 输入与 HR 目标。"""
    hr = np.asarray(x_hr, dtype=np.float32)
    lr = anti_alias_downsample(hr, scale)
    lr_up = upsample_to_length(lr, len(hr), kind=upsample, scale=scale)
    return lr_up.astype(np.float32), hr
