"""下游地层边界拾取评估。
在高分辨率曲线和各方法重建上以显著 |gradient| 峰检测边界, 在深度容差内匹配后计算 precision/recall/F1 和位置误差。
读取 run_benchmark.py 在 save_predictions: true 下写出的 predictions/*.npz (键 test_pred, test_hr), 输出 CSV。

用法:
    PYTHONPATH=src python scripts/eval_boundary_downstream.py \
        --pred-dir results/<run>/predictions --out results/<run>/boundary_downstream.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def detect_boundaries(curve: np.ndarray, prominence_quantile: float = 0.80, min_distance: int = 4) -> np.ndarray:
    grad = np.abs(np.gradient(curve.astype(np.float64)))
    if grad.max() <= 1e-12:
        return np.array([], dtype=int)
    prominence = np.quantile(grad, prominence_quantile)
    peaks, _ = find_peaks(grad, height=prominence, distance=min_distance)
    return peaks


def match_boundaries(true_idx: np.ndarray, pred_idx: np.ndarray, tolerance: int = 2) -> tuple[int, int, int, float]:
    """容差内贪心一对一匹配, 返回 tp, fp, fn 和平均位置误差。"""
    if len(true_idx) == 0:
        return 0, len(pred_idx), 0, np.nan
    if len(pred_idx) == 0:
        return 0, 0, len(true_idx), np.nan
    used = np.zeros(len(pred_idx), dtype=bool)
    tp = 0
    errors = []
    for t in true_idx:
        dist = np.abs(pred_idx - t)
        dist[used] = tolerance + 10**6
        j = int(np.argmin(dist))
        if dist[j] <= tolerance:
            used[j] = True
            tp += 1
            errors.append(abs(int(pred_idx[j]) - int(t)))
    fp = int((~used).sum())
    fn = len(true_idx) - tp
    return tp, fp, fn, float(np.mean(errors)) if errors else np.nan


STEM_RE = re.compile(r"^(?P<dataset>[a-z0-9]+)_(?P<curves>[A-Z\-]+)_(?P<scales>[\d\-]+)_(?P<method>.+?)_(?P<suffix>raw|final)\.npz$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--prominence-quantile", type=float, default=0.80)
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.pred_dir).glob("*.npz")):
        m = STEM_RE.match(path.name)
        if not m:
            continue
        data = np.load(path)
        if "test_pred" not in data or "test_hr" not in data:
            continue
        preds, hrs = data["test_pred"], data["test_hr"]
        tp = fp = fn = 0
        pos_errors = []
        for pred, hr in zip(preds, hrs, strict=True):
            true_idx = detect_boundaries(hr, args.prominence_quantile)
            pred_idx = detect_boundaries(pred, args.prominence_quantile)
            wtp, wfp, wfn, err = match_boundaries(true_idx, pred_idx, args.tolerance)
            tp += wtp
            fp += wfp
            fn += wfn
            if np.isfinite(err):
                pos_errors.append(err)
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
            else np.nan
        )
        rows.append(
            {
                "dataset": m["dataset"],
                "curves": m["curves"],
                "scales": m["scales"],
                "method": m["method"],
                "suffix": m["suffix"],
                "n_windows": len(preds),
                "boundary_precision": precision,
                "boundary_recall": recall,
                "boundary_f1": f1,
                "boundary_pos_error": float(np.mean(pos_errors)) if pos_errors else np.nan,
            }
        )
    df = pd.DataFrame(rows).sort_values(["dataset", "curves", "scales", "boundary_f1"], ascending=[True, True, True, False])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
