"""对引导重建结果做下游地层边界拾取评估。
读取 run_benchmark_v2.py 在 save_predictions: true 下写出的 predictions/*.npz (文件名 {dataset}_{curve}_{scale}_{method}.npz),
用 eval_boundary_downstream.py 的检测器、容差和匹配规则打分, 输出 CSV。

用法:
    PYTHONPATH=src python scripts/eval_boundary_guided.py \
        --pred-root results/v2_evidence --out results/v2_summary/downstream_guided.csv
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("ebd", Path(__file__).with_name("eval_boundary_downstream.py"))
_ebd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ebd)


def score(pred: np.ndarray, hr: np.ndarray, tolerance: int = 2) -> dict:
    tp = fp = fn = 0
    errors = []
    for i in range(len(hr)):
        t = _ebd.detect_boundaries(hr[i])
        q = _ebd.detect_boundaries(pred[i])
        a, b, c, e = _ebd.match_boundaries(t, q, tolerance=tolerance)
        tp += a; fp += b; fn += c
        if np.isfinite(e):
            errors.append(e)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return dict(boundary_precision=precision, boundary_recall=recall, boundary_f1=f1,
                boundary_pos_error=float(np.mean(errors)) if errors else np.nan, n_windows=len(hr))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", default="results/v2_evidence")
    ap.add_argument("--out", default="results/v2_summary/downstream_guided.csv")
    ap.add_argument("--tolerance", type=int, default=2)
    args = ap.parse_args()
    rows = []
    seen_obs = set()
    for f in sorted(Path(args.pred_root).glob("*/seed*/predictions/*.npz")):
        parts = f.stem.split("_")
        dataset, curve, scale, method = parts[0], parts[1], int(parts[2]), "_".join(parts[3:])
        seed = f.parent.parent.name
        d = np.load(f)
        rows.append(dict(dataset=dataset, curve=curve, scale=scale, method=method, seed=seed,
                         **score(d["test_pred"], d["test_hr"], args.tolerance)))
        print(f"{dataset:9s} {curve:5s} {scale}x {method:15s} F1 {rows[-1]['boundary_f1']:.4f} "
              f"pos {rows[-1]['boundary_pos_error']:.3f}")
        # 模型输入的插值观测, 作为各方法必须超过的下限
        key = (dataset, curve, scale, seed)
        if key not in seen_obs:
            seen_obs.add(key)
            rows.append(dict(dataset=dataset, curve=curve, scale=scale, method="observation", seed=seed,
                             **score(d["test_lr"][:, 0], d["test_hr"], args.tolerance)))
            print(f"{dataset:9s} {curve:5s} {scale}x {'observation':15s} F1 {rows[-1]['boundary_f1']:.4f}")
    t = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out, index=False)
    piv = t.pivot_table(index=["dataset", "scale"], columns="method", values="boundary_f1")
    print("\nseeds:", sorted(t.seed.unique()))
    print("\nF1 by panel and scale (seed means):\n", piv.round(4).to_string())
    if t.seed.nunique() > 1:
        spread = (t.groupby(["dataset", "curve", "scale", "method"]).boundary_f1
                   .agg(lambda x: 100 * (x.max() - x.min()) / x.mean()))
        print(f"\nthree-seed F1 range as percent of mean: median {spread.median():.1f}, "
              f"90th pct {spread.quantile(0.9):.1f}, max {spread.max():.1f}")
    if {"geosr2", "ssm"} <= set(piv.columns):
        print("\nSuiteSR vs single-curve state-space, F1 change (%):\n",
              (100 * (piv["geosr2"] - piv["ssm"]) / piv["ssm"]).round(2).to_string())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
