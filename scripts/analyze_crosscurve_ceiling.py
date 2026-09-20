"""跨曲线信息上限分析: 用全分辨率引导曲线的高通分量对目标曲线的插值残差做逐窗口线性回归 (oracle, 在测试窗口自身拟合系数), 统计可解释比例。
按数据面板和目标曲线在 4x 下输出 CSV。

用法:
    PYTHONPATH=src python scripts/analyze_crosscurve_ceiling.py --out results/v2_summary/crosscurve_ceiling.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from wellsr.data import build_multicurve_windows

CURVES = {"geolink": ["GR", "DTC", "RHOB", "NPHI"], "taranaki": ["GR", "DTC", "RHOB"],
          "teapot": ["GR", "RHOB", "NPHI"], "cnfield": ["GR", "DTC", "RHOB", "NPHI"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v2_summary/crosscurve_ceiling.csv")
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args()
    rows = []
    for ds, curves in CURVES.items():
        for tgt in curves:
            aux = [c for c in curves if c != tgt]
            ws = build_multicurve_windows(ds, tgt, aux, args.scale, "test", 128, 128, max_windows=300, aux_scale_divisor=1000)
            if not ws:
                continue
            hr = np.stack([w.hr for w in ws]); lr = np.stack([w.lr[0] for w in ws])
            A = np.stack([w.aux_hr for w in ws])[:, :len(aux)]
            res = hr - lr
            X = np.stack([A[:, j] - gaussian_filter1d(A[:, j], 2.0, axis=1) for j in range(len(aux))], -1)
            base = float(np.abs(res).mean())

            def oracle(cols):
                maes = []
                for i in range(len(hr)):
                    Xi = np.c_[X[i][:, cols], np.ones(hr.shape[1])]
                    b, *_ = np.linalg.lstsq(Xi, res[i], rcond=None)
                    maes.append(np.abs(res[i] - Xi @ b).mean())
                return float(np.mean(maes))

            orc = oracle(list(range(len(aux))))
            row = dict(dataset=ds, curve=tgt, residual_mae=base, oracle_mae=orc, explained_pct=100 * (1 - orc / base))
            # 单引导曲线 oracle
            for j, g in enumerate(aux):
                row[f"explained_{g}_only"] = 100 * (1 - oracle([j]) / base)
            rows.append(row)
            print(f"{ds:9s} {tgt:5s} residual {base:.4f} -> oracle {orc:.4f} ({row['explained_pct']:.0f}% explained; "
                  + ", ".join(f"{g} alone {row[f'explained_{g}_only']:.0f}%" for g in aux) + ")")
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
