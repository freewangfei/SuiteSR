"""检测各面板源曲线在标称采样步长上是否含有信息。
对测试井逐曲线统计二阶差分为零的比例、粗网格 Nyquist 以上的能量占比, 以及从未滤波 stride-2 样本线性插值的 MAE, 结果写入 CSV。

用法:
    PYTHONPATH=src python scripts/audit_source_resolution.py \
        --out results/fair_source_resolution_audit.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from wellsr.data import build_windows

DATASETS = {"geolink": ["GR", "DTC", "RHOB", "NPHI"], "taranaki": ["GR", "DTC", "RHOB"],
            "teapot": ["GR", "RHOB", "NPHI"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/fair_source_resolution_audit.csv")
    ap.add_argument("--zero-tol", type=float, default=1e-4,
                    help="二阶差分低于此阈值 (归一化单位) 视为零")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="额外的 dataset:curve 对, 如 cnfield:GR")
    args = ap.parse_args()
    pairs = [(d, c) for d, cs in DATASETS.items() for c in cs]
    pairs += [tuple(e.split(":")) for e in args.extra]

    rows = []
    for ds, curve in pairs:
        ws = build_windows(ds, [curve], [2], "test", 128, 128, max_windows=512)
        hr = np.stack([w.hr for w in ws]).astype(np.float64)
        d2 = np.abs(np.diff(hr, n=2, axis=1))
        frac_zero = float((d2 < args.zero_tol).mean())
        # 2x 粗网格 Nyquist 以上的能量 (归一化频率 > 0.25)
        x = hr - hr.mean(axis=1, keepdims=True)
        spec = np.abs(np.fft.rfft(x * np.hanning(hr.shape[1]), axis=1)) ** 2
        f = np.fft.rfftfreq(hr.shape[1])
        frac_hf = float(spec[:, f > 0.25].sum() / spec.sum())
        # 未滤波 stride-2 样本的线性插值
        idx = np.arange(0, hr.shape[1], 2)
        full = np.arange(hr.shape[1])
        lin = np.stack([np.interp(full, idx, h[idx]) for h in hr])
        mae_lin = float(np.abs(lin - hr).mean())
        rows.append(dict(dataset=ds, curve=curve, n_windows=len(ws),
                         frac_zero_second_diff=frac_zero, energy_frac_above_half_nyquist=frac_hf,
                         mae_linear_from_stride2=mae_lin))
        print(f"{ds:9s} {curve:5s} zero-d2 {frac_zero:6.1%}  HF energy {frac_hf:6.2%}  "
              f"linear-from-stride-2 MAE {mae_lin:.4f}")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
