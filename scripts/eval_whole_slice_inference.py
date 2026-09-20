"""整段切片推理评估: 窗口拼接缝、重叠平均与单次整段推理。
每个方法在基准窗口上训练一次, 再以三种模式重建完整测试切片: tiled (不重叠 128 点窗口拼接)、overlap (stride 32 + Hann 加权)、whole (整段单次前向)。
对每种模式输出切片级 MAE、Grad-MAE 以及拼接缝比值 (缝处一阶差分绝对值均值 / 非缝处), 写入 CSV。

用法:
    PYTHONPATH=src python scripts/eval_whole_slice_inference.py \
        --dataset geolink --curve GR --scales 4 8 --methods geosr geosr_deploy ssm \
        --out results/fair_whole_slice/geolink_GR.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wellsr.data import SRWindow, build_windows, load_panel
from wellsr.degradation import make_lr_hr_pair
from wellsr.train import main_prediction, resolve_device, train_model

W = 128
KW = dict(batch_size=256, epochs=300, min_epochs=0, min_train_epochs=100, patience=30,
          learning_rate=6e-4, device="auto")
GEO_LOSS = {"l1": 1.0, "mse": 0.35, "gradient": 0.12, "spectral": 0.012}
BASE_LOSS = {"l1": 1.0, "gradient": 0.15, "spectral": 0.02}
CFG = {
    "geosr": dict(channels=64, depth=4, modes=48, loss_weights=GEO_LOSS),
    "geosr_deploy": dict(channels=32, depth=3, modes=24, loss_weights=GEO_LOSS),
    "ssm": dict(channels=48, depth=3, modes=32, loss_weights=BASE_LOSS),
    "dmcnet": dict(channels=48, depth=3, modes=32, loss_weights=BASE_LOSS),
    "lstm": dict(channels=48, depth=3, modes=32, loss_weights=BASE_LOSS),
}


def test_slices(dataset: str, curve: str, scale: int, min_len: int = 2 * W):
    """返回完整测试切片列表 (well, lr, hr), 含退化观测。"""
    df = load_panel(dataset)
    df = df[df["SPLIT"].astype(str).str.lower() == "test"]
    group_cols = ["WELL", "SLICE_ID"] if "SLICE_ID" in df.columns else ["WELL"]
    out = []
    for key, g in df.groupby(group_cols, sort=False):
        g = g.sort_values("DEPTH")
        arr = g[curve].to_numpy(dtype=np.float32)
        if len(arr) < min_len or np.isfinite(arr).mean() < 0.95:
            continue
        arr = pd.Series(arr).interpolate(limit_direction="both").to_numpy(dtype=np.float32)
        n = (len(arr) // W) * W          # 截断使平铺整除
        arr = arr[:n]
        lr, hr = make_lr_hr_pair(arr, scale=scale, upsample="linear")
        out.append((str(key[0] if isinstance(key, tuple) else key), lr, hr))
    return out


def predict(model, lr_rows: list[np.ndarray], device, batch: int = 256) -> list[np.ndarray]:
    import torch

    preds = []
    lengths = {len(r) for r in lr_rows}
    with torch.no_grad():
        for L in sorted(lengths):
            idx = [i for i, r in enumerate(lr_rows) if len(r) == L]
            for s in range(0, len(idx), batch):
                chunk = [lr_rows[i] for i in idx[s:s + batch]]
                mean = np.array([c.mean() for c in chunk], np.float32)
                std = np.array([c.std() + 1e-6 for c in chunk], np.float32)
                x = torch.from_numpy(np.stack([(c - m) / sd for c, m, sd in zip(chunk, mean, std)])[:, None, :]).to(device)
                y = main_prediction(model(x)).cpu().numpy()[:, 0, :]
                for i, p, m, sd in zip(idx[s:s + batch], y, mean, std):
                    preds.append((i, p * sd + m))
    preds.sort(key=lambda t: t[0])
    return [p for _, p in preds]


def reconstruct(model, lr: np.ndarray, device, mode: str, stride: int = 32) -> np.ndarray:
    n = len(lr)
    if mode == "whole":
        return predict(model, [lr], device)[0]
    if mode == "tiled":
        rows = [lr[s:s + W] for s in range(0, n, W)]
        return np.concatenate(predict(model, rows, device))
    starts = list(range(0, n - W + 1, stride))
    rows = [lr[s:s + W] for s in starts]
    preds = predict(model, rows, device)
    taper = np.hanning(W + 2)[1:-1].astype(np.float64) + 1e-3
    acc = np.zeros(n); wsum = np.zeros(n)
    for s, p in zip(starts, preds):
        acc[s:s + W] += taper * p; wsum[s:s + W] += taper
    return (acc / wsum).astype(np.float32)


def seam_jump(x: np.ndarray) -> tuple[float, float]:
    """返回拼接缝处与非缝处的一阶差分绝对值均值。"""
    d = np.abs(np.diff(x))
    seams = np.arange(W, len(x), W) - 1
    mask = np.zeros(len(d), bool); mask[seams] = True
    return float(d[mask].mean()), float(d[~mask].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="geolink")
    ap.add_argument("--curve", default="GR")
    ap.add_argument("--scales", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--methods", nargs="+", default=["geosr", "geosr_deploy", "ssm"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results/fair_whole_slice/geolink_GR.csv")
    args = ap.parse_args()
    import torch

    device = resolve_device("auto")
    rows = []
    for scale in args.scales:
        tr = build_windows(args.dataset, [args.curve], [scale], "train", W, W, max_windows=1024)
        va = build_windows(args.dataset, [args.curve], [scale], "val", W, W, max_windows=256)
        te = build_windows(args.dataset, [args.curve], [scale], "test", W, W, max_windows=512)
        slices = test_slices(args.dataset, args.curve, scale)
        print(f"{args.dataset} {args.curve} {scale}x: {len(slices)} test slices, "
              f"{sum(len(h) for _, _, h in slices)} samples")
        for method in args.methods:
            torch.manual_seed(args.seed); np.random.seed(args.seed)
            r = train_model(method, tr, va, te, **CFG[method], **KW)
            model = r.model
            model.eval()
            for mode in ("tiled", "overlap", "whole"):
                maes, gmaes, seams, offs, rseams, roffs = [], [], [], [], [], []
                for k, (well, lr, hr) in enumerate(slices):
                    p = reconstruct(model, lr, device, mode)
                    if k == 0:
                        # 保存每个 method/mode 的第一个测试区间供绘图
                        np.savez(Path(args.out).with_suffix("").as_posix() + f"_{scale}x_{method}_{mode}.npz",
                                 well=well, lr=lr, hr=hr, pred=p)
                    maes.append(np.abs(p - hr).mean())
                    gmaes.append(np.abs(np.diff(p) - np.diff(hr)).mean())
                    a, b = seam_jump(p); seams.append(a); offs.append(b)
                    a, b = seam_jump(hr); rseams.append(a); roffs.append(b)
                # 拼接缝比值: 缝处 |一阶差分| / 非缝处, 重建与参考 (应接近 1) 各算一次
                rows.append(dict(dataset=args.dataset, curve=args.curve, scale=scale, method=method,
                                 mode=mode, n_slices=len(slices), mae=float(np.mean(maes)),
                                 grad_mae=float(np.mean(gmaes)), seam_jump=float(np.mean(seams)),
                                 offseam_jump=float(np.mean(offs)),
                                 seam_ratio=float(np.mean(seams) / np.mean(offs)),
                                 seam_ratio_reference=float(np.mean(rseams) / np.mean(roffs))))
                print(f"  {method:13s} {mode:8s} MAE {rows[-1]['mae']:.5f} Grad-MAE {rows[-1]['grad_mae']:.5f} "
                      f"seam ratio {rows[-1]['seam_ratio']:.2f} (ref {rows[-1]['seam_ratio_reference']:.2f})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
