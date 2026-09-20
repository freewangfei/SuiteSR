"""现场面板的逐井分解与窗口级 bootstrap。
对 12 个现场任务, 分别在每口测试井上计算验证集选出的 SuiteSR 相对最优单曲线和最优引导对比方法的 MAE 增益,
并对合并增益做井内重采样的配对 bootstrap (10000 次, seed 2026), 结果写入 field_well_bootstrap.csv。

用法:
    PYTHONPATH=src python scripts/analyze_field_wells.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

SINGLE = ["lstm", "transformer", "dmcnet", "cascade_sr", "ssm"]
GUIDED = [m + "_multi" for m in SINGLE] + ["lstm_native", "dmcnet_native", "ssm_native"]
ENCODED = ["dmcnet_morph_native", "ssm_morph_native"]
ROOTS = ["results/v2_guided", "results/v5_native", "results/v5_morph"]
SUM = Path("results/v2_summary")
B = 10000


def load_windows(dataset: str = "cnfield") -> pd.DataFrame:
    frames = []
    for root in ROOTS:
        for f in sorted(glob.glob(f"{root}/{dataset}/seed*/window_metrics.csv")):
            frames.append(pd.read_csv(f).assign(seed=Path(f).parent.name))
    d = pd.concat(frames, ignore_index=True)
    return d[d.condition.isna() | (d.condition == "matched")] if "condition" in d else d


def selected_form() -> dict:
    """从 guide_selection.csv 读取每个任务由验证规则选定的 SuiteSR 形式。"""
    sel = pd.read_csv(SUM / "guide_selection.csv")
    return {(r.dataset, r.curve, int(r.scale)): r.selected for r in sel.itertuples()}


def main() -> None:
    d = load_windows()
    sel = selected_form()
    # 对 seed 取平均, 使每个 (well, depth_start, task, method) 成为一个配对观测
    key = ["curve", "scale", "well", "depth_start", "method"]
    w = d.groupby(key, as_index=False).mae.mean()

    rng = np.random.default_rng(2026)
    rows = []
    for (curve, scale), sub in w.groupby(["curve", "scale"]):
        piv = sub.pivot_table(index=["well", "depth_start"], columns="method", values="mae")
        form = sel.get(("cnfield", curve, int(scale)), "geosr2")
        if form not in piv:
            continue
        ours = piv[form].to_numpy()
        for group, members in (("single", SINGLE), ("guided", GUIDED), ("encoded", ENCODED)):
            have = [m for m in members if m in piv]
            if not have:
                continue
            # 按任务均值选出该组最优对比方法
            best = min(have, key=lambda m: piv[m].mean())
            ref = piv[best].to_numpy()
            wells = piv.index.get_level_values("well").to_numpy()
            gain = 100.0 * (ref.mean() - ours.mean()) / ref.mean()
            per_well = {}
            for wl in np.unique(wells):
                k = wells == wl
                per_well[wl] = 100.0 * (ref[k].mean() - ours[k].mean()) / ref[k].mean()
            # bootstrap: 井内重采样窗口, 保持配对
            idx_by_well = [np.flatnonzero(wells == wl) for wl in np.unique(wells)]
            boot = np.empty(B)
            for b in range(B):
                take = np.concatenate([rng.choice(ix, ix.size, replace=True) for ix in idx_by_well])
                boot[b] = 100.0 * (ref[take].mean() - ours[take].mean()) / ref[take].mean()
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(dict(curve=curve, scale=int(scale), group=group, ref=best, form=form,
                             gain=gain, lo=lo, hi=hi, sig=bool(lo > 0),
                             **{f"well_{k}": v for k, v in per_well.items()}))
    out = pd.DataFrame(rows).sort_values(["group", "scale", "curve"])
    SUM.mkdir(parents=True, exist_ok=True)
    out.to_csv(SUM / "field_well_bootstrap.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

    print("\nsummary by group and scale (pooled gain, and how many of the 4 tasks have a CI above zero)")
    for (g, s), sub in out.groupby(["group", "scale"]):
        wc = [c for c in out.columns if c.startswith("well_")]
        agree = int(((sub[wc[0]] > 0) & (sub[wc[1]] > 0)).sum()) if len(wc) == 2 else -1
        print(f"  {g:8s} {s}x  gain {sub.gain.mean():5.2f}%  CI>0 on {sub.sig.sum()}/{len(sub)}"
              f"  both wells agree in sign on {agree}/{len(sub)}"
              f"  per-well means {sub[wc[0]].mean():5.2f} / {sub[wc[1]].mean():5.2f}")
    print("\nwells:", [c[5:] for c in out.columns if c.startswith("well_")])


if __name__ == "__main__":
    main()
