"""引导基准的 seed 集成评估 (部署配置)。
读取 run_benchmark_v2.py 在同一 data seed、不同 model_seed 下写出的预测文件, 平均三个 model seed 的测试预测,
在相同窗口上用基准指标为单模型和集成打分; 对比方法的分数取自 results/v2_guided, v5_native, v5_morph。
输出到 results/v6_summary: ensemble_task_metrics.csv, ensemble_margins.csv, ensemble_summary.csv 和 ensemble_table.tex。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wellsr.metrics import compute_metrics  # noqa: E402

DATA_SEEDS = (2026, 2027, 2028)
MODEL_SEEDS = (2026, 2027, 2028)
PANELS = ("geolink", "taranaki", "teapot", "cnfield")
RUNS = {  # method -> (结果根目录, 标签)
    "geosr2": ("results/v6_ensemble", "SuiteSR"),
    "ssm_morph_native": ("results/v6_ensemble", "encoded state-space"),
    "geosr2_obs": ("results/v6_ensemble_obs", "SuiteSR-obs"),
}
ARTIFACT = {("taranaki", "GR", 2), ("taranaki", "DTC", 2), ("taranaki", "RHOB", 2)}
SINGLE = ["lstm", "ssm", "dmcnet", "cascade_sr", "transformer"]
GUIDED = [m + "_multi" for m in SINGLE] + ["lstm_native", "ssm_native", "dmcnet_native"]
ENCODED = ["ssm_morph_native", "dmcnet_morph_native"]
METRICS = ["mae", "rmse", "r2", "pearson", "psnr", "grad_mae", "hf_rmse", "spectral_angle"]
OUT = ROOT / "results/v6_summary"


def data_range_from_rows(rows: pd.DataFrame) -> float:
    """从 window_metrics 行反推运行时登记的固定 PSNR 参考量程。"""
    ok = rows[np.isfinite(rows.psnr) & np.isfinite(rows.rmse)]
    return float(np.median((ok.rmse + 1e-12) * 10 ** (ok.psnr / 20)))


def window_scores(hr, pred, starts, lengths, scale, data_range):
    out = []
    for h, p, s, n in zip(hr, pred, starts, lengths):
        s = int(s); n = int(n)
        if n > 0:
            h = h[s:s + n]; p = p[s:s + n]
        out.append(compute_metrics(h, p, scale=scale, data_range=data_range).__dict__)
    return pd.DataFrame(out)


def load_task_bundles(root: Path, panel: str, ds: int, method: str, curve: str, scale: int):
    bundles = []
    for ms in MODEL_SEEDS:
        p = root / panel / f"ds{ds}_ms{ms}" / "predictions" / f"{panel}_{curve}_{scale}_{method}.npz"
        if not p.exists():
            return None
        bundles.append(np.load(p, allow_pickle=False))
    ref = bundles[0]
    for b in bundles[1:]:
        if not (np.array_equal(b["test_wells"], ref["test_wells"]) and np.array_equal(b["test_depth"], ref["test_depth"])):
            raise RuntimeError(f"test windows differ between model seeds for {panel} {curve} {scale} {method}")
    return bundles


def evaluate() -> pd.DataFrame:
    rows = []
    for method, (root, _) in RUNS.items():
        root = ROOT / root
        for panel in PANELS:
            for ds in DATA_SEEDS:
                wm_path = root / panel / f"ds{ds}_ms{MODEL_SEEDS[0]}" / "window_metrics.csv"
                if not wm_path.exists():
                    continue
                wm = pd.read_csv(wm_path)
                wm = wm[wm.method == method]
                for (curve, scale), sub in wm.groupby(["curve", "scale"]):
                    bundles = load_task_bundles(root, panel, ds, method, curve, int(scale))
                    if bundles is None:
                        continue
                    dr = data_range_from_rows(sub)
                    b0 = bundles[0]
                    hr, st, ln = b0["test_hr"], b0["test_eval_start"], b0["test_eval_length"]
                    preds = [b["test_pred"] for b in bundles]
                    forms = {f"single_ms{ms}": p for ms, p in zip(MODEL_SEEDS, preds)}
                    forms["ensemble3"] = np.mean(np.stack(preds), axis=0)
                    for form, p in forms.items():
                        sc = window_scores(hr, p, st, ln, int(scale), dr).mean(numeric_only=True)
                        rows.append(dict(dataset=panel, curve=curve, scale=int(scale), data_seed=ds, method=method,
                                         form=form, n_windows=len(hr), **{m: float(sc[m]) for m in METRICS}))
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    single = d[d.form.str.startswith("single")].groupby(["dataset", "curve", "scale", "data_seed", "method"])[METRICS].mean().reset_index()
    single["form"] = "single_mean"; single["n_windows"] = np.nan
    return pd.concat([d, single], ignore_index=True)


def comparators(ds: int) -> pd.DataFrame:
    frames = []
    for root in ("results/v2_guided", "results/v5_native", "results/v5_morph"):
        for panel in PANELS:
            p = ROOT / root / panel / f"seed{ds}" / "summary_metrics.csv"
            if p.exists():
                frames.append(pd.read_csv(p))
    c = pd.concat(frames, ignore_index=True)
    return c[c.condition.eq("matched")] if "condition" in c else c


def margins(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selp = ROOT / "results/v2_summary/guide_selection.csv"
    selection = {}
    if selp.exists():
        gs = pd.read_csv(selp)
        selection = {(r.dataset, r.curve, int(r.scale)): r.selected for r in gs.itertuples()}
    for ds in DATA_SEEDS:
        comp = comparators(ds)
        if comp.empty:
            continue
        piv = comp.pivot_table(index=["dataset", "curve", "scale"], columns="method", values="mae")
        sub = d[d.data_seed == ds]
        for (panel, curve, scale, method), g in sub.groupby(["dataset", "curve", "scale", "method"]):
            key = (panel, curve, scale)
            if key not in piv.index:
                continue
            ref = piv.loc[key]
            best = {
                "pchip": ref.get("pchip", np.nan),
                "best_single": ref.reindex(SINGLE).min(),
                "best_guided": ref.reindex(GUIDED).min(),
                "best_encoded": ref.reindex(ENCODED).min(),
                "geosr2_benchmark": ref.get("geosr2", np.nan),
            }
            forms = g.set_index("form").mae
            # 基准中实际部署的 SuiteSR: 由验证集选定引导或仅目标形式
            sel = selection.get(key)
            if sel is not None and sel in ref.index and np.isfinite(ref[sel]):
                best["geosr2_selected"] = ref[sel]
            # 同一 model seed 下无观测掩码的 SuiteSR
            base = sub[(sub.method == "geosr2") & (sub.form == "single_mean")]
            base = base[(base.dataset == panel) & (base.curve == curve) & (base.scale == scale)]
            if not base.empty and method == "geosr2_obs":
                best["geosr2_single_mean"] = float(base.mae.iloc[0])
            # 同样做三模型集成的编码对比方法
            enc = sub[(sub.method == "ssm_morph_native") & (sub.form == "ensemble3")]
            enc = enc[(enc.dataset == panel) & (enc.curve == curve) & (enc.scale == scale)]
            if not enc.empty and method != "ssm_morph_native":
                best["encoded_ensemble"] = float(enc.mae.iloc[0])
            for form in ("single_mean", "ensemble3"):
                if form not in forms:
                    continue
                for rname, rval in best.items():
                    if np.isfinite(rval):
                        rows.append(dict(dataset=panel, curve=curve, scale=scale, data_seed=ds, method=method, form=form,
                                         reference=rname, ref_mae=rval, mae=forms[form],
                                         gain_pct=100 * (rval - forms[form]) / rval,
                                         artifact=key in ARTIFACT))
            if "ensemble3" in forms and "single_mean" in forms:
                rows.append(dict(dataset=panel, curve=curve, scale=scale, data_seed=ds, method=method, form="ensemble3",
                                 reference="single_mean", ref_mae=forms["single_mean"], mae=forms["ensemble3"],
                                 gain_pct=100 * (forms["single_mean"] - forms["ensemble3"]) / forms["single_mean"],
                                 artifact=key in ARTIFACT))
    return pd.DataFrame(rows)


def summarize(m: pd.DataFrame) -> pd.DataFrame:
    m = m[~m.artifact]
    # 先对 data seed 平均每任务增益, 再跨任务汇总
    t = m.groupby(["dataset", "curve", "scale", "method", "form", "reference"]).gain_pct.mean().reset_index()
    rows = []
    for (method, form, ref), g in t.groupby(["method", "form", "reference"]):
        for scale, gg in [("all", g)] + [(s, g[g.scale == s]) for s in (2, 4, 8)]:
            rows.append(dict(method=method, form=form, reference=ref, scale=scale, tasks=len(gg),
                             mean=gg.gain_pct.mean(), median=gg.gain_pct.median(), wins=int((gg.gain_pct > 0).sum()),
                             min=gg.gain_pct.min(), max=gg.gain_pct.max()))
    return pd.DataFrame(rows)


def latex_table(s: pd.DataFrame, path: Path) -> None:
    def cell(method, form, ref, scale):
        r = s[(s.method == method) & (s.form == form) & (s.reference == ref) & (s.scale == scale)]
        if r.empty:
            return "--"
        r = r.iloc[0]
        return f"{r['mean']:+.1f} / {r['median']:+.1f} ({r['wins']}/{r['tasks']})"
    lines = [
        "% Auto-generated by scripts/eval_seed_ensemble.py",
        r"\begin{table*}[!htbp]", r"\centering",
        r"\caption{Deployment configuration: three-model seed ensembles on the guided benchmark. Three models per task differ only in parameter initialisation and batch order (\texttt{model\_seed}) and share the data seed's window subsample; their test predictions are averaged. Cells give the mean / median MAE gain (\%) over the assessable tasks of the scale, with the number of tasks on which the ensemble is better in parentheses; task values are means over the three data seeds. References: the mean of the three single models of the same method (\emph{vs.\ single}), the best single-curve comparator, the best guided comparator (fusion-stem or widened) and the best encoded comparator, all single models of the same data seed.}",
        r"\label{tab:v6-ensemble}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Model & Reference & All & $2\times$ & $4\times$ & $8\times$ \\", r"\midrule",
    ]
    for method, (_, label) in RUNS.items():
        if s[s.method == method].empty:
            continue
        if method == "geosr2_obs":
            lines.append(" & ".join([f"{label} single", "vs.\\ SuiteSR single (same model seeds)"] + [cell(method, "single_mean", "geosr2_single_mean", sc) for sc in ("all", 2, 4, 8)]) + r" \\")
        for ref, rlabel in [("single_mean", "vs.\\ single (same method)"), ("geosr2_selected", "vs.\\ SuiteSR, validation-selected"),
                            ("best_single", "vs.\\ best single-curve comparator"),
                            ("best_guided", "vs.\\ best guided comparator"), ("best_encoded", "vs.\\ best encoded comparator"),
                            ("encoded_ensemble", "vs.\\ encoded state-space ensemble")]:
            if ref == "encoded_ensemble" and method == "ssm_morph_native":
                continue
            lines.append(" & ".join([f"{label} ensemble", rlabel] + [cell(method, "ensemble3", ref, sc) for sc in ("all", 2, 4, 8)]) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}\end{adjustbox}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = evaluate()
    if d.empty:
        print("no complete ensemble groups yet"); return
    d.to_csv(OUT / "ensemble_task_metrics.csv", index=False)
    m = margins(d); m.to_csv(OUT / "ensemble_margins.csv", index=False)
    s = summarize(m); s.to_csv(OUT / "ensemble_summary.csv", index=False)
    latex_table(s, OUT / "ensemble_table.tex")
    done = d[d.form == "ensemble3"].groupby(["method", "data_seed"]).size()
    print("complete ensemble tasks per method/data seed:\n", done)
    key = s[s.reference.isin(["single_mean", "geosr2_selected", "best_encoded", "encoded_ensemble", "best_guided", "best_single"]) & (s.form == "ensemble3")]
    print(key.sort_values(["method", "reference", "scale"]).to_string(index=False))


if __name__ == "__main__":
    main()
