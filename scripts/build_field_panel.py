"""从 FORWARD_TEXT_FORMAT 井文件构建 CN-Field 陆上面板。
输入为十口 0.125 m 采样的陆上井 (GR, DT->DTC, RHOB, NPHI), 只保留测量曲线, 丢弃解释成果 (POR, SW, SH, PERM 等)。
井按名称排序后以固定 3:1:1 轮转分配 train/val/test, 归一化常数仅由训练井估计, 输出 panel.parquet。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# 物理合理的采集范围; 超出范围的样本按缺失处理而非裁剪, 不作为重建目标。
PHYSICAL_RANGE = {
    "GR": (1.0, 400.0),      # API
    "DTC": (130.0, 700.0),   # us/m
    "RHOB": (1.50, 3.10),    # g/cm3
    "NPHI": (0.0, 1.00),     # v/v
}

SOURCE_TO_PANEL = {"GR": "GR", "DT": "DTC", "RHOB": "RHOB", "NPHI": "NPHI"}
CURVES = ["GR", "DTC", "RHOB", "NPHI"]


def parse_forward_text(path: Path) -> tuple[pd.DataFrame, dict]:
    """解析一个 FORWARD_TEXT_FORMAT_1.0 文件, 返回按深度索引的 DataFrame 和头部字典。"""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header: dict[str, str] = {}
    names: list[str] | None = None
    body_start: int | None = None
    for i, line in enumerate(text):
        s = line.strip()
        if s.startswith("CURVENAME"):
            names = [x.strip() for x in s.split("=", 1)[1].split(",")]
        elif s == "END":
            body_start = i + 1
            break
        elif "=" in s:
            k, v = s.split("=", 1)
            header[k.strip()] = v.strip()
    if names is None or body_start is None:
        raise ValueError(f"{path}: malformed header")

    columns = ["DEPTH"] + names
    rows = []
    for line in text[body_start:]:
        parts = line.split()
        if len(parts) < len(columns):
            continue
        try:
            rows.append([float(x) for x in parts[: len(columns)]])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"{path}: no numeric rows")
    return pd.DataFrame(rows, columns=columns), header


def extract_curves(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"DEPTH": df["DEPTH"].to_numpy(dtype=np.float64)})
    for src, dst in SOURCE_TO_PANEL.items():
        if src in df.columns:
            v = df[src].to_numpy(dtype=np.float64).copy()
            lo, hi = PHYSICAL_RANGE[dst]
            v[~np.isfinite(v)] = np.nan
            v[(v < lo) | (v > hi)] = np.nan
            out[dst] = v
        else:
            out[dst] = np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/opt/code/cejinquxian/data/000")
    ap.add_argument("--out", default="data/processed/cnfield")
    args = ap.parse_args()

    files = sorted(Path(args.source).glob("*.txt"))
    if not files:
        raise SystemExit(f"no source files under {args.source}")

    wells = []
    for idx, path in enumerate(files):
        raw, header = parse_forward_text(path)
        frame = extract_curves(raw)
        well_id = f"CN-W{idx + 1:02d}"
        split = {3: "val", 4: "test"}.get(idx % 5, "train")
        frame.insert(0, "WELL", well_id)
        frame["SLICE_ID"] = idx
        frame["SPLIT"] = split
        wells.append(
            {
                "well": well_id,
                "source_name": path.stem,
                "split": split,
                "rows": int(len(frame)),
                "step_m": float(header.get("RLEV", "nan")),
                "depth_start_m": float(frame["DEPTH"].iloc[0]),
                "depth_end_m": float(frame["DEPTH"].iloc[-1]),
                "frame": frame,
            }
        )

    panel = pd.concat([w["frame"] for w in wells], ignore_index=True)

    # 归一化常数仅来自训练井
    train = panel[panel["SPLIT"] == "train"]
    norm = {}
    for curve in CURVES:
        v = train[curve].to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        norm[curve] = {"mean": float(v.mean()), "std": float(v.std() + 1e-6), "n_train": int(v.size)}
        panel[curve] = (panel[curve] - norm[curve]["mean"]) / norm[curve]["std"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    panel = panel[["WELL", "DEPTH", *CURVES, "SLICE_ID", "SPLIT"]]
    panel.to_parquet(out / "panel.parquet", index=False)
    pd.DataFrame({"curve": CURVES}).to_csv(out / "curves.csv", index=False)

    report = {
        "source": str(Path(args.source).resolve()),
        "sampling_interval_m": 0.125,
        "curves": CURVES,
        "dropped_columns_note": "interpretation products (POR/SW/SH/PERM/...) excluded; measured logs only",
        "physical_range_filter": {k: list(v) for k, v in PHYSICAL_RANGE.items()},
        "split_rule": "wells sorted by file name, 3:1:1 round robin over train/val/test",
        "normalisation": norm,
        "wells": [{k: v for k, v in w.items() if k != "frame"} for w in wells],
        "rows_total": int(len(panel)),
        "rows_by_split": panel["SPLIT"].value_counts().to_dict(),
        "finite_fraction": {c: float(np.isfinite(panel[c]).mean()) for c in CURVES},
    }
    (out / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "wells"}, indent=2, ensure_ascii=False))
    print("\nwells:")
    for w in report["wells"]:
        print(f"  {w['well']}  {w['source_name']:<10s} {w['split']:<5s} rows={w['rows']:6d} "
              f"{w['depth_start_m']:.1f}-{w['depth_end_m']:.1f} m")


if __name__ == "__main__":
    main()
