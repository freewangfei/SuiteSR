"""用真实井名重建公开数据面板。
从 Gama 等基准 fold-0 切片数组及其元数据 JSON 读入, 基准 val 切片作为 test, 在井级别从基准 train 中划出 val (15% 井, 固定 seed),
并把真实井名写入 WELL、真实起始采样点写入 DEPTH, 输出 panel.parquet。

用法:
    PYTHONPATH=src python scripts/build_public_panels.py \
        --benchmark-root /path/to/imputation-processed-datasets \
        --out-root data/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CURVES = {
    "geolink": ["GR", "DTC", "RHOB", "NPHI"],
    "taranaki": ["GR", "DTC", "RHOB"],
    "teapot": ["GR", "NPHI", "RHOB"],
}
VAL_WELL_FRACTION = 0.15


def find_fold0_files(root: Path, dataset: str) -> dict[str, tuple[Path, Path]]:
    out = {}
    for split in ("train", "val"):
        npy = next(root.rglob(f"{dataset}_fold_0_well_log_sliced_{split}.npy"), None)
        meta = next(root.rglob(f"{dataset}_fold_0_well_log_slices_meta_{split}.json"), None)
        if npy is None or meta is None:
            raise FileNotFoundError(f"missing fold-0 {split} files for {dataset} under {root}")
        out[split] = (npy, meta)
    return out


def load_split(npy_path: Path, meta_path: Path, curves: list[str], split_label: str) -> pd.DataFrame:
    values = np.load(npy_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if values.ndim != 3 or values.shape[2] != len(curves):
        raise ValueError(f"{npy_path}: shape {values.shape} does not match curves {curves}")
    if len(meta) != values.shape[0]:
        raise ValueError(f"{meta_path}: {len(meta)} records for {values.shape[0]} slices")
    frames = []
    for slice_idx, record in enumerate(meta):
        well, start = str(record[0]), int(record[1])
        frame = pd.DataFrame(values[slice_idx].astype(np.float32), columns=curves)
        frame.insert(0, "DEPTH", np.arange(values.shape[1], dtype=np.float64) + float(start))
        frame.insert(0, "WELL", well)
        frame["SLICE_ID"] = slice_idx
        frame["SPLIT"] = split_label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_panel(dataset: str, benchmark_root: Path, seed: int) -> pd.DataFrame:
    curves = CURVES[dataset]
    files = find_fold0_files(benchmark_root, dataset)
    train = load_split(*files["train"], curves=curves, split_label="train")
    test = load_split(*files["val"], curves=curves, split_label="test")
    train_wells = set(train["WELL"])
    test_wells = set(test["WELL"])
    overlap = train_wells & test_wells
    if overlap:
        raise AssertionError(f"{dataset}: train/test share wells: {sorted(overlap)[:5]}")
    ordered = np.array(sorted(train_wells))
    _, val_wells = train_test_split(ordered, test_size=VAL_WELL_FRACTION, random_state=seed)
    train.loc[train["WELL"].isin(set(val_wells)), "SPLIT"] = "val"
    panel = pd.concat([train, test], ignore_index=True)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-root",
        required=True,
        help="包含 Gama 等 (2025) 基准 fold-0 切片数组及 *_well_log_slices_meta_*.json 元数据的目录 (递归搜索)。",
    )
    parser.add_argument("--out-root", default="data/processed")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--datasets", nargs="*", default=list(CURVES))
    args = parser.parse_args()
    benchmark_root = Path(args.benchmark_root)
    out_root = Path(args.out_root)
    for dataset in args.datasets:
        panel = build_panel(dataset, benchmark_root, args.seed)
        out_dir = out_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "panel.parquet"
        if target.exists():
            legacy = out_dir / "panel_legacy_sliceids.parquet"
            if not legacy.exists():
                target.rename(legacy)
        panel.to_parquet(target, index=False)
        counts = panel.groupby("SPLIT")["WELL"].nunique().to_dict()
        rows = panel.groupby("SPLIT").size().to_dict()
        print(f"{dataset}: wells per split {counts}; rows per split {rows}")
        print(f"{dataset}: rows {len(panel)}, wells {panel['WELL'].nunique()} -> {target}")


if __name__ == "__main__":
    main()
