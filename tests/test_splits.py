from pathlib import Path

import pandas as pd
import pytest

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "processed"
DATASETS = ["geolink", "taranaki", "teapot"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_splits_are_well_disjoint(dataset):
    path = DATA_ROOT / dataset / "panel.parquet"
    if not path.exists():
        pytest.skip(f"{path} not present")
    df = pd.read_parquet(path)
    per_split = df.groupby("SPLIT")["WELL"].agg(set)
    splits = list(per_split.index)
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            overlap = per_split[a] & per_split[b]
            assert not overlap, f"{dataset}: wells shared by {a}/{b}: {sorted(overlap)[:5]}"


@pytest.mark.parametrize("dataset", DATASETS)
def test_wells_have_real_identities(dataset):
    path = DATA_ROOT / dataset / "panel.parquet"
    if not path.exists():
        pytest.skip(f"{path} not present")
    df = pd.read_parquet(path)
    anonymous = df["WELL"].astype(str).str.startswith("slice_")
    assert not anonymous.any(), f"{dataset}: {anonymous.sum()} rows still carry slice_* pseudo-wells"


@pytest.mark.parametrize("dataset", DATASETS)
def test_windows_carry_pairing_keys(dataset):
    path = DATA_ROOT / dataset / "panel.parquet"
    if not path.exists():
        pytest.skip(f"{path} not present")
    from wellsr.data import build_windows

    windows = build_windows(
        dataset,
        curves=None,
        scales=[2],
        split="test",
        window=128,
        stride=128,
        data_root=DATA_ROOT,
        max_windows=32,
    )
    assert windows, "no test windows built"
    keys = {(w.curve, w.well, w.depth_start) for w in windows}
    assert len(keys) == len(windows), "window identity keys are not unique"
    assert all(not w.well.startswith("slice_") for w in windows)
