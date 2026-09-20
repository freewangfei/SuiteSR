import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from wellsr.data import build_windows, synthetic_panel
from wellsr.degradation import anti_alias_downsample, make_lr_hr_pair, upsample_to_length


@pytest.mark.parametrize("scale", [2, 4, 8])
def test_lr_up_is_aligned_on_ramp(scale):
    """线性斜坡经高斯滤波和线性插值后内部样本应精确保持, 残差即网格偏移。"""
    length = 128
    hr = np.arange(length, dtype=np.float32)
    lr_up, target = make_lr_hr_pair(hr, scale)
    margin = 4 * scale  # 跳过滤波边缘效应和外推尾部
    interior = slice(margin, length - margin)
    err = np.abs(lr_up[interior] - target[interior])
    assert err.max() < 1e-3, f"interior misalignment {err.max():.4f} at scale {scale}"


@pytest.mark.parametrize("scale", [2, 4, 8])
def test_lr_up_preserves_low_frequency_sine(scale):
    """远低于抽稀 Nyquist 的正弦应以很小的相位误差恢复。"""
    length = 256
    freq = 0.5 / scale / 8.0  # 远低于 LR Nyquist
    hr = np.sin(2 * np.pi * freq * np.arange(length)).astype(np.float32)
    lr_up, target = make_lr_hr_pair(hr, scale)
    interior = slice(4 * scale, length - 4 * scale)
    assert np.abs(lr_up[interior] - target[interior]).max() < 0.05


def test_upsample_positions_match_decimation():
    length, scale = 64, 4
    hr = np.random.default_rng(0).normal(size=length).astype(np.float32)
    lr = anti_alias_downsample(hr, scale)
    up = upsample_to_length(lr, length, kind="linear", scale=scale)
    # 保留样本位于原索引处
    np.testing.assert_allclose(up[::scale], lr, rtol=0, atol=1e-6)


def test_scale_inference_matches_explicit():
    hr = np.random.default_rng(1).normal(size=128).astype(np.float32)
    lr = anti_alias_downsample(hr, 8)
    np.testing.assert_allclose(
        upsample_to_length(lr, 128, scale=8),
        upsample_to_length(lr, 128),
        rtol=0,
        atol=0,
    )


def test_continuous_response_is_filtered_before_windowing(monkeypatch):
    """continuous 作用域不应在每个窗口重新启动反射滤波。"""
    panel = synthetic_panel(n_wells=1, length=256, seed=2026)
    monkeypatch.setattr("wellsr.data.load_panel", lambda _dataset, _root="data/processed": panel)

    windowed = build_windows(
        "toy", ["GR"], [4], "test", 128, 128, response_scope="window"
    )[0]
    continuous = build_windows(
        "toy", ["GR"], [4], "test", 128, 128, response_scope="continuous"
    )[0]
    signal = panel.loc[panel.WELL == "SYN_000", "GR"].to_numpy(np.float32)
    filtered = gaussian_filter1d(signal, sigma=2.0, mode="reflect")
    expected = upsample_to_length(filtered[:128:4], 128, scale=4)

    np.testing.assert_allclose(continuous.lr, expected, rtol=0, atol=1e-6)
    assert np.max(np.abs(windowed.lr - continuous.lr)) > 1e-4
