import numpy as np

from wellsr.metrics import compute_metrics, spectral_angle, super_nyquist_component


def test_perfect_prediction():
    y = np.sin(np.linspace(0, 12, 128))
    m = compute_metrics(y, y, scale=4)
    assert m.mae == 0.0
    assert m.rmse == 0.0
    assert abs(m.r2 - 1.0) < 1e-12
    assert abs(m.pearson - 1.0) < 1e-9
    assert m.hf_rmse < 1e-12
    assert m.spectral_angle < 1e-6


def test_degenerate_windows_are_nan_not_zero():
    y = np.full(128, 3.0)
    m = compute_metrics(y, y + 0.1, scale=4)
    assert np.isnan(m.r2)
    assert np.isnan(m.pearson)
    assert np.isnan(m.psnr)


def test_psnr_uses_fixed_range_when_given():
    rng = np.random.default_rng(0)
    y = rng.normal(size=128)
    p = y + 0.1
    m_fixed = compute_metrics(y, p, scale=2, data_range=10.0)
    m_local = compute_metrics(y, p, scale=2)
    expected = 20 * np.log10(10.0 / (m_fixed.rmse + 1e-12))
    assert abs(m_fixed.psnr - expected) < 1e-9
    assert m_fixed.psnr != m_local.psnr


def test_super_nyquist_component_band_split():
    n = 256
    t = np.arange(n)
    low = np.sin(2 * np.pi * 0.02 * t)  # 低于 1/(2*8) = 0.0625
    high = np.sin(2 * np.pi * 0.2 * t)  # 高于 8x 截止频率
    hf = super_nyquist_component(low + high, scale=8)
    # 低频部分应被去除, 高频部分保留
    assert np.sqrt(np.mean((hf - high) ** 2)) < 0.05


def test_hf_rmse_ignores_sub_nyquist_error():
    """完全低于 LR Nyquist 的误差不应计入 hf_rmse。"""
    n = 256
    t = np.arange(n)
    y = np.sin(2 * np.pi * 0.2 * t)
    pred = y + 0.5 * np.sin(2 * np.pi * 0.02 * t)
    m = compute_metrics(y, pred, scale=8)
    assert m.hf_rmse < 0.02
    assert m.mae > 0.2


def test_spectral_angle_excludes_dc():
    y = np.sin(2 * np.pi * 0.1 * np.arange(128))
    assert abs(spectral_angle(y, y + 100.0)) < 1e-6
