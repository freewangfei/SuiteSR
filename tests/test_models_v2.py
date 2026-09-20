"""SuiteSR 系列及对比方法引导形式的输出形状、有限性和零初始化检查。"""
import torch

from wellsr.models_v2 import GeoSRv2, build_model_v2

NAMES = ["geosr2", "geosr2_long", "geosr2_nomorph", "geosr2_norecur", "geosr2_plainblock", "geosr2_deep",
         "geosr2_nogate", "geosr2_morphtarget", "geosr2_morphguide", "geosr2_dropgrad", "geosr2_ma5",
         "geosr2_gd", "lstm_native", "transformer_native", "dmcnet_native", "cascade_sr_native", "ssm_native",
         "ssm_morph_native", "dmcnet_morph_native", "lstm", "ssm", "dmcnet", "cascade_sr", "transformer"]


def test_every_final_model_builds_and_returns_the_target_shape():
    x = torch.randn(2, 7, 128)
    for name in NAMES:
        model = build_model_v2(name, channels=16, depth=2, in_channels=7)
        y = model(x)
        assert y.shape == (2, 1, 128), name
        assert torch.isfinite(y).all(), name


def test_observation_mask_form_reads_eight_rows():
    model = build_model_v2("geosr2_obs_mask", channels=16, depth=2, in_channels=8)
    y = model(torch.randn(2, 8, 128))
    assert y.shape == (2, 1, 128)


def test_residual_head_starts_at_the_interpolated_target():
    model = GeoSRv2(16, 2, 7)
    x = torch.randn(2, 7, 128)
    assert torch.allclose(model(x), x[:, :1], atol=1e-6)


def test_single_curve_form_matches_parameter_count_ordering():
    guided = sum(p.numel() for p in GeoSRv2(32, 3, 7).parameters())
    single = sum(p.numel() for p in GeoSRv2(32, 3, 1).parameters())
    assert guided > single
