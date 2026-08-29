import numpy as np
import torch

from experiments.lr_filter_closed_loop_1h.data import FilterWindows
from experiments.lr_filter_closed_loop_1h.metrics import compute_metrics
from experiments.lr_filter_closed_loop_1h.training import fit_scalers, reconstruct_physical


def _windows():
    features = np.arange(20 * 12 * 5, dtype=np.float32).reshape(20, 12, 5)
    targets = np.arange(20, dtype=np.float32).reshape(-1, 1) + 100
    anchors = targets - 2
    return FilterWindows(features, targets, anchors, targets[:, 0], np.arange(20), np.arange(20), np.arange(20) + 1, np.arange(20))


def test_scalers_fit_only_requested_training_prefix():
    windows = _windows()
    scaler = fit_scalers(windows, np.arange(8), "direct")
    altered = _windows()
    altered.features[8:] += 1_000_000
    altered.targets[8:] += 1_000_000
    other = fit_scalers(altered, np.arange(8), "direct")
    np.testing.assert_array_equal(scaler.input_mean, other.input_mean)
    assert scaler.target_mean == other.target_mean


def test_lr_target_has_independent_scaler_and_uses_anchor_for_reconstruction():
    windows = _windows()
    direct = fit_scalers(windows, np.arange(8), "direct")
    lr = fit_scalers(windows, np.arange(8), "lr")
    assert direct.target_mean != lr.target_mean
    prediction = reconstruct_physical(np.array([[0.0]]), lr.target_mean, lr.target_scale, np.array([[98.0]]), "lr")
    assert prediction.item() == 100.0


def test_metrics_include_tail_dynamic_and_fixed_windows():
    truth = np.linspace(0, 1, 120)
    prediction = truth + 0.1
    result = compute_metrics(truth, prediction)
    for key in ("mse", "rmse", "mae", "medae", "r2", "pearson", "std_ratio", "p95_abs_error", "cvar95", "top_5pct_sse_share", "worst_24h_rmse", "worst_72h_rmse", "best_lag", "lag_adjusted_pearson", "residual_pearson"):
        assert key in result
    assert np.isfinite(result["rmse"])

