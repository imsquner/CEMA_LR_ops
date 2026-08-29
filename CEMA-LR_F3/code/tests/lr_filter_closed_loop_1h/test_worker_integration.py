import numpy as np

from experiments.lr_filter_closed_loop_1h.data import FilterWindows, save_windows
from experiments.lr_filter_closed_loop_1h.worker import execute_fit


def test_cpu_worker_completes_causal_fold_fit(tmp_path):
    rng = np.random.default_rng(42)
    count = 80
    features = rng.normal(size=(count, 12, 5)).astype(np.float32)
    anchors = (0.6 + 0.01 * rng.normal(size=(count, 1))).astype(np.float32)
    targets = anchors + (0.001 * rng.normal(size=(count, 1))).astype(np.float32)
    windows = FilterWindows(features, targets, anchors, targets[:, 0], np.arange(count), np.arange(count), np.arange(count) + 1, np.arange(count))
    cache = tmp_path / "cache.npz"
    save_windows(cache, windows, {"data_sha256": "synthetic"})
    spec = {
        "task_id": "integration__F1__FC1__gru__lr__GRU-C01__fold1",
        "stage": "integration", "filter_id": "F1", "dataset": "FC1", "backbone": "gru",
        "target_mode": "lr", "replicate": "fold1", "config_id": "GRU-C01",
        "config": {"candidate_id": "GRU-C01", "hidden_size": 8, "num_layers": 1, "head_dropout": 0.0, "learning_rate": 0.001, "weight_decay": 0.0, "batch_size": 16},
        "windows_path": str(cache), "device": "cpu", "max_epochs": 2, "patience_checks": 1,
    }
    result = execute_fit(spec, tmp_path)
    assert result["best_epoch"] == 2
    assert result["rmse"] >= 0
    assert result["data_sha256"] == "synthetic"

