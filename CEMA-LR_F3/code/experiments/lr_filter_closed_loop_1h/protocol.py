from __future__ import annotations


FILTERS = {
    "F0": {"input": "ema9", "target": "ema9", "anchor": "ema9"},
    "F1": {"input": "ema5", "target": "ema9", "anchor": "ema9"},
    "F2": {"input": "ema5", "target": "ema5", "anchor": "ema5"},
    "F3": {"input": "dema5", "target": "ema9", "anchor": "ema9"},
    "F4": {"input": "ema3", "target": "ema9", "anchor": "ema9"},
    "F5": {"input": "ema7", "target": "ema9", "anchor": "ema9"},
    "F6": {"input": "raw_1h", "target": "ema9", "anchor": "ema9"},
    "F7": {"input": "sma3", "target": "ema9", "anchor": "ema9"},
}


PROTOCOL = {
    "loop_id": "lr_filter_closed_loop_1h",
    "protocol_id": "causal-filter-closed-loop-1h-v1",
    "resample": {"rule": "1h", "closed": "right", "label": "right"},
    "lookback": 12,
    "horizon": 1,
    "development_fraction": 0.8,
    "folds": [[0.0, 0.5, 0.5, 0.6], [0.0, 0.6, 0.6, 0.7], [0.0, 0.7, 0.7, 0.8]],
    "warmup_hours": 27,
    "optimizer": "AdamW",
    "loss": "MSELoss",
    "max_epochs": 120,
    "validate_every": 2,
    "patience_checks": 15,
    "gradient_clip_norm": 1.0,
    "precision": "FP32",
    "amp": False,
    "scheduler": None,
    "candidate_seed": 42,
    "formal_seeds": [42, 2024, 2026, 3407, 7777],
    "minimum_workers_per_gpu": 1,
    "initial_workers_per_gpu": 2,
    "maximum_workers_per_gpu": 4,
}

FORMAL_SEEDS = tuple(PROTOCOL["formal_seeds"])
