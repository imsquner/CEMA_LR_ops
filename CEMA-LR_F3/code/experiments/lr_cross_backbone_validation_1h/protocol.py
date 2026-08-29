from __future__ import annotations

import itertools
import random


PROTOCOL = {
    "resample": {"rule": "1h", "closed": "right", "label": "right"},
    "ema": {"span": 9, "alpha": 0.2, "adjust": False, "warmup_hours": 27},
    "features": [
        "ema_voltage_v", "current_a", "h2_out_flow_l_min",
        "air_out_pressure_mbar", "coolant_in_temp_c",
    ],
    "lookback": 12,
    "horizon": 1,
    "dev_fraction": 0.8,
    "folds": [[0.0, 0.5, 0.5, 0.6], [0.0, 0.6, 0.6, 0.7], [0.0, 0.7, 0.7, 0.8]],
    "optimizer": "AdamW",
    "loss": "MSELoss",
    "max_epochs": 120,
    "validate_every": 2,
    "patience_checks": 15,
    "gradient_clip_norm": 1.0,
    "amp": False,
    "precision": "FP32",
    "scheduler": None,
    "optuna": False,
    "pruner": False,
    "candidate_pool_seed": 42,
    "candidate_training_seed": 42,
    "formal_seeds": [42, 2024, 2026],
    "max_cuda_workers": 2,
    "transformer_activation": "gelu",
    "transformer_norm_first": True,
    "attention_mask": None,
    "pair_objective": "(direct_mean_rmse + lr_mean_rmse) / 2",
}


def _records(prefix: str, names: tuple[str, ...], values: list[tuple], count: int) -> list[dict]:
    return [
        {"candidate_id": f"{prefix}-C{index:02d}", **dict(zip(names, item))}
        for index, item in enumerate(values[:count], 1)
    ]


def generate_candidate_pools(seed: int = 42, backup_count: int = 10) -> dict:
    rng = random.Random(seed)
    common = ([1, 2], [0.0, 0.1, 0.2], [3e-4, 5e-4, 1e-3, 2e-3], [0.0, 1e-5, 1e-4], [32, 64])
    lstm = list(itertools.product([32, 64, 96], *common))
    bigru = list(itertools.product([32, 64, 96], *common))
    transformer = [
        item for item in itertools.product(
            [32, 64, 96], [2, 4], [1, 2], [64, 128, 192], [0.0, 0.1, 0.2],
            [3e-4, 5e-4, 1e-3, 2e-3], [0.0, 1e-5, 1e-4], [32, 64],
        )
        if item[0] % item[1] == 0 and item[3] >= item[0]
    ]
    for values in (lstm, bigru, transformer):
        rng.shuffle(values)
    total = 5 + backup_count
    specs = {
        "lstm": _records("LSTM", ("hidden_size", "num_layers", "head_dropout", "learning_rate", "weight_decay", "batch_size"), lstm, total),
        "bigru": _records("BIGRU", ("total_hidden", "num_layers", "head_dropout", "learning_rate", "weight_decay", "batch_size"), bigru, total),
        "transformer": _records("TRANSFORMER", ("d_model", "nhead", "num_layers", "dim_feedforward", "dropout", "learning_rate", "weight_decay", "batch_size"), transformer, total),
    }
    return {
        "seed": seed,
        **{name: {"primary": rows[:5], "backups": rows[5:]} for name, rows in specs.items()},
    }


def candidate_parameters(config: dict) -> dict:
    return {key: value for key, value in config.items() if key != "candidate_id"}

