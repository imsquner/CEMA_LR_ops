from __future__ import annotations

import numpy as np
import pandas as pd


def _failure_repaired(rows: pd.DataFrame) -> bool:
    fc1_gru = rows[(rows.dataset == "FC1") & (rows.backbone.str.upper() == "GRU")]
    fc2_tcn = rows[(rows.dataset == "FC2") & (rows.backbone.str.upper() == "TCN")]
    return bool((not fc1_gru.empty and fc1_gru.rmse_improvement_pct.iloc[0] > 0) or (not fc2_tcn.empty and fc2_tcn.rmse_improvement_pct.iloc[0] > 0))


def evaluate_expansion_trigger(rows: pd.DataFrame) -> dict:
    rmse = rows.rmse_improvement_pct.to_numpy(dtype=float)
    mae = rows.mae_improvement_pct.to_numpy(dtype=float)
    checks = {
        "original_failure_repaired": _failure_repaired(rows),
        "at_least_3_of_4_rmse_improved": int(np.sum(rmse > 0)) >= 3,
        "mean_rmse_improvement_gt_3pct": float(np.mean(rmse)) > 3.0,
        "no_rmse_worse_than_5pct": bool(np.all(rmse >= -5.0)),
        "at_least_3_of_4_mae_improved": int(np.sum(mae > 0)) >= 3,
        "cvar95_not_worse_than_10pct": bool(np.all(rows.cvar95_ratio.to_numpy(dtype=float) <= 1.10)),
        "worst_24h_not_worse_than_10pct": bool(np.all(rows.worst_24h_ratio.to_numpy(dtype=float) <= 1.10)),
    }
    return {"FILTER_EXPANSION_TRIGGER": all(checks.values()), "checks": checks}


def evaluate_global_success(rows: pd.DataFrame) -> dict:
    rmse = rows.rmse_improvement_pct.to_numpy(dtype=float)
    mae = rows.mae_improvement_pct.to_numpy(dtype=float)
    std_error = np.abs(rows.residual_std_ratio.to_numpy(dtype=float) - 1.0)
    baseline_std_error = np.abs(rows.baseline_residual_std_ratio.to_numpy(dtype=float) - 1.0)
    dynamic_better = bool(
        np.mean(std_error) < np.mean(baseline_std_error)
        or rows.residual_pearson.mean() > rows.baseline_residual_pearson.mean()
    )
    checks = {
        "at_least_8_of_10_rmse_improved": int(np.sum(rmse > 0)) >= 8,
        "original_failure_repaired": _failure_repaired(rows),
        "mean_rmse_improvement_ge_5pct": float(np.mean(rmse)) >= 5.0,
        "no_rmse_worse_than_5pct": bool(np.all(rmse >= -5.0)),
        "at_least_7_of_10_mae_improved": int(np.sum(mae > 0)) >= 7,
        "at_least_7_of_10_cvar95_safe": int(np.sum(rows.cvar95_ratio.to_numpy(dtype=float) <= 1.05)) >= 7,
        "dynamic_response_improved": dynamic_better,
    }
    return {"FILTER_DEV_GLOBAL_SUCCESS": all(checks.values()), "checks": checks}


def should_run_f7(success_by_filter: dict[str, bool]) -> bool:
    return not any(bool(success_by_filter.get(f"F{index}", False)) for index in range(1, 7))


def evaluate_minimum_safe(rows: pd.DataFrame) -> dict:
    rmse = rows.rmse_improvement_pct.to_numpy(dtype=float)
    checks = {
        "at_least_6_of_10_rmse_improved": int(np.sum(rmse > 0)) >= 6,
        "mean_rmse_improvement_gt_0": float(np.mean(rmse)) > 0.0,
        "original_failure_repaired": _failure_repaired(rows),
        "no_rmse_worse_than_10pct": bool(np.all(rmse >= -10.0)),
    }
    return {"FILTER_MINIMUM_SAFE": all(checks.values()), "checks": checks}


def rank_filters(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.sort_values(
        [
            "rmse_improved_pairs",
            "mean_relative_rmse",
            "failure_pair_mean_improvement_pct",
            "mean_cvar95_ratio",
            "mean_worst_24h_ratio",
            "mean_abs_residual_std_error",
            "mean_residual_pearson",
            "compute_seconds",
        ],
        ascending=[False, True, False, True, True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def classify_final_status(global_success: bool, repaired_count: int, improved_pairs: int, mean_improvement_pct: float) -> str:
    if global_success:
        return "FILTER_GLOBAL_SUCCESS"
    if repaired_count >= 1 or (improved_pairs >= 6 and mean_improvement_pct > 0):
        return "FILTER_PARTIAL_SUCCESS"
    return "FILTER_NO_SUCCESS"
