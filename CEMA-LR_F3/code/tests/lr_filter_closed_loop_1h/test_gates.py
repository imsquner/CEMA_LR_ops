from __future__ import annotations

import pandas as pd

from experiments.lr_filter_closed_loop_1h.gates import (
    classify_final_status,
    evaluate_expansion_trigger,
    evaluate_global_success,
    rank_filters,
    should_run_f7,
)


def screening_rows(rmse=(4.0, 5.0, 6.0, -2.0), mae=(1.0, 2.0, 3.0, -1.0)) -> pd.DataFrame:
    return pd.DataFrame({
        "dataset": ["FC1", "FC1", "FC2", "FC2"],
        "backbone": ["GRU", "TCN", "GRU", "TCN"],
        "rmse_improvement_pct": rmse,
        "mae_improvement_pct": mae,
        "cvar95_ratio": [1.0, 1.02, 0.98, 1.09],
        "worst_24h_ratio": [1.0, 1.01, 0.99, 1.08],
    })


def test_expansion_trigger_requires_original_failure_repair_and_all_safety_gates():
    passed = evaluate_expansion_trigger(screening_rows())
    assert passed["FILTER_EXPANSION_TRIGGER"] is True
    failed = evaluate_expansion_trigger(screening_rows(rmse=(-1.0, 8.0, 8.0, -1.0)))
    assert failed["FILTER_EXPANSION_TRIGGER"] is False
    assert failed["checks"]["original_failure_repaired"] is False


def test_f7_runs_only_when_no_f1_to_f6_filter_has_global_success():
    assert should_run_f7({f"F{i}": False for i in range(1, 7)}) is True
    statuses = {f"F{i}": False for i in range(1, 7)}
    statuses["F4"] = True
    assert should_run_f7(statuses) is False


def test_global_success_enforces_ten_pair_safety_and_dynamic_improvement():
    rows = pd.DataFrame({
        "dataset": ["FC1"] * 5 + ["FC2"] * 5,
        "backbone": ["GRU", "TCN", "LSTM", "BiGRU", "Transformer"] * 2,
        "rmse_improvement_pct": [6, 7, 8, 9, -2, 6, 7, 8, 9, 5],
        "mae_improvement_pct": [1] * 8 + [-1, -1],
        "cvar95_ratio": [1.0] * 8 + [1.1, 1.1],
        "residual_std_ratio": [0.9] * 10,
        "residual_pearson": [0.7] * 10,
        "baseline_residual_std_ratio": [0.5] * 10,
        "baseline_residual_pearson": [0.6] * 10,
    })
    result = evaluate_global_success(rows)
    assert result["FILTER_DEV_GLOBAL_SUCCESS"] is True
    rows.loc[0, "rmse_improvement_pct"] = -6
    assert evaluate_global_success(rows)["FILTER_DEV_GLOBAL_SUCCESS"] is False


def test_filter_ranking_uses_frozen_hierarchy_and_final_status_is_bounded():
    candidates = pd.DataFrame([
        {"filter_id": "F1", "rmse_improved_pairs": 8, "mean_relative_rmse": 0.90, "failure_pair_mean_improvement_pct": 4, "mean_cvar95_ratio": 1.0, "mean_worst_24h_ratio": 1.0, "mean_abs_residual_std_error": 0.2, "mean_residual_pearson": 0.7, "compute_seconds": 20},
        {"filter_id": "F2", "rmse_improved_pairs": 8, "mean_relative_rmse": 0.85, "failure_pair_mean_improvement_pct": 3, "mean_cvar95_ratio": 1.0, "mean_worst_24h_ratio": 1.0, "mean_abs_residual_std_error": 0.2, "mean_residual_pearson": 0.7, "compute_seconds": 10},
    ])
    assert list(rank_filters(candidates).filter_id) == ["F2", "F1"]
    assert classify_final_status(global_success=True, repaired_count=1, improved_pairs=8, mean_improvement_pct=5) == "FILTER_GLOBAL_SUCCESS"
    assert classify_final_status(global_success=False, repaired_count=1, improved_pairs=5, mean_improvement_pct=-1) == "FILTER_PARTIAL_SUCCESS"
    assert classify_final_status(global_success=False, repaired_count=0, improved_pairs=5, mean_improvement_pct=2) == "FILTER_NO_SUCCESS"

