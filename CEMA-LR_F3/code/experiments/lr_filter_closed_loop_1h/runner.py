from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import build_filter_windows, load_hourly, prepare_filter_frames, save_windows
from .dispatch import run_specs
from .gates import evaluate_expansion_trigger, evaluate_global_success, evaluate_minimum_safe, rank_filters, should_run_f7
from .protocol import FILTERS, PROTOCOL
from .scheduler import Calibration
from .search_space import BACKBONES, candidate_pools
from .state import atomic_json
from .tasks import DATASETS, FOLDS, TARGET_MODES, FitTask


PACKAGE = Path(__file__).resolve().parent


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(root: Path, arguments: list[str]):
    try:
        return subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}", *arguments], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _fixed_configs() -> dict:
    return json.loads((PACKAGE / "legacy_fixed_configs.json").read_text(encoding="utf-8"))


def prepare(root: Path, output: Path) -> dict:
    root, output = Path(root), Path(output)
    for directory in ("configs", "data_cache", "tasks", "task_artifacts", "logs", "analysis"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    data_rows = []
    for dataset in DATASETS:
        hourly, source_audit = load_hourly(root, dataset)
        frames = prepare_filter_frames(hourly)
        timestamp_hashes = set()
        for filter_id, frame in frames.items():
            windows, window_audit = build_filter_windows(frame)
            timestamp_hash = __import__("hashlib").sha256(windows.timestamps.tobytes()).hexdigest()
            timestamp_hashes.add(timestamp_hash)
            audit = source_audit | window_audit | {
                "filter_id": filter_id, "timestamp_sha256": timestamp_hash,
                "max_input_le_origin": bool(np.all(windows.max_input_positions <= windows.origin_positions)),
                "target_is_origin_plus_horizon": bool(np.all(windows.target_positions == windows.origin_positions + PROTOCOL["horizon"])),
                "data_sha256": source_audit["sha256"],
            }
            save_windows(output / "data_cache" / f"{dataset}_{filter_id}.npz", windows, audit)
            data_rows.append(audit)
        if len(timestamp_hashes) != 1:
            raise RuntimeError(f"common support failed for {dataset}")
    pd.DataFrame(data_rows).to_csv(output / "data_audit.csv", index=False)
    atomic_json(output / "configs" / "protocol.json", PROTOCOL | {"filters": FILTERS})
    atomic_json(output / "configs" / "legacy_fixed_configs.json", _fixed_configs())
    atomic_json(output / "configs" / "candidate_pools.json", candidate_pools())
    manifest = {
        "loop_id": PROTOCOL["loop_id"], "created_at": _utc(), "project_root": str(root),
        "git_commit": _git(root, ["rev-parse", "HEAD"]), "git_dirty": bool(_git(root, ["status", "--porcelain"])),
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "historical_results_read_only": True, "test_access_policy": "locked until formal",
    }
    atomic_json(output / "run_manifest.json", manifest)
    atomic_json(output / "phase_status.json", {"phase": "prepared", "time": _utc()})
    return {"status": "prepared", "cached_windows": len(data_rows), "output": str(output)}


def _task_spec(root: Path, output: Path, task: FitTask, config: dict, **extra) -> dict:
    return task.to_dict() | {
        "project_root": str(root), "windows_path": str(output / "data_cache" / f"{task.dataset}_{task.filter_id}.npz"),
        "config": config, "seed": int(extra.pop("seed", 42)), "device": extra.pop("device", "cuda:0"), **extra,
    }


def _results(output: Path, *, stage=None, filter_id=None) -> pd.DataFrame:
    rows = []
    for path in (output / "tasks" / "done").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))["result"]
        if (stage is None or row["stage"] == stage) and (filter_id is None or row["filter_id"] == filter_id):
            rows.append(row)
    return pd.DataFrame(rows)


def _pair_rows(fold_rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for keys, group in fold_rows.groupby(["dataset", "backbone", "config_id"], sort=False):
        direct = group[group.target_mode == "direct"]
        lr = group[group.target_mode == "lr"]
        if len(direct) != 3 or len(lr) != 3:
            continue
        row = {"dataset": keys[0], "backbone": keys[1], "config_id": keys[2]}
        for metric in ("rmse", "mae", "cvar95", "worst_24h_rmse", "residual_std_ratio", "residual_pearson"):
            row[f"direct_{metric}"] = float(direct[metric].mean())
            row[f"lr_{metric}"] = float(lr[metric].mean())
        row.update({
            "j_pair": (row["direct_rmse"] + row["lr_rmse"]) / 2.0,
            "rmse_improvement_pct": 100 * (row["direct_rmse"] - row["lr_rmse"]) / row["direct_rmse"],
            "mae_improvement_pct": 100 * (row["direct_mae"] - row["lr_mae"]) / row["direct_mae"],
            "cvar95_ratio": row["lr_cvar95"] / row["direct_cvar95"],
            "worst_24h_ratio": row["lr_worst_24h_rmse"] / row["direct_worst_24h_rmse"],
            "residual_std_ratio": row["lr_residual_std_ratio"],
            "baseline_residual_std_ratio": row["direct_residual_std_ratio"],
            "residual_pearson": row["lr_residual_pearson"],
            "baseline_residual_pearson": row["direct_residual_pearson"],
        })
        output.append(row)
    return pd.DataFrame(output)


def _calibration(output: Path) -> Calibration:
    path = output / "scheduler_calibration.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    return Calibration(**row["calibration"])


def smoke(root: Path, output: Path) -> dict:
    if not (output / "data_cache" / "FC1_F1.npz").exists():
        prepare(root, output)
    pools = candidate_pools()
    chosen = [("tcn", max(pools["tcn"]["primary"], key=lambda x: x["batch_size"])), ("transformer", max(pools["transformer"]["primary"], key=lambda x: (x["d_model"], x["num_layers"], x["batch_size"])))]
    before = _gpu_memory()
    specs = []
    for backbone, config in chosen:
        task = FitTask("calibration", "F1", "FC1", backbone, "direct", "fold1", config["candidate_id"])
        specs.append(_task_spec(root, output, task, config, max_epochs=2, patience_checks=1))
    conservative = Calibration(4096, 4096, 2048, 120)
    run_specs(specs, output, conservative, force_serial=True)
    rows = _results(output, stage="calibration")
    after = _gpu_memory()
    calibration = Calibration(
        nvidia_delta_mib=max(1.0, before - after),
        torch_reserved_mib=max(1.0, float(rows.peak_reserved_mib.max())),
        worker_rss_mib=2048.0,
        median_seconds=float(rows.training_seconds.median()),
    )
    payload = {"time": _utc(), "calibration": calibration.__dict__, "tasks": rows.to_dict("records")}
    atomic_json(output / "scheduler_calibration.json", payload)
    atomic_json(output / "phase_status.json", {"phase": "smoke_complete", "time": _utc()})
    return payload


def _gpu_memory() -> float:
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    return float(result.strip().splitlines()[0])


def _fixed_stage_specs(root, output, filter_id, backbones, stage):
    fixed = _fixed_configs()
    return [
        _task_spec(root, output, FitTask(stage, filter_id, dataset, backbone, target, f"fold{fold}", fixed[dataset][backbone]["config"]["candidate_id"]), fixed[dataset][backbone]["config"])
        for dataset in DATASETS for backbone in backbones for target in TARGET_MODES for fold in FOLDS
    ]


def _shared_specs(root, output, filter_id):
    pools = candidate_pools()
    return [
        _task_spec(root, output, FitTask("shared", filter_id, dataset, backbone, target, f"fold{fold}", config["candidate_id"]), config)
        for dataset in DATASETS for backbone in BACKBONES for config in pools[backbone]["primary"] for target in TARGET_MODES for fold in FOLDS
    ]


def _shared_verdict(output: Path, filter_id: str) -> tuple[pd.DataFrame, dict, dict]:
    paired = _pair_rows(_results(output, stage="shared", filter_id=filter_id))
    selected = paired.loc[paired.groupby(["dataset", "backbone"]).j_pair.idxmin()].reset_index(drop=True)
    global_gate = evaluate_global_success(selected)
    safe_gate = evaluate_minimum_safe(selected)
    selected.to_csv(output / "analysis" / f"{filter_id}_shared_selected_pairs.csv", index=False)
    atomic_json(output / "analysis" / f"{filter_id}_global_gate.json", global_gate | safe_gate)
    return selected, global_gate, safe_gate


def _ranking_row(filter_id: str, rows: pd.DataFrame, seconds: float) -> dict:
    rmse = rows.rmse_improvement_pct.to_numpy()
    failure = rows[((rows.dataset == "FC1") & (rows.backbone == "gru")) | ((rows.dataset == "FC2") & (rows.backbone == "tcn"))]
    return {
        "filter_id": filter_id, "rmse_improved_pairs": int(np.sum(rmse > 0)),
        "mean_relative_rmse": float(np.mean(rows.lr_rmse / rows.direct_rmse)),
        "failure_pair_mean_improvement_pct": float(failure.rmse_improvement_pct.mean()),
        "mean_cvar95_ratio": float(rows.cvar95_ratio.mean()), "mean_worst_24h_ratio": float(rows.worst_24h_ratio.mean()),
        "mean_abs_residual_std_error": float(np.mean(np.abs(rows.residual_std_ratio - 1))),
        "mean_residual_pearson": float(rows.residual_pearson.mean()), "compute_seconds": seconds,
    }


def run_loop(root: Path, output: Path) -> dict:
    if not (output / "scheduler_calibration.json").exists():
        smoke(root, output)
    calibration = _calibration(output)
    successes, expanded, ranking_rows = {}, [], []
    for filter_id in [f"F{index}" for index in range(7)]:
        atomic_json(output / "phase_status.json", {"phase": "screening", "filter_id": filter_id, "time": _utc()})
        run_specs(_fixed_stage_specs(root, output, filter_id, ("gru", "tcn"), "screen"), output, calibration)
        screening = _pair_rows(_results(output, stage="screen", filter_id=filter_id))
        screening.to_csv(output / "analysis" / f"{filter_id}_screening_pairs.csv", index=False)
        gate = evaluate_expansion_trigger(screening)
        atomic_json(output / "analysis" / f"{filter_id}_expansion_gate.json", gate)
        if gate["FILTER_EXPANSION_TRIGGER"]:
            expanded.append(filter_id)
            run_specs(_fixed_stage_specs(root, output, filter_id, ("lstm", "bigru", "transformer"), "expand"), output, calibration)
            run_specs(_shared_specs(root, output, filter_id), output, calibration)
            selected, global_gate, safe_gate = _shared_verdict(output, filter_id)
            successes[filter_id] = global_gate["FILTER_DEV_GLOBAL_SUCCESS"]
            seconds = float(_results(output, stage="shared", filter_id=filter_id).training_seconds.sum())
            ranking_rows.append(_ranking_row(filter_id, selected, seconds))
        else:
            successes[filter_id] = False
    if should_run_f7(successes):
        filter_id = "F7"
        run_specs(_fixed_stage_specs(root, output, filter_id, ("gru", "tcn"), "screen"), output, calibration)
        screening = _pair_rows(_results(output, stage="screen", filter_id=filter_id))
        gate = evaluate_expansion_trigger(screening)
        atomic_json(output / "analysis" / f"{filter_id}_expansion_gate.json", gate)
        if gate["FILTER_EXPANSION_TRIGGER"]:
            expanded.append(filter_id)
            run_specs(_fixed_stage_specs(root, output, filter_id, ("lstm", "bigru", "transformer"), "expand"), output, calibration)
            run_specs(_shared_specs(root, output, filter_id), output, calibration)
            selected, global_gate, safe_gate = _shared_verdict(output, filter_id)
            successes[filter_id] = global_gate["FILTER_DEV_GLOBAL_SUCCESS"]
            ranking_rows.append(_ranking_row(filter_id, selected, float(_results(output, stage="shared", filter_id=filter_id).training_seconds.sum())))
    ranking = rank_filters(pd.DataFrame(ranking_rows)) if ranking_rows else pd.DataFrame()
    ranking.to_csv(output / "filter_ranking.csv", index=False)
    safe_filters = []
    for filter_id in ranking.filter_id if not ranking.empty else []:
        _, _, safe = _shared_verdict(output, filter_id)
        if safe["FILTER_MINIMUM_SAFE"]:
            safe_filters.append(filter_id)
    selected_filter = safe_filters[0] if safe_filters else "F0"
    decision = {
        "selected_filter": selected_filter, "expanded_filters": expanded, "global_success": successes,
        "f7_ran": "F7" in successes, "nonbaseline_formal_allowed": bool(safe_filters),
        "next_phase": "independent_tuning" if safe_filters else "baseline_retained",
    }
    atomic_json(output / "development_decision.json", decision)
    atomic_json(output / "phase_status.json", {"phase": decision["next_phase"], "time": _utc(), **decision})
    return decision

