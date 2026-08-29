from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np
import torch

from .data import load_windows, make_folds
from .metrics import compute_metrics
from .models import count_parameters, paired_initial_states
from .search_space import strip_metadata
from .state import TaskStore, atomic_json
from .training import predict_physical, train_model


def _indices(windows, replicate: str, stage: str):
    folds, development_end = make_folds(len(windows.features))
    if replicate.startswith("fold"):
        fold = folds[int(replicate.removeprefix("fold")) - 1]
        return fold.train_indices, fold.validation_indices
    if stage == "formal" and replicate.startswith("seed"):
        return np.arange(development_end), np.arange(development_end, len(windows.features))
    raise ValueError(f"unsupported replicate: {replicate}")


def execute_fit(spec: dict, output_root: Path) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    windows, data_audit = load_windows(Path(spec["windows_path"]))
    train_indices, evaluation_indices = _indices(windows, spec["replicate"], spec["stage"])
    params = strip_metadata(spec["config"])
    seed = int(spec.get("seed", 42))
    left, right = paired_initial_states(spec["backbone"], params, seed, disable_cudnn=bool(spec.get("disable_cudnn", False)))
    initial = left if spec["target_mode"] == "direct" else right
    heartbeat_path = output_root / "tasks" / "heartbeats" / f"{spec['task_id']}.json"

    def heartbeat(epoch, maximum, row):
        atomic_json(heartbeat_path, {"task_id": spec["task_id"], "pid": os.getpid(), "epoch": epoch, "max_epochs": maximum, "metrics": row})

    fixed_epochs = spec.get("fixed_epochs") if spec["stage"] == "formal" else None
    result = train_model(
        spec["backbone"], params, windows, train_indices, evaluation_indices,
        spec["target_mode"], seed, spec.get("device", "cuda:0"),
        max_epochs=int(spec.get("max_epochs", 120)), validate_every=2,
        patience_checks=int(spec.get("patience_checks", 15)), fixed_epochs=fixed_epochs,
        initial_state=initial, heartbeat=heartbeat,
        disable_cudnn=bool(spec.get("disable_cudnn", False)),
    )
    prediction = predict_physical(result, windows, evaluation_indices, spec["target_mode"], params["batch_size"], spec.get("device", "cuda:0"))
    truth = windows.targets[evaluation_indices, 0]
    metrics = compute_metrics(truth, prediction)
    payload = {
        **{key: spec[key] for key in ("task_id", "stage", "filter_id", "dataset", "backbone", "target_mode", "replicate", "config_id")},
        **metrics,
        "best_epoch": result.best_epoch,
        "training_seconds": result.seconds,
        "peak_allocated_mib": result.peak_allocated_mib,
        "peak_reserved_mib": result.peak_reserved_mib,
        "parameter_count": count_parameters(result.model),
        "initial_state_hash": result.initial_state_hash,
        "data_sha256": data_audit.get("data_sha256"),
    }
    if spec["stage"] == "formal":
        artifact = output_root / "task_artifacts" / spec["task_id"]
        artifact.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": result.model.state_dict(), "scaler": result.scaler.__dict__, "config": spec["config"],
            "seed": seed, "fixed_epochs": fixed_epochs, "target_mode": spec["target_mode"],
        }, artifact / "checkpoint.pt")
        np.savez_compressed(
            artifact / "predictions.npz", timestamp=windows.timestamps[evaluation_indices],
            truth=truth, prediction=prediction, raw_truth=windows.raw_targets[evaluation_indices],
            anchor=windows.anchors[evaluation_indices, 0],
        )
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    output = Path(args.output)
    store = TaskStore(output / "tasks", max_attempts=3)
    claim = store.claim(spec["task_id"], spec)
    if claim is None:
        existing = store.load(spec["task_id"])
        return 0 if existing and existing.status == "done" else 3
    try:
        result = execute_fit(spec, output)
        store.complete(spec["task_id"], result)
        print(json.dumps({"event": "task_done", "task_id": spec["task_id"], "rmse": result["rmse"]}), flush=True)
        return 0
    except torch.cuda.OutOfMemoryError:
        store.fail(spec["task_id"], traceback.format_exc(), retryable=True)
        return 42
    except Exception:
        store.fail(spec["task_id"], traceback.format_exc(), retryable=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
