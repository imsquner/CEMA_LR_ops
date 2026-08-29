from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .scheduler import Calibration, ResourceSnapshot, initial_concurrency, next_concurrency, should_pause_new_tasks
from .state import TaskStore, atomic_json


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resource_snapshot(output: Path) -> tuple[ResourceSnapshot, dict]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    ).stdout.strip().splitlines()[0]
    free_vram, total_vram = (float(value.strip()) for value in query.split(","))
    meminfo = {}
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            meminfo[key] = float(value.strip().split()[0]) / 1048576
    total_ram, available_ram = meminfo["MemTotal"], meminfo["MemAvailable"]
    swap_used = meminfo.get("SwapTotal", 0.0) - meminfo.get("SwapFree", 0.0)
    disk_free = shutil.disk_usage(output).free / 1073741824
    snapshot = ResourceSnapshot(
        free_vram, total_vram, available_ram, total_ram,
        100.0 * (1.0 - available_ram / total_ram), swap_used > 0,
        False, True,
    )
    return snapshot, {**asdict(snapshot), "swap_used_gib": swap_used, "disk_free_gib": disk_free}


def append_resource_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_specs(specs: list[dict], output: Path, calibration: Calibration, *, force_serial: bool = False) -> dict:
    output = Path(output)
    (output / "task_specs").mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    store = TaskStore(output / "tasks", max_attempts=3)
    pending = []
    for spec in specs:
        state = store.load(spec["task_id"])
        if state is None or state.status == "failed" and state.attempts < 3:
            pending.append(spec)
    snapshot, _ = resource_snapshot(output)
    concurrency = 1 if force_serial else initial_concurrency(snapshot.total_vram_mib, snapshot.free_vram_mib, calibration)
    running: dict[str, tuple[subprocess.Popen, dict, object, float]] = {}
    completed = 0
    stable = 0
    oom_events = 0
    while pending or running:
        snapshot, resource = resource_snapshot(output)
        heartbeat_ok = True
        now = time.time()
        for task_id, (process, spec, log_handle, started) in list(running.items()):
            code = process.poll()
            heartbeat = output / "tasks" / "heartbeats" / f"{task_id}.json"
            if code is None and heartbeat.exists() and now - heartbeat.stat().st_mtime > 900:
                heartbeat_ok = False
                process.terminate()
            if code is not None:
                log_handle.close()
                del running[task_id]
                if code == 0:
                    completed += 1
                    stable += 1
                else:
                    stable = 0
                    if code == 42:
                        oom_events += 1
                        concurrency = max(1, concurrency - 1)
                    state = store.load(task_id)
                    if state and state.attempts < 3:
                        pending.insert(0, spec)
        snapshot = ResourceSnapshot(**(asdict(snapshot) | {"heartbeat_ok": heartbeat_ok}))
        reasons = should_pause_new_tasks(snapshot)
        disk_free = resource["disk_free_gib"]
        if disk_free < 10:
            reasons.append("disk_free_below_10gib")
        append_resource_log(output / "resource_monitor.csv", {"time": _utc(), "running": len(running), "pending": len(pending), "concurrency": concurrency, "pause_reasons": "|".join(reasons), **resource})
        if not force_serial:
            concurrency = next_concurrency(concurrency, stable, snapshot, had_oom=oom_events > 0)
            if stable >= 3:
                stable = 0
        while pending and len(running) < concurrency and not reasons:
            spec = pending.pop(0)
            spec_path = output / "task_specs" / f"{spec['task_id']}.json"
            atomic_json(spec_path, spec)
            log_path = output / "logs" / f"{spec['task_id']}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
            process = subprocess.Popen(
                [sys.executable, "-m", "experiments.lr_filter_closed_loop_1h.worker", "--spec", str(spec_path), "--output", str(output)],
                stdout=log_handle, stderr=subprocess.STDOUT, cwd=spec["project_root"], env=environment,
            )
            running[spec["task_id"]] = (process, spec, log_handle, time.time())
        atomic_json(output / "scheduler_status.json", {
            "time": _utc(), "pending": len(pending), "running": list(running), "completed_this_call": completed,
            "concurrency": concurrency, "oom_events": oom_events, "pause_reasons": reasons,
        })
        if pending or running:
            time.sleep(2)
    return {"requested": len(specs), "completed_this_call": completed, "oom_events": oom_events}

