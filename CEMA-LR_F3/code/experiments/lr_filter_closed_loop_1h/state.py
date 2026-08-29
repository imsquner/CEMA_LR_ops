from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+(?:__[A-Za-z0-9_-]+)*$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class TaskState:
    task_id: str
    status: str
    attempts: int
    payload: dict


class TaskStore:
    """Filesystem task ledger; each transition is an atomic rename/write."""

    def __init__(self, root: Path, max_attempts: int = 2):
        self.root = Path(root)
        self.max_attempts = max_attempts
        for name in ("running", "done", "failed", "heartbeats"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def _validate(self, task_id: str) -> None:
        if not SAFE_ID.fullmatch(task_id):
            raise ValueError(f"unsafe task id: {task_id}")

    def _path(self, status: str, task_id: str) -> Path:
        return self.root / status / f"{task_id}.json"

    def load(self, task_id: str) -> TaskState | None:
        self._validate(task_id)
        for directory, status in (("done", "done"), ("running", "running"), ("failed", "failed")):
            path = self._path(directory, task_id)
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
                return TaskState(task_id, status, int(row["attempts"]), row)
        return None

    def claim(self, task_id: str, payload: dict) -> TaskState | None:
        self._validate(task_id)
        previous = self.load(task_id)
        if previous and previous.status in {"running", "done"}:
            return None
        attempts = 1 if previous is None else previous.attempts + 1
        if attempts > self.max_attempts:
            return None
        row = {"task_id": task_id, "attempts": attempts, "started_at": utc_now(), "payload": payload}
        atomic_json(self._path("running", task_id), row)
        return TaskState(task_id, "running", attempts, row)

    def heartbeat(self, task_id: str, progress: dict) -> None:
        current = self.load(task_id)
        if current is None or current.status != "running":
            raise RuntimeError(f"task is not running: {task_id}")
        atomic_json(self._path("heartbeats", task_id), {"task_id": task_id, "time": utc_now(), "progress": progress})

    def complete(self, task_id: str, result: dict) -> None:
        current = self.load(task_id)
        if current is None or current.status != "running":
            raise RuntimeError(f"task is not running: {task_id}")
        row = current.payload | {"completed_at": utc_now(), "result": result}
        atomic_json(self._path("done", task_id), row)
        self._path("running", task_id).unlink(missing_ok=True)

    def fail(self, task_id: str, error: str, *, retryable: bool) -> None:
        current = self.load(task_id)
        if current is None or current.status != "running":
            raise RuntimeError(f"task is not running: {task_id}")
        row = current.payload | {"failed_at": utc_now(), "error": error, "retryable": retryable}
        atomic_json(self._path("failed", task_id), row)
        self._path("running", task_id).unlink(missing_ok=True)

