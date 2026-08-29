import json
from pathlib import Path

import pytest

from experiments.lr_filter_closed_loop_1h.state import TaskStore


def test_task_store_claim_heartbeat_complete_and_resume(tmp_path):
    store = TaskStore(tmp_path)
    task_id = "screen__F1__FC1__gru__direct__fold1"
    claimed = store.claim(task_id, {"kind": "screen"})
    assert claimed.status == "running"
    assert store.claim(task_id, {"kind": "screen"}) is None
    store.heartbeat(task_id, {"epoch": 2})
    store.complete(task_id, {"rmse": 0.1})
    assert store.load(task_id).status == "done"
    assert json.loads((tmp_path / "done" / f"{task_id}.json").read_text())["result"]["rmse"] == 0.1


def test_failed_task_can_retry_only_once(tmp_path):
    store = TaskStore(tmp_path)
    task_id = "task"
    store.claim(task_id, {})
    store.fail(task_id, "oom", retryable=True)
    assert store.claim(task_id, {}) is not None
    store.fail(task_id, "oom again", retryable=True)
    assert store.claim(task_id, {}) is None
    assert store.load(task_id).status == "failed"


def test_task_id_rejects_path_traversal(tmp_path):
    store = TaskStore(tmp_path)
    with pytest.raises(ValueError):
        store.claim("../escape", {})

