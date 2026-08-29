from __future__ import annotations

import pytest

from experiments.lr_filter_closed_loop_1h.scheduler import (
    Calibration,
    ResourceSnapshot,
    initial_concurrency,
    memory_slot_mib,
    next_concurrency,
    should_pause_new_tasks,
)


def test_memory_calibration_reserves_context_fragmentation_and_twenty_percent():
    calibration = Calibration(nvidia_delta_mib=1800, torch_reserved_mib=2000, worker_rss_mib=3000, median_seconds=100)
    assert memory_slot_mib(calibration) == 3512
    assert initial_concurrency(total_vram_mib=10240, free_vram_mib=10000, calibration=calibration) == 2


def test_scheduler_increases_one_slot_only_after_three_stable_completions():
    healthy = ResourceSnapshot(free_vram_mib=5000, total_vram_mib=10240, available_ram_gib=30, total_ram_gib=40, ram_used_pct=25, swap_growing=False, xid_error=False, heartbeat_ok=True)
    assert next_concurrency(current=2, stable_completions=2, snapshot=healthy, had_oom=False) == 2
    assert next_concurrency(current=2, stable_completions=3, snapshot=healthy, had_oom=False) == 3
    assert next_concurrency(current=3, stable_completions=3, snapshot=healthy, had_oom=True) == 3


def test_scheduler_pauses_for_vram_ram_swap_xid_or_missing_heartbeat():
    snapshot = ResourceSnapshot(free_vram_mib=1900, total_vram_mib=10240, available_ram_gib=7, total_ram_gib=40, ram_used_pct=91, swap_growing=True, xid_error=False, heartbeat_ok=True)
    reasons = should_pause_new_tasks(snapshot)
    assert set(reasons) == {"gpu_free_below_20pct", "gpu_free_below_2048mib", "ram_used_above_85pct", "ram_available_below_8gib", "swap_growing"}


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("free_vram_mib", 1500, "gpu_free_below_2048mib"),
        ("free_vram_mib", 1900, "gpu_free_below_20pct"),
        ("ram_used_pct", 86, "ram_used_above_85pct"),
        ("ram_used_pct", 99, "ram_used_above_85pct"),
        ("available_ram_gib", 7, "ram_available_below_8gib"),
        ("available_ram_gib", 0, "ram_available_below_8gib"),
        ("swap_growing", True, "swap_growing"),
        ("xid_error", True, "gpu_xid_error"),
        ("heartbeat_ok", False, "worker_heartbeat_missing"),
        ("free_vram_mib", 0, "gpu_free_below_20pct"),
        ("free_vram_mib", 2047, "gpu_free_below_2048mib"),
        ("ram_used_pct", 85.1, "ram_used_above_85pct"),
        ("available_ram_gib", 7.99, "ram_available_below_8gib"),
        ("free_vram_mib", 1999, "gpu_free_below_20pct"),
        ("free_vram_mib", 100, "gpu_free_below_2048mib"),
    ],
)
def test_pause_boundaries(field, value, reason):
    values = dict(free_vram_mib=8000, total_vram_mib=10000, available_ram_gib=50, total_ram_gib=64, ram_used_pct=20, swap_growing=False, xid_error=False, heartbeat_ok=True)
    values[field] = value
    assert reason in should_pause_new_tasks(ResourceSnapshot(**values))


@pytest.mark.parametrize(
    "total,free,delta,reserved,expected",
    [
        (10000, 10000, 1000, 1000, 2), (10000, 5000, 1000, 1000, 1),
        (10000, 3000, 1000, 1000, 1), (10000, 1000, 1000, 1000, 1),
        (10000, 10000, 5000, 1000, 1), (24000, 24000, 2000, 1000, 2),
        (8000, 8000, 2000, 2000, 1), (12000, 12000, 2500, 2500, 2),
        (12000, 7000, 2500, 2500, 1), (40000, 40000, 8000, 1000, 2),
    ],
)
def test_initial_concurrency_cases(total, free, delta, reserved, expected):
    assert initial_concurrency(total, free, Calibration(delta, reserved, 1000, 10)) == expected


@pytest.mark.parametrize(
    "current,stable,free,available,oom,expected",
    [
        (1, 0, 9000, 50, False, 1), (1, 2, 9000, 50, False, 1),
        (1, 3, 9000, 50, False, 2), (2, 3, 9000, 50, False, 3),
        (3, 3, 9000, 50, False, 4), (4, 3, 9000, 50, False, 4),
        (2, 3, 3000, 50, False, 2), (2, 3, 9000, 10, False, 2),
        (2, 3, 9000, 50, True, 2), (3, 99, 9000, 50, True, 3),
    ],
)
def test_next_concurrency_cases(current, stable, free, available, oom, expected):
    snapshot = ResourceSnapshot(free, 10000, available, 64, 20, False, False, True)
    assert next_concurrency(current, stable, snapshot, had_oom=oom) == expected
