from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Calibration:
    nvidia_delta_mib: float
    torch_reserved_mib: float
    worker_rss_mib: float
    median_seconds: float


@dataclass(frozen=True)
class ResourceSnapshot:
    free_vram_mib: float
    total_vram_mib: float
    available_ram_gib: float
    total_ram_gib: float
    ram_used_pct: float
    swap_growing: bool
    xid_error: bool
    heartbeat_ok: bool


def memory_slot_mib(calibration: Calibration) -> int:
    return int(math.ceil(1.5 * max(calibration.nvidia_delta_mib, calibration.torch_reserved_mib) + 512.0))


def initial_concurrency(total_vram_mib: float, free_vram_mib: float, calibration: Calibration) -> int:
    reserve = max(0.20 * total_vram_mib, 2048.0)
    memory_slots = math.floor(max(0.0, free_vram_mib - reserve) / memory_slot_mib(calibration))
    return max(1, min(2, memory_slots, 4))


def should_pause_new_tasks(snapshot: ResourceSnapshot) -> list[str]:
    reasons = []
    if snapshot.free_vram_mib < 0.20 * snapshot.total_vram_mib:
        reasons.append("gpu_free_below_20pct")
    if snapshot.free_vram_mib < 2048:
        reasons.append("gpu_free_below_2048mib")
    if snapshot.ram_used_pct > 85:
        reasons.append("ram_used_above_85pct")
    if snapshot.available_ram_gib < 8:
        reasons.append("ram_available_below_8gib")
    if snapshot.swap_growing:
        reasons.append("swap_growing")
    if snapshot.xid_error:
        reasons.append("gpu_xid_error")
    if not snapshot.heartbeat_ok:
        reasons.append("worker_heartbeat_missing")
    return reasons


def next_concurrency(current: int, stable_completions: int, snapshot: ResourceSnapshot, had_oom: bool) -> int:
    if current >= 4 or stable_completions < 3 or had_oom:
        return current
    healthy = (
        snapshot.free_vram_mib > 0.35 * snapshot.total_vram_mib
        and snapshot.available_ram_gib > 0.30 * snapshot.total_ram_gib
        and not snapshot.swap_growing
        and not snapshot.xid_error
        and snapshot.heartbeat_ok
    )
    return current + 1 if healthy else current

