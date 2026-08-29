from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.lr_filter_closed_loop_1h.data import build_filter_windows, prepare_filter_frames
from experiments.lr_filter_closed_loop_1h.protocol import FILTERS, PROTOCOL


def hourly_fixture(rows: int = 40) -> pd.DataFrame:
    x = np.arange(rows, dtype=float)
    return pd.DataFrame({
        "time_h": x,
        "raw_voltage_v": 10.0 + x,
        "current_a": 100.0 + x,
        "h2_out_flow_l_min": 200.0 + x,
        "air_out_pressure_mbar": 300.0 + x,
        "coolant_in_temp_c": 400.0 + x,
    })


def test_filter_definitions_preserve_frozen_execution_order_and_anchor_contract():
    assert list(FILTERS) == ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"]
    assert FILTERS["F1"] == {"input": "ema5", "target": "ema9", "anchor": "ema9"}
    assert FILTERS["F2"] == {"input": "ema5", "target": "ema5", "anchor": "ema5"}
    assert FILTERS["F7"]["input"] == "sma3"
    assert PROTOCOL["warmup_hours"] == 27
    assert PROTOCOL["formal_seeds"] == [42, 2024, 2026, 3407, 7777]


def test_all_filters_are_prefix_invariant_when_future_voltage_changes():
    original = hourly_fixture()
    changed = original.copy()
    changed.loc[35:, "raw_voltage_v"] += 10000.0
    left = prepare_filter_frames(original)
    right = prepare_filter_frames(changed)
    for filter_id in FILTERS:
        pd.testing.assert_frame_equal(left[filter_id].iloc[:8], right[filter_id].iloc[:8])


def test_filter_frames_share_identical_timestamps_and_use_target_anchor():
    frames = prepare_filter_frames(hourly_fixture())
    timestamp_sets = [tuple(frame.time_h) for frame in frames.values()]
    assert all(stamps == timestamp_sets[0] for stamps in timestamp_sets)
    assert len(timestamp_sets[0]) == 13
    np.testing.assert_allclose(frames["F1"].voltage_input, frames["F2"].voltage_input)
    np.testing.assert_allclose(frames["F1"].target_voltage, frames["F1"].anchor_voltage)
    assert not np.allclose(frames["F1"].voltage_input, frames["F1"].anchor_voltage)
    np.testing.assert_allclose(frames["F2"].target_voltage, frames["F2"].anchor_voltage)


def test_windows_use_only_history_and_lr_anchor_comes_from_target_filter():
    frame = prepare_filter_frames(hourly_fixture(60))["F1"]
    windows, audit = build_filter_windows(frame, lookback=12, horizon=1)
    assert np.all(windows.max_input_positions == windows.origin_positions)
    assert np.all(windows.target_positions == windows.origin_positions + 1)
    np.testing.assert_allclose(windows.anchors[:, 0], frame.anchor_voltage.to_numpy()[windows.origin_positions])
    assert audit["deleted_total"] == 0

