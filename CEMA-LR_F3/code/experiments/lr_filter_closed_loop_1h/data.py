from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from .protocol import FILTERS, PROTOCOL
from experiments.lr_gru_tcn_paired_1h.data import causal_hourly_resample, discover_data, load_source


NON_VOLTAGE_FEATURES = (
    "current_a",
    "h2_out_flow_l_min",
    "air_out_pressure_mbar",
    "coolant_in_temp_c",
)


@dataclass
class FilterWindows:
    features: np.ndarray
    targets: np.ndarray
    anchors: np.ndarray
    raw_targets: np.ndarray
    timestamps: np.ndarray
    origin_positions: np.ndarray
    target_positions: np.ndarray
    max_input_positions: np.ndarray

    def subset(self, indices) -> "FilterWindows":
        idx = np.asarray(indices)
        return FilterWindows(*(
            getattr(self, field)[idx]
            for field in self.__dataclass_fields__
        ))


@dataclass(frozen=True)
class Fold:
    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hourly(project_root: Path, dataset: str) -> tuple[pd.DataFrame, dict]:
    path = discover_data(Path(project_root), dataset)
    source = load_source(path)
    hourly = causal_hourly_resample(source)
    return hourly, {
        "dataset": dataset,
        "path": str(path),
        "sha256": sha256(path),
        "source_rows": len(source),
        "hourly_rows": len(hourly),
    }


def make_folds(sample_count: int) -> tuple[list[Fold], int]:
    bounds = [int(sample_count * fraction) for fraction in (0.5, 0.6, 0.7, 0.8)]
    folds = [
        Fold(index + 1, np.arange(bounds[index]), np.arange(bounds[index], bounds[index + 1]))
        for index in range(3)
    ]
    return folds, bounds[-1]


def save_windows(path: Path, windows: FilterWindows, audit: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **{field: getattr(windows, field) for field in windows.__dataclass_fields__}, audit=np.array([audit], dtype=object))
    temporary.replace(path)


def load_windows(path: Path) -> tuple[FilterWindows, dict]:
    with np.load(path, allow_pickle=True) as archive:
        windows = FilterWindows(*(archive[field] for field in FilterWindows.__dataclass_fields__))
        audit = dict(archive["audit"].item())
    return windows, audit


def _filter_signals(raw_voltage: pd.Series) -> dict[str, pd.Series]:
    ema3 = raw_voltage.ewm(span=3, adjust=False).mean()
    ema5 = raw_voltage.ewm(span=5, adjust=False).mean()
    ema7 = raw_voltage.ewm(span=7, adjust=False).mean()
    ema9 = raw_voltage.ewm(span=9, adjust=False).mean()
    ema5_twice = ema5.ewm(span=5, adjust=False).mean()
    return {
        "raw_1h": raw_voltage,
        "ema3": ema3,
        "ema5": ema5,
        "ema7": ema7,
        "ema9": ema9,
        "dema5": 2.0 * ema5 - ema5_twice,
        "sma3": raw_voltage.rolling(window=3, min_periods=3, center=False).mean(),
    }


def prepare_filter_frames(hourly: pd.DataFrame, warmup_hours: int | None = None) -> dict[str, pd.DataFrame]:
    warmup = PROTOCOL["warmup_hours"] if warmup_hours is None else warmup_hours
    required = {"time_h", "raw_voltage_v", *NON_VOLTAGE_FEATURES}
    missing = required.difference(hourly.columns)
    if missing:
        raise ValueError(f"missing hourly columns: {sorted(missing)}")
    base = hourly.loc[:, ["time_h", "raw_voltage_v", *NON_VOLTAGE_FEATURES]].copy()
    signals = _filter_signals(base.raw_voltage_v.astype(float))
    valid = np.ones(len(base), dtype=bool)
    for signal in signals.values():
        valid &= np.isfinite(signal.to_numpy(dtype=float))
    valid[:warmup] = False
    frames = {}
    for filter_id, definition in FILTERS.items():
        frame = base.loc[valid].copy()
        frame["voltage_input"] = signals[definition["input"]].loc[valid].to_numpy(dtype=float)
        frame["target_voltage"] = signals[definition["target"]].loc[valid].to_numpy(dtype=float)
        frame["anchor_voltage"] = signals[definition["anchor"]].loc[valid].to_numpy(dtype=float)
        frames[filter_id] = frame.reset_index(drop=True)
    timestamps = [tuple(frame.time_h) for frame in frames.values()]
    if not all(value == timestamps[0] for value in timestamps):
        raise RuntimeError("filter timestamps are not aligned")
    return frames


def build_filter_windows(frame: pd.DataFrame, lookback: int = 12, horizon: int = 1) -> tuple[FilterWindows, dict]:
    feature_columns = ("voltage_input", *NON_VOLTAGE_FEATURES)
    values = frame.loc[:, feature_columns].to_numpy(dtype=float)
    target_values = frame.target_voltage.to_numpy(dtype=float)
    anchor_values = frame.anchor_voltage.to_numpy(dtype=float)
    raw_values = frame.raw_voltage_v.to_numpy(dtype=float)
    times = frame.time_h.to_numpy(dtype=float)
    rows = {name: [] for name in ("features", "targets", "anchors", "raw_targets", "timestamps", "origins", "targets_pos", "max_inputs")}
    audit = {"candidate_samples": 0, "input_nan": 0, "label_nan": 0, "anchor_nan": 0}
    for target_position in range(lookback - 1 + horizon, len(frame)):
        audit["candidate_samples"] += 1
        origin = target_position - horizon
        start = origin - lookback + 1
        inputs = values[start:origin + 1]
        if not np.isfinite(inputs).all():
            audit["input_nan"] += 1
            continue
        if not np.isfinite(target_values[target_position]):
            audit["label_nan"] += 1
            continue
        if not np.isfinite(anchor_values[origin]):
            audit["anchor_nan"] += 1
            continue
        rows["features"].append(inputs)
        rows["targets"].append([target_values[target_position]])
        rows["anchors"].append([anchor_values[origin]])
        rows["raw_targets"].append(raw_values[target_position])
        rows["timestamps"].append(times[target_position])
        rows["origins"].append(origin)
        rows["targets_pos"].append(target_position)
        rows["max_inputs"].append(origin)
    if not rows["features"]:
        raise ValueError("no valid filter windows")
    windows = FilterWindows(
        np.asarray(rows["features"], dtype=np.float32),
        np.asarray(rows["targets"], dtype=np.float32),
        np.asarray(rows["anchors"], dtype=np.float32),
        np.asarray(rows["raw_targets"], dtype=float),
        np.asarray(rows["timestamps"], dtype=float),
        np.asarray(rows["origins"], dtype=int),
        np.asarray(rows["targets_pos"], dtype=int),
        np.asarray(rows["max_inputs"], dtype=int),
    )
    audit["valid_samples"] = len(windows.features)
    audit["deleted_total"] = audit["candidate_samples"] - audit["valid_samples"]
    return windows, audit
