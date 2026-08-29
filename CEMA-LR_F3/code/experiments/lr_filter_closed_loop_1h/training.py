from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import FilterWindows
from .models import build_model, seed_all, state_hash


@dataclass(frozen=True)
class Scalers:
    input_mean: np.ndarray
    input_scale: np.ndarray
    target_mean: float
    target_scale: float

    def transform_x(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.input_mean) / self.input_scale).astype(np.float32)

    def transform_y(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.target_mean) / self.target_scale).astype(np.float32)


@dataclass
class TrainResult:
    model: nn.Module
    scaler: Scalers
    best_epoch: int
    history: list[dict]
    seconds: float
    peak_allocated_mib: float
    peak_reserved_mib: float
    initial_state_hash: str


def fit_scalers(windows: FilterWindows, indices: np.ndarray, target_mode: str) -> Scalers:
    indices = np.asarray(indices, dtype=int)
    flat = windows.features[indices].reshape(-1, windows.features.shape[-1])
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    target = windows.targets[indices] if target_mode == "direct" else windows.targets[indices] - windows.anchors[indices]
    target_mean = float(target.mean())
    target_scale = float(target.std()) or 1.0
    return Scalers(mean, scale, target_mean, target_scale)


def reconstruct_physical(standardized, target_mean: float, target_scale: float, anchors, target_mode: str):
    physical = np.asarray(standardized) * target_scale + target_mean
    if target_mode == "direct":
        return physical
    if target_mode == "lr":
        return np.asarray(anchors) + physical
    raise ValueError(f"unknown target mode: {target_mode}")


def _predict_standardized(model, features, batch_size, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=batch_size, shuffle=False, num_workers=0)
    parts = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            parts.append(model(batch.to(device)).detach().cpu().numpy())
    return np.concatenate(parts)


def train_model(
    backbone: str,
    params: dict,
    windows: FilterWindows,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    target_mode: str,
    seed: int,
    device: str,
    *,
    max_epochs: int = 120,
    validate_every: int = 2,
    patience_checks: int = 15,
    fixed_epochs: int | None = None,
    initial_state: dict | None = None,
    heartbeat=None,
    disable_cudnn: bool = False,
) -> TrainResult:
    torch.set_num_threads(1)
    seed_all(seed, disable_cudnn=disable_cudnn)
    scaler = fit_scalers(windows, train_indices, target_mode)
    features = scaler.transform_x(windows.features)
    physical_target = windows.targets if target_mode == "direct" else windows.targets - windows.anchors
    targets = scaler.transform_y(physical_target)
    model = build_model(backbone, params)
    initial = copy.deepcopy(model.state_dict() if initial_state is None else initial_state)
    model.load_state_dict(initial)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features[train_indices]), torch.from_numpy(targets[train_indices])),
        batch_size=params["batch_size"], shuffle=True, generator=loader_generator, num_workers=0,
    )
    loss_function = nn.MSELoss()
    best_rmse = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    epoch_limit = fixed_epochs if fixed_epochs is not None else max_epochs
    for epoch in range(1, epoch_limit + 1):
        model.train()
        running = 0.0
        samples = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(batch_x)
            samples += len(batch_x)
        row = {"epoch": epoch, "train_loss": running / samples}
        validate = fixed_epochs is None and epoch % validate_every == 0
        if validate:
            standardized = _predict_standardized(model, features[validation_indices], params["batch_size"], device)
            prediction = reconstruct_physical(standardized, scaler.target_mean, scaler.target_scale, windows.anchors[validation_indices], target_mode).reshape(-1)
            truth = windows.targets[validation_indices, 0]
            rmse = float(np.sqrt(np.mean(np.square(prediction - truth))))
            row["val_rmse"] = rmse
            if rmse < best_rmse - 1e-12:
                best_rmse, best_epoch, best_state, stale = rmse, epoch, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
        history.append(row)
        if heartbeat and (validate or epoch == epoch_limit):
            heartbeat(epoch, epoch_limit, row)
        if validate and stale >= patience_checks:
            break
    if fixed_epochs is not None:
        best_epoch, best_state = epoch_limit, copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("no validation checkpoint was produced")
    model.load_state_dict(best_state)
    seconds = time.perf_counter() - started
    allocated = torch.cuda.max_memory_allocated() / 1048576 if device.startswith("cuda") else 0.0
    reserved = torch.cuda.max_memory_reserved() / 1048576 if device.startswith("cuda") else 0.0
    return TrainResult(model, scaler, best_epoch, history, seconds, allocated, reserved, state_hash(initial))


def predict_physical(result: TrainResult, windows: FilterWindows, indices: np.ndarray, target_mode: str, batch_size: int, device: str) -> np.ndarray:
    standardized = _predict_standardized(result.model, result.scaler.transform_x(windows.features[indices]), batch_size, device)
    return reconstruct_physical(standardized, result.scaler.target_mean, result.scaler.target_scale, windows.anchors[indices], target_mode).reshape(-1)

