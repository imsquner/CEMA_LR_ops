from __future__ import annotations

import copy
import hashlib
import io
import random

import numpy as np
import torch
from torch import nn

from experiments.lr_cross_backbone_validation_1h.models import (
    BiGRURegressor,
    LSTMRegressor,
    TransformerRegressor,
)
from experiments.lr_gru_tcn_paired_1h.models import GRURegressor, TCNRegressor


def seed_all(seed: int, *, disable_cudnn: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # The old Windows workaround must not silently disable cuDNN on AutoDL Linux.
    torch.backends.cudnn.enabled = not disable_cudnn


def build_model(backbone: str, params: dict) -> nn.Module:
    if backbone == "gru":
        return GRURegressor(5, params["hidden_size"], params["num_layers"], params["head_dropout"])
    if backbone == "tcn":
        return TCNRegressor(5, params["channels"], params["residual_blocks"], params["kernel_size"], params["dropout"], 12)
    if backbone == "lstm":
        return LSTMRegressor(5, params["hidden_size"], params["num_layers"], params["head_dropout"])
    if backbone == "bigru":
        return BiGRURegressor(5, params["total_hidden"], params["num_layers"], params["head_dropout"])
    if backbone == "transformer":
        return TransformerRegressor(5, params["d_model"], params["nhead"], params["num_layers"], params["dim_feedforward"], params["dropout"], 12)
    raise ValueError(f"unknown backbone: {backbone}")


def state_hash(state: dict) -> str:
    stream = io.BytesIO()
    torch.save(state, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def paired_initial_states(backbone: str, params: dict, seed: int, *, disable_cudnn: bool = False) -> tuple[dict, dict]:
    seed_all(seed, disable_cudnn=disable_cudnn)
    initial = copy.deepcopy(build_model(backbone, params).state_dict())
    return copy.deepcopy(initial), copy.deepcopy(initial)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

