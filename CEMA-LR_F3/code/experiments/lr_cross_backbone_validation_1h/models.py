from __future__ import annotations

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 32, num_layers: int = 1, head_dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False, dropout=0.0)
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(inputs)
        return self.head(self.dropout(hidden[-1]))


class BiGRURegressor(nn.Module):
    def __init__(self, input_size: int = 5, total_hidden: int = 32, num_layers: int = 1, head_dropout: float = 0.0):
        super().__init__()
        if total_hidden % 2:
            raise ValueError("total_hidden must be even")
        per_direction = total_hidden // 2
        self.gru = nn.GRU(input_size, per_direction, num_layers, batch_first=True, bidirectional=True, dropout=0.0)
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(total_hidden, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(inputs)
        final = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.head(self.dropout(final))


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_size: int = 5,
        d_model: int = 32,
        nhead: int = 2,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.0,
        lookback: int = 12,
    ):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if dim_feedforward < d_model:
            raise ValueError("dim_feedforward must be at least d_model")
        self.input_projection = nn.Linear(input_size, d_model)
        self.position = nn.Parameter(torch.zeros(1, lookback, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1] != self.position.shape[1]:
            raise ValueError(f"expected lookback {self.position.shape[1]}, got {inputs.shape[1]}")
        tokens = self.input_projection(inputs) + self.position
        encoded = self.encoder(tokens)
        return self.head(self.dropout(encoded[:, -1]))


def build_model(backbone: str, params: dict) -> nn.Module:
    if backbone == "lstm":
        return LSTMRegressor(5, params["hidden_size"], params["num_layers"], params["head_dropout"])
    if backbone == "bigru":
        return BiGRURegressor(5, params["total_hidden"], params["num_layers"], params["head_dropout"])
    if backbone == "transformer":
        return TransformerRegressor(5, params["d_model"], params["nhead"], params["num_layers"], params["dim_feedforward"], params["dropout"], 12)
    raise ValueError(f"unknown backbone: {backbone}")

