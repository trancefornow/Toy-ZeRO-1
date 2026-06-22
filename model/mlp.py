"""Small MLP models used for correctness tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 10
    num_hidden_layers: int = 2
    activation: str = "gelu"
    dropout: float = 0.0


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name!r}.")


class MLPClassifier(nn.Module):
    """A compact classifier that is easy to scale for memory experiments."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        if config.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1.")

        layers: list[nn.Module] = []
        in_features = config.input_dim
        for _ in range(config.num_hidden_layers):
            layers.append(nn.Linear(in_features, config.hidden_dim))
            layers.append(_activation(config.activation))
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_features = config.hidden_dim

        layers.append(nn.Linear(in_features, config.output_dim))
        self.net = nn.Sequential(*layers)
        self.config = config

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def build_mlp(config: MLPConfig | None = None) -> MLPClassifier:
    return MLPClassifier(config or MLPConfig())


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    parameters = model.parameters()
    if trainable_only:
        return sum(param.numel() for param in parameters if param.requires_grad)
    return sum(param.numel() for param in parameters)

