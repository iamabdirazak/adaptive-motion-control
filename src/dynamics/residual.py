"""The learned part of the dynamics: the torque the nominal model gets wrong.

The residual is a learned feature map followed by a linear head,

    residual = features(state)ᵀ · head

The feature network is trained offline on operating logs from many conditions.
Online, only the head changes, so adapting it is a linear estimation problem
that recursive least squares solves exactly at every control step.
"""
import numpy as np
import torch
from torch import nn


def encode(angle: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
    """The network's input: angles as sine and cosine, so it sees no wrap-around."""
    return np.concatenate([np.sin(angle), np.cos(angle), velocity, acceleration], axis=-1).astype(np.float32)


class Features(nn.Module):
    """Maps the arm's state to `width` features, the last of which is a constant 1."""

    def __init__(self, joints: int, width: int = 32, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(4 * joints, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, width - 1), nn.Tanh(),  # bounded features keep the head well conditioned
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.body(x), torch.ones_like(x[..., :1])], dim=-1)
