"""State-value network V(s) matching the policy body architecture."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def _activation_module(name: str) -> nn.Module:
    """
    Map a configuration activation name to a PyTorch module.

    Args:
        name: ``"tanh"`` or ``"relu"`` (case-insensitive).

    Returns:
        An activation module instance.

    Raises:
        ValueError: If ``name`` is not supported.
    """
    n = name.lower()
    if n == "tanh":
        return nn.Tanh()
    if n == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


class ValueNetwork(nn.Module):
    """
    MLP that outputs a scalar value estimate V(s) with separate parameters from policy.

    Hidden layers mirror ``hidden_sizes`` and ``activation`` from configuration.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: List[int],
        activation: str,
    ) -> None:
        """
        Initialize the value network.

        Args:
            obs_dim: Observation dimension.
            hidden_sizes: Hidden layer widths (same as policy body).
            activation: Activation name from config.
        """
        super().__init__()
        layers: List[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(_activation_module(activation))
            prev = h
        self.net = nn.Sequential(*layers, nn.Linear(prev, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict scalar value(s) for state(s).

        Args:
            state: Tensor ``(obs_dim,)`` or batch ``(B, obs_dim)``.

        Returns:
            Shape ``(1,)`` for one state or ``(B, 1)`` for batch.
        """
        x = state
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)
