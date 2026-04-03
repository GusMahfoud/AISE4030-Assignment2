"""Gaussian policy network for continuous BipedalWalker-style action spaces."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal, Independent


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


class GaussianPolicyNetwork(nn.Module):
    """
    Continuous Gaussian policy with a tanh-bounded mean and learnable log-std.

    The body uses configurable hidden sizes and activation. The mean head outputs
    ``action_dim`` values, each passed through ``tanh`` so means lie in [-1, 1].
    Log standard deviation is a learnable vector (not state-dependent), initialized to 0.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int],
        activation: str,
    ) -> None:
        """
        Build the policy network.

        Args:
            obs_dim: Dimension of the observation vector (24 for BipedalWalker-v3).
            action_dim: Number of action dimensions (4).
            hidden_sizes: Hidden layer widths for the shared body (e.g., [64, 64]).
            activation: Activation name from config (``tanh`` or ``relu``).
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        layers: List[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(_activation_module(activation))
            prev = h
        self.body = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map states to Gaussian mean (bounded) and per-dimension standard deviation.

        Args:
            state: Tensor of shape ``(B, obs_dim)`` or ``(obs_dim,)``.

        Returns:
            Tuple ``(mean, std)`` each of shape ``(..., action_dim)``, where
            ``mean`` is in [-1, 1] and ``std`` is ``exp(log_std)`` broadcast over batch.
        """
        x = state
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.body(x)
        mean = torch.tanh(self.mean_head(h))
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def sample_action(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action with the reparameterization trick and clamp to [-1, 1].

        Log-probability is the sum of independent Normal log-densities evaluated at the
        **pre-clamp** reparameterized sample (standard for clipped Gaussian exploration).

        Args:
            state: Observation tensor ``(obs_dim,)`` or batch ``(B, obs_dim)``.

        Returns:
            Tuple ``(action, log_prob)`` where ``action`` has the same batch shape as
            ``mean`` and ``log_prob`` is shape ``(B,)`` when batched or scalar when not.
        """
        single = state.dim() == 1
        mean, std = self.forward(state)

        dist = Independent(Normal(mean, std), 1)
        pre = dist.rsample()
        action = torch.clamp(pre, -1.0, 1.0)
        log_prob = dist.log_prob(pre)
        if single:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
        return action, log_prob

    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Return the deterministic mean action (evaluation / deployment).

        Args:
            state: Observation tensor ``(obs_dim,)`` or ``(B, obs_dim)``.

        Returns:
            Mean action with same leading dimensions as state (no sampling noise).
        """
        mean, _ = self.forward(state)
        return mean

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the entropy of the action distribution for the given state(s).

        Uses independent Normal marginals; entropy sums across action dimensions.

        Args:
            state: Observation tensor ``(obs_dim,)`` or ``(B, obs_dim)``.

        Returns:
            Entropy scalars with shape ``()`` for a single state or ``(B,)`` for batches.
        """
        mean, std = self.forward(state)
        single = mean.dim() == 1
        if single:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        dist = Independent(Normal(mean, std), 1)
        ent = dist.entropy()
        if single:
            ent = ent.squeeze(0)
        return ent
