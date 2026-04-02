"""Placeholder for future Proximal Policy Optimization (PPO) implementation."""

from __future__ import annotations

from typing import Any, Dict


class PPOAgent:
    """
    Stub PPO agent reserved for Assignment 2 Task 3.

    This class exists so imports and configuration remain stable; training
    with ``agent_type: "ppo"`` is not supported in this repository pass.
    """

    def __init__(self, config: Dict[str, Any], obs_dim: int, action_dim: int) -> None:
        """
        Initialize the stub (not implemented).

        Args:
            config: Full configuration dictionary.
            obs_dim: Observation dimension.
            action_dim: Action dimension.

        Raises:
            NotImplementedError: Always, because PPO is not implemented.
        """
        raise NotImplementedError(
            "PPO is not implemented in this pass; set agent_type to 'reinforce' "
            "in config.yaml or extend ppo_agent.py for Task 3."
        )
