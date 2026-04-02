"""Placeholder rollout storage for future PPO and on-policy batched algorithms."""

from __future__ import annotations

from typing import Any


class RolloutBuffer:
    """
    Stub rollout buffer for PPO-style training.

    Not used by REINFORCE in this pass. Implement storage and batching here
    when adding PPO for comparative analysis.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Construct the buffer (not implemented).

        Args:
            *args: Placeholder positional arguments.
            **kwargs: Placeholder keyword arguments.

        Raises:
            NotImplementedError: Always, until PPO rollout logic is added.
        """
        raise NotImplementedError(
            "RolloutBuffer is not implemented in this pass; it is scaffolded "
            "for future PPO training."
        )
