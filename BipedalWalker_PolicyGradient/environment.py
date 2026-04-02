"""Gymnasium environment factory for BipedalWalker-v3 with seeding support."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import Env


def make_env(
    config: Dict[str, Any],
    render_mode: Optional[str] = None,
) -> Env:
    """
    Create the BipedalWalker-v3 Gymnasium environment from configuration.

    Observation space is 24-dimensional continuous. Action space is
    Box(-1.0, 1.0, (4,), float32). The policy must output continuous actions
    clamped to [-1, 1] at selection time.

    Args:
        config: Parsed configuration dict containing an ``environment`` section
            with at least ``name`` (str) and optional ``seed`` (int).
        render_mode: Optional Gymnasium render mode (for example ``human``) for
            on-screen rendering during evaluation.

    Returns:
        A Gymnasium ``Env`` instance. Callers should ``close()`` when done.
    """
    env_cfg = config.get("environment", {})
    name = str(env_cfg.get("name", "BipedalWalker-v3"))
    kwargs: Dict[str, Any] = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    env = gym.make(name, **kwargs)
    return env


def reset_env(
    env: Env,
    seed: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reset the environment with optional seeding for reproducibility.

    Args:
        env: A Gymnasium environment instance.
        seed: Optional RNG seed forwarded to ``env.reset(seed=seed)``.
        options: Optional reset options passed through to the environment.

    Returns:
        Tuple of ``(observation, info)`` from the environment reset call.
    """
    if options is None:
        reset_out = env.reset(seed=seed)
    else:
        reset_out = env.reset(seed=seed, options=options)
    obs, info = reset_out[0], reset_out[1]
    return obs, info
