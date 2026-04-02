"""REINFORCE with baseline (value network) for continuous control."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from environment import reset_env
from policy_network import GaussianPolicyNetwork
from utils import get_device
from value_network import ValueNetwork


def _discount_returns(rewards: List[float], gamma: float) -> List[float]:
    """
    Compute discounted returns G_t for each time step of an episode.

    Args:
        rewards: List of scalar rewards ``r_0, ..., r_{T-1}``.
        gamma: Discount factor in [0, 1).

    Returns:
        List of returns ``G_0, ..., G_{T-1}`` with the same length as ``rewards``.
    """
    g = 0.0
    returns: List[float] = []
    for r in reversed(rewards):
        g = float(r) + gamma * g
        returns.append(g)
    returns.reverse()
    return returns


class ReinforceBaselineAgent:
    """
    Episodic REINFORCE agent with a learned state-value baseline.

    Performs one gradient update per completed episode using separate optimizers
    for the policy and value networks, with optional return normalization and
    gradient clipping.
    """

    def __init__(self, config: Dict[str, Any], obs_dim: int, action_dim: int) -> None:
        """
        Construct networks and optimizers from configuration.

        Args:
            config: Full YAML config (nested dict) including ``shared`` and ``reinforce``.
            obs_dim: Length of the observation vector.
            action_dim: Number of continuous action dimensions.
        """
        self.config = config
        shared = config["shared"]
        rf = config["reinforce"]

        self.gamma = float(shared["gamma"])
        self.max_grad_norm = float(shared["max_grad_norm"])
        self.normalize_returns = bool(rf.get("normalize_returns", True))
        self.return_norm_eps = float(rf.get("return_norm_epsilon", 1e-8))

        hidden: List[int] = [int(x) for x in shared["hidden_sizes"]]
        activation = str(shared["activation"])
        device_str = str(shared["device"])
        self.device = get_device(device_str)

        self.policy = GaussianPolicyNetwork(
            obs_dim, action_dim, hidden, activation
        ).to(self.device)
        self.value_net = ValueNetwork(obs_dim, hidden, activation).to(self.device)

        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=float(rf["policy_lr"])
        )
        self.value_optimizer = Adam(
            self.value_net.parameters(), lr=float(rf["value_lr"])
        )

        self._states: List[torch.Tensor] = []
        self._actions: List[torch.Tensor] = []
        self._log_probs: List[torch.Tensor] = []
        self._rewards: List[float] = []

    def reset_episode_storage(self) -> None:
        """Clear per-episode trajectory buffers after an update or reset."""
        self._states.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._rewards.clear()

    def select_action(
        self, obs: np.ndarray, stochastic: bool = True
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Choose an action for the environment; optionally record training data.

        When ``stochastic`` is True, episode buffers store state, action tensor,
        and log-probability for ``end_episode_update``.

        Args:
            obs: Environment observation as a NumPy vector.
            stochastic: If True, sample from the Gaussian; else use deterministic mean.

        Returns:
            Tuple of ``(action, log_prob)``. ``action`` is a ``float32`` NumPy array
            of shape ``(action_dim,)``, clamped to [-1, 1]. ``log_prob`` is the summed
            log-density of the pre-clamp Gaussian sample (Section 4.2.1), or ``None``
            in deterministic mode.
        """
        st = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        log_prob_out: Optional[float] = None
        if stochastic:
            action_t, logp = self.policy.sample_action(st)
            self._states.append(st)
            self._actions.append(action_t)
            self._log_probs.append(logp)
            log_prob_out = float(logp.detach().cpu().item())
        else:
            action_t = self.policy.deterministic_action(st)
        a = action_t.detach().cpu().numpy().astype(np.float32)
        return np.clip(a, -1.0, 1.0), log_prob_out

    def record_reward(self, reward: float) -> None:
        """
        Append the scalar reward for the most recent transition when training.

        Args:
            reward: Environment reward for the step that just occurred.
        """
        self._rewards.append(float(reward))

    def end_episode_update(self) -> Tuple[float, float, float]:
        """
        Apply one REINFORCE + baseline update after a full episode.

        Discounted Monte Carlo returns ``G_t`` follow Section 4.2.2 of the
        assignment PDF. The value loss is MSE between ``V(s_t)`` and the **raw**
        discounted returns ``G_t`` (Section 4.2.4: actual returns). The policy
        term uses ``A_t = G_t - V(s_t)`` with ``V`` detached for the actor; when
        ``normalize_returns`` is True, that baseline-subtracted vector is
        standardized per episode (mean/variance with epsilon) before multiplying
        log-probabilities, giving a variance-reduced policy gradient signal while
        the critic still regresses unscaled returns.

        Returns:
            Tuple ``(policy_loss, value_loss, mean_entropy)`` as Python floats.
            ``mean_entropy`` is the mean Gaussian policy entropy over visited
            states **before** the optimizer step (Section 5.4 logging).

        Raises:
            RuntimeError: If episode buffers are empty or length-mismatched.
        """
        if not self._rewards:
            raise RuntimeError("end_episode_update called with no recorded rewards")
        if not (len(self._states) == len(self._actions) == len(self._log_probs)):
            raise RuntimeError("Inconsistent episode buffer lengths")

        returns = _discount_returns(self._rewards, self.gamma)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        states_b = torch.stack(self._states, dim=0)
        log_probs_b = torch.stack(self._log_probs, dim=0)
        values = self.value_net(states_b).squeeze(-1)

        with torch.no_grad():
            mean_entropy = float(self.policy.entropy(states_b).mean().item())

        advantage = ret_t - values.detach()
        if self.normalize_returns:
            a_mean = advantage.mean()
            a_std = advantage.std(unbiased=False).clamp_min(0.0)
            policy_signal = (advantage - a_mean) / (a_std + self.return_norm_eps)
        else:
            policy_signal = advantage

        policy_loss = -(log_probs_b * policy_signal.detach()).mean()
        value_loss = nn.functional.mse_loss(values, ret_t)

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
        self.value_optimizer.step()

        self.reset_episode_storage()

        return (
            float(policy_loss.item()),
            float(value_loss.item()),
            mean_entropy,
        )

    def save(self, path: str | Path) -> None:
        """
        Persist policy, value network weights, and optimizer states.

        Args:
            path: Destination ``.pt`` file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value": self.value_net.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path, map_location: Optional[str] = None) -> None:
        """
        Load policy, value network, and optimizer checkpoints.

        Args:
            path: Checkpoint file produced by ``save``.
            map_location: Optional torch ``map_location`` (defaults to agent device).
        """
        loc = map_location or str(self.device)
        ckpt = torch.load(path, map_location=loc, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.value_net.load_state_dict(ckpt["value"])
        if "policy_optimizer" in ckpt:
            self.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
        if "value_optimizer" in ckpt:
            self.value_optimizer.load_state_dict(ckpt["value_optimizer"])

    def evaluate(
        self,
        env: Any,
        num_episodes: int,
        seed: Optional[int] = None,
        render: bool = False,
    ) -> Tuple[float, float]:
        """
        Run deterministic (mean-action) evaluation episodes.

        Args:
            env: Gymnasium environment.
            num_episodes: Number of evaluation rollouts.
            seed: Optional base seed passed to ``env.reset(seed=...)`` each episode.
            render: If True, call ``env.render()`` each step when available.

        Returns:
            Tuple ``(mean_episode_reward, mean_episode_length)``.
        """
        rewards: List[float] = []
        lengths: List[int] = []
        for ep in range(num_episodes):
            rs, ep_seed = seed, None
            if rs is not None:
                ep_seed = int(rs) + ep
            obs, _ = reset_env(env, seed=ep_seed)
            done = False
            total = 0.0
            steps = 0
            while not done:
                a, _ = self.select_action(obs, stochastic=False)
                if render:
                    try:
                        env.render()
                    except Exception:
                        pass
                obs, r, term, trunc, _ = env.step(a)
                total += float(r)
                steps += 1
                done = bool(term or trunc)
            rewards.append(total)
            lengths.append(steps)
        return float(np.mean(rewards)), float(np.mean(lengths))
