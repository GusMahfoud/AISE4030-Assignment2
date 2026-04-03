from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
from torch.optim import Adam

from environment import reset_env
from policy_network import GaussianPolicyNetwork
from rollout_buffer import RolloutBuffer
from utils import get_device
from value_network import ValueNetwork
 
 
class PPOAgent:

    """
    PPO agent with clipped surrogate objective for continuous control.
 
    Collects fixed-length trajectory rollouts, computes advantages via
    Generalized Advantage Estimation (GAE), and performs multiple epochs
    of mini-batch gradient updates using a clipped objective to prevent
    destructive policy changes 
 
    Attributes:
        device (torch.device): Compute device (CPU or CUDA).
        policy (GaussianPolicyNetwork): Gaussian actor network.
        value_net (ValueNetwork): State-value critic network.
        optimizer (Adam): Shared optimizer for both networks.
        buffer (RolloutBuffer): On-policy trajectory storage.
    """
 
    def __init__(self, config: Dict[str, Any], obs_dim: int, action_dim: int, ) -> None:
        
        """
        Construct networks, optimizer, and rollout buffer from configuration.
 
        Args:
            config: Full YAML config dict including ``shared`` and ``ppo`` sections.
            obs_dim: Length of the observation vector (24 for BipedalWalker-v3).
            action_dim: Number of continuous action dimensions (4).
        """

        self.config = config
        shared = config["shared"]
        ppo_cfg = config["ppo"]
 
        # Hyperparameters
        self.gamma = float(shared["gamma"])
        self.max_grad_norm = float(shared["max_grad_norm"])
        self.clip_epsilon = float(ppo_cfg["clip_epsilon"])
        self.gae_lambda = float(ppo_cfg["gae_lambda"])
        self.value_loss_coef = float(ppo_cfg["value_loss_coef"])
        self.entropy_coef = float(ppo_cfg["entropy_coef"])
        self.epochs = int(ppo_cfg["epochs"])
        self.mini_batch_size = int(ppo_cfg["mini_batch_size"])
        self.rollout_length = int(ppo_cfg["rollout_length"])
 
        # Device
        device_str = str(shared["device"])
        self.device = get_device(device_str)
 
        # Networks
        hidden: List[int] = [int(x) for x in shared["hidden_sizes"]]
        activation = str(shared["activation"])
 
        self.policy = GaussianPolicyNetwork(
            obs_dim, action_dim, hidden, activation
        ).to(self.device)
        self.value_net = ValueNetwork(obs_dim, hidden, activation).to(self.device)
 
        # Single shared optimizer (Section 4.3.4)
        self.optimizer = Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=float(ppo_cfg["lr"]),
        )
 
        # Rollout buffer
        self.buffer = RolloutBuffer(
            self.rollout_length, obs_dim, action_dim, self.device
        )
 
        self.obs_dim = obs_dim
        self.action_dim = action_dim
 
    def select_action( self, obs: np.ndarray, stochastic: bool = True) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
        
        """
        In stochastic mode (training), samples from the Gaussian policy and
        returns the action, its log-probability, and the value estimate.
        In deterministic mode (evaluation), returns the mean action.
 
        Args:
            obs: Environment observation as a NumPy vector of shape ``(obs_dim,)``.
            stochastic: If True, sample from Gaussian; else use deterministic mean.
 
        Returns:
            Tuple of (action, log_prob, value). In deterministic mode,
            log_prob and value are None.
        """

        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
 
        if stochastic:
            with torch.no_grad():
                action_t, log_prob_t = self.policy.sample_action(state)
                value_t = self.value_net(state).squeeze(-1)
 
            action = action_t.cpu().numpy().astype(np.float32).flatten()
            action = np.clip(action, -1.0, 1.0)
            log_prob = float(log_prob_t.cpu().item())
            value = float(value_t.cpu().item())
            return action, log_prob, value
        else:
            with torch.no_grad():
                action_t = self.policy.deterministic_action(state)
            action = action_t.cpu().numpy().astype(np.float32).flatten()
            return np.clip(action, -1.0, 1.0), None, None
 
    def collect_rollout(self, env: Any, obs: np.ndarray) -> Tuple[np.ndarray, List[float], List[int]]:
       
        """
        Collect a fixed-length rollout by interacting with the environment.
        If an episode terminates mid-rollout, the environment is reset and
        collection continues until the buffer is full
 
        Args:
            env: Gymnasium environment instance.
            obs: Current observation to start collecting from.
 
        Returns:
            Tuple of (next_obs, episode_rewards, episode_lengths) where
            next_obs is the observation after the last transition,
            episode_rewards is a list of total rewards for episodes
            completed during this rollout, and episode_lengths is a list
            of corresponding episode lengths.
        """
        self.buffer.reset()
        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        current_ep_reward = 0.0
        current_ep_length = 0
 
        for _ in range(self.rollout_length):
            action, log_prob, value = self.select_action(obs, stochastic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
 
            self.buffer.add(obs, action, log_prob, float(reward), value, done)
 
            current_ep_reward += float(reward)
            current_ep_length += 1
 
            if done:
                episode_rewards.append(current_ep_reward)
                episode_lengths.append(current_ep_length)
                current_ep_reward = 0.0
                current_ep_length = 0
                next_obs, _ = reset_env(env)
 
            obs = next_obs
 
        # Bootstrap value for the last state
        with torch.no_grad():
            state_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            next_value = float(self.value_net(state_t).squeeze(-1).cpu().item())
 
        self.buffer.compute_gae(next_value, self.gamma, self.gae_lambda)
 
        return obs, episode_rewards, episode_lengths
 
    def update(self) -> Tuple[float, float, float]:
       
        """
        Perform K epochs of mini-batch PPO updates on the current rollout.
 
        For each mini-batch:
        1. Recompute log-probabilities under the current policy.
        2. Compute the probability ratio r_t(theta).
        3. Compute the clipped surrogate loss.
        4. Compute the value function MSE loss.
        5. Compute the entropy bonus.
        6. Combine into total loss and perform a gradient step.
 
        Returns:
            Tuple of (mean_policy_loss, mean_value_loss, mean_entropy)
            averaged over all mini-batch updates across all epochs.
        """

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
 
        for _ in range(self.epochs):
            for batch in self.buffer.get_batches(self.mini_batch_size):
                states, actions, old_log_probs, advantages, returns = batch
 
                # Recompute log-probs and entropy under current policy
                mean, std = self.policy(states)
                dist = Independent(Normal(mean, std), 1)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
 
                # Probability ratio (Section 4.3.3)
                ratio = torch.exp(new_log_probs - old_log_probs)
 
                # Clipped surrogate loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
 
                # Value function loss
                values = self.value_net(states).squeeze(-1)
                value_loss = nn.functional.mse_loss(values, returns)
 
                # Combined loss (Section 4.3.4)
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
 
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()
 
                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                num_updates += 1
 
        mean_policy_loss = total_policy_loss / max(num_updates, 1)
        mean_value_loss = total_value_loss / max(num_updates, 1)
        mean_entropy = total_entropy / max(num_updates, 1)
 
        return mean_policy_loss, mean_value_loss, mean_entropy
 
    def save(self, path: str | Path) -> None:
        
        """
        Persist policy, value network weights, and optimizer state.
 
        Args:
            path: Destination .pt file path.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value": self.value_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
 
    def load(self, path: str | Path, map_location: Optional[str] = None) -> None:
       
        """
        Load policy, value network, and optimizer checkpoint.
 
        Args:
            path: Checkpoint file produced by ``save``.
            map_location: Optional torch ``map_location`` (defaults to agent device).
        """

        loc = map_location or str(self.device)
        ckpt = torch.load(path, map_location=loc, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.value_net.load_state_dict(ckpt["value"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
 
    def evaluate(self, env: Any, num_episodes: int, seed: Optional[int] = None, render: bool = False,) -> Tuple[float, float]:
        
        """
        Run deterministic (mean-action) evaluation episodes.
 
        Args:
            env: Gymnasium environment.
            num_episodes: Number of evaluation rollouts.
            seed: Optional base seed passed to ``env.reset(seed=...)`` each episode.
            render: If True, call ``env.render()`` each step when available.
 
        Returns:
            Tuple (mean_episode_reward, mean_episode_length).
        """
        
        rewards: List[float] = []
        lengths: List[int] = []
        for ep in range(num_episodes):
            ep_seed = None
            if seed is not None:
                ep_seed = int(seed) + ep
            obs, _ = reset_env(env, seed=ep_seed)
            done = False
            total = 0.0
            steps = 0
            while not done:
                action, _, _ = self.select_action(obs, stochastic=False)
                if render:
                    try:
                        env.render()
                    except Exception:
                        pass
                obs, r, term, trunc, _ = env.step(action)
                total += float(r)
                steps += 1
                done = bool(term or trunc)
            rewards.append(total)
            lengths.append(steps)
        return float(np.mean(rewards)), float(np.mean(lengths))
