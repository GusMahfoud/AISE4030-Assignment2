
from __future__ import annotations
from typing import Any
from typing import Generator, Tuple 
import numpy as np
import torch
 
 
class RolloutBuffer:

    """
    Fixed-length trajectory buffer for PPO training.
    Stores transitions collected during a single rollout and provides
    GAE advantage computation and shuffled mini-batch iteration for
    multi-epoch updates.

    Attributes:
        rollout_length (int): Maximum number of transitions per rollout.
        obs_dim (int): Observation vector dimension.
        action_dim (int): Action vector dimension.
        device (torch.device): Device for tensor conversion.
    """
 
    def __init__(self, rollout_length: int, obs_dim: int, action_dim: int, device: torch.device,) -> None:
       
        """
        Allocate storage arrays for a fixed-length rollout.
 
        Args:
            rollout_length: Number of transitions to collect per rollout.
            obs_dim: Dimension of the observation vector.
            action_dim: Dimension of the action vector.
            device: Torch device for tensor conversion during training.
        """
        self.rollout_length = rollout_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
 
        self.states = np.zeros((rollout_length, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_length, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(rollout_length, dtype=np.float32)
        self.rewards = np.zeros(rollout_length, dtype=np.float32)
        self.values = np.zeros(rollout_length, dtype=np.float32)
        self.dones = np.zeros(rollout_length, dtype=np.float32)
 
        self.advantages = np.zeros(rollout_length, dtype=np.float32)
        self.returns = np.zeros(rollout_length, dtype=np.float32)
 
        self.ptr = 0
 
    def add(self, state: np.ndarray, action: np.ndarray, log_prob: float, reward: float, value: float, done: bool, ) -> None:
        
        """
        Store a single transition in the buffer.
 
        Args:
            state: Observation vector of shape (obs_dim,).
            action: Action vector of shape (action_dim,).
            log_prob: Log-probability of the action under the current policy.
            reward: Scalar reward from the environment.
            value: Value estimate V(s) from the critic.
            done: Whether the episode terminated at this step.
        """

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = float(done)
        self.ptr += 1
 
    def compute_gae( self, next_value: float, gamma: float, gae_lambda: float,) -> None:
        
        """
        Compute Generalized Advantage Estimation (GAE) and return targets.
        Processes the rollout in reverse order to efficiently compute the
        recursive GAE formula. Advantages are normalized 
        before being used in the loss computation. Return targets are
        computed as R_t = A_t + V(s_t).
 
        Args:
            next_value: Bootstrap value V(s') for the state after the last
                        transition in the rollout.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter controlling bias-variance tradeoff.
        """

        gae = 0.0
        for t in reversed(range(self.rollout_length)):
            if t == self.rollout_length - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]
 
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * non_terminal * next_val - self.values[t]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            self.advantages[t] = gae
 
        self.returns = self.advantages + self.values
 
        # Normalize advantages
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std()
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
 
    def get_batches(self, mini_batch_size: int, ) -> Generator[
        
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        None,
        None,
    ]:

        """
        Yield shuffled mini-batches of rollout data as device tensors.
        Each mini-batch contains a subset of the rollout transitions,
        shuffled randomly. Used for multi-epoch PPO updates.
 
        Args:
            mini_batch_size: Number of transitions per mini-batch.
 
        Returns:
            Tuple of (states, actions, old_log_probs, advantages, returns)
            where each element is a tensor on self.device.
        """

        indices = np.random.permutation(self.rollout_length)
        for start in range(0, self.rollout_length, mini_batch_size):
            end = start + mini_batch_size
            batch_idx = indices[start:end]
 
            yield (
                torch.tensor(self.states[batch_idx], dtype=torch.float32, device=self.device),
                torch.tensor(self.actions[batch_idx], dtype=torch.float32, device=self.device),
                torch.tensor(self.log_probs[batch_idx], dtype=torch.float32, device=self.device),
                torch.tensor(self.advantages[batch_idx], dtype=torch.float32, device=self.device),
                torch.tensor(self.returns[batch_idx], dtype=torch.float32, device=self.device),
            )
 
    def reset(self) -> None:
        
        """Reset the buffer pointer for a new rollout."""
        self.ptr = 0