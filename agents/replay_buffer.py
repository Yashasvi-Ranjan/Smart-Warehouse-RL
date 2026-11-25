"""Uniform experience replay buffer, implemented from scratch with preallocated numpy arrays."""
from __future__ import annotations

from typing import Optional

import numpy as np


class ReplayBuffer:
    """Fixed-size ring buffer of transitions ``(s, a, r, s', terminated, mask(s'))``.

    * ``terminated`` is the TRUE terminal flag (delivery / battery depletion). Time-limit truncation is
      deliberately NOT stored as done: the bootstrap target must continue through truncation.
    * ``next_mask`` stores the valid-action mask of s' so a masked max in the Bellman target is possible.
    * Sampling is uniform with replacement from a private seeded ``numpy`` generator (reproducible).
    """

    def __init__(self, capacity: int, obs_dim: int, n_actions: int = 7, seed: Optional[int] = None,
                 device="cpu"):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.terminated = np.zeros(capacity, np.float32)
        self.next_mask = np.ones((capacity, n_actions), np.bool_)
        self.pos = 0
        self.size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, obs, action, reward, next_obs, terminated, next_mask=None) -> None:
        i = self.pos
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminated[i] = float(bool(terminated))
        self.next_mask[i] = True if next_mask is None else next_mask
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        """Uniform random batch as torch tensors on ``self.device``."""
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")
        import torch
        idx = self._rng.integers(0, self.size, size=int(batch_size))
        d = self.device
        return {
            "obs": torch.from_numpy(self.obs[idx]).to(d),
            "actions": torch.from_numpy(self.actions[idx]).to(d),
            "rewards": torch.from_numpy(self.rewards[idx]).to(d),
            "next_obs": torch.from_numpy(self.next_obs[idx]).to(d),
            "terminated": torch.from_numpy(self.terminated[idx]).to(d),
            "next_mask": torch.from_numpy(self.next_mask[idx]).to(d),
        }
