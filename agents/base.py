"""Common agent interface shared by every agent (random, A*, Q-learning, DQN, PPO)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class Agent(ABC):
    """All agents expose the same minimal API so that evaluation, the UI and the
    analytics code never need to know which algorithm is behind an agent."""

    name: str = "agent"

    def reset(self) -> None:
        """Called at the start of every episode (clear plans, caches...)."""

    @abstractmethod
    def act(self, obs: np.ndarray, env: Optional[Any] = None, deterministic: bool = True) -> int:
        """Return a discrete action in [0, 6].

        `env` is the (possibly wrapped) WarehouseEnv. Learned agents must NOT read
        hidden state from it except the public `action_mask`; privileged agents
        (A*) are allowed to read the map and must say so in their docstring.
        """

    def save(self, path: str) -> None:  # pragma: no cover - overridden by learners
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "Agent":  # pragma: no cover - overridden by learners
        raise NotImplementedError
