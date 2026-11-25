"""Random baseline agent."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from agents.base import Agent


class RandomAgent(Agent):
    """Uniform random policy.

    ``use_action_mask=False`` (default): uniform over all 7 actions (the true random baseline).
    ``use_action_mask=True``: uniform over the currently valid actions (read from
    ``env.unwrapped.action_masks()``); falls back to all actions if no env is given.
    """

    name = "random"

    def __init__(self, seed: Optional[int] = None, use_action_mask: bool = False):
        self.seed = seed
        self.use_action_mask = bool(use_action_mask)
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, env: Optional[Any] = None, deterministic: bool = True) -> int:
        if self.use_action_mask and env is not None:
            valid = np.flatnonzero(env.unwrapped.action_masks())
            if len(valid):
                return int(self._rng.choice(valid))
        return int(self._rng.integers(7))
