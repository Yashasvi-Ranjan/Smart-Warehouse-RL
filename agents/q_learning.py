"""Tabular Q-learning agent, implemented from scratch with numpy + dicts (no RL library)."""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np

from agents.base import Agent

N_ACTIONS = 7
HPARAM_KEYS = ("alpha", "gamma", "epsilon_start", "epsilon_end", "epsilon_decay_episodes",
               "battery_buckets", "use_action_mask")


class QLearningAgent(Agent):
    """Epsilon-greedy tabular Q-learning.

    Table: ``dict[tuple] -> np.ndarray(7, float32)``; unseen states are created lazily as zeros.
    Keys come from ONE helper, :meth:`state_key`, i.e. ``env.unwrapped.discrete_state(battery_buckets)``
    (the observable tuple documented in INTERFACES.md section 3), used identically in training and
    in :meth:`act`. The agent reads nothing else from the env except ``action_masks()`` (only if
    ``use_action_mask``).
    """

    name = "qlearning"

    def __init__(self, alpha: float = 0.2, gamma: float = 0.99, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05, epsilon_decay_episodes: int = 12000,
                 battery_buckets: int = 6, use_action_mask: bool = False, seed: Optional[int] = None):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_episodes = int(epsilon_decay_episodes)
        self.battery_buckets = int(battery_buckets)
        self.use_action_mask = bool(use_action_mask)
        self.seed = seed
        self.q: dict = {}
        self.epsilon = self.epsilon_start  # used by act(deterministic=False)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ helpers
    def hparams(self) -> dict:
        return {k: getattr(self, k) for k in HPARAM_KEYS}

    def state_key(self, env: Any) -> tuple:
        """The single place where table keys are built."""
        return tuple(env.unwrapped.discrete_state(self.battery_buckets))

    def q_values(self, key: tuple) -> np.ndarray:
        v = self.q.get(key)
        if v is None:
            v = np.zeros(N_ACTIONS, dtype=np.float32)
            self.q[key] = v
        return v

    def epsilon_at(self, episode: int) -> float:
        """Linear decay start -> end over ``epsilon_decay_episodes`` episodes, then constant."""
        n = self.epsilon_decay_episodes
        if n <= 0:
            return self.epsilon_end
        frac = min(1.0, max(0.0, episode / n))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def greedy_action(self, key: tuple, mask: Optional[np.ndarray] = None) -> int:
        """argmax with uniformly random tie-breaking (among valid actions if a mask is given)."""
        q = self.q_values(key)
        if mask is not None:
            if not mask.any():
                mask = None
        if mask is None:
            best = q.max()
            cand = np.flatnonzero(q == best)
        else:
            best = q[mask].max()
            cand = np.flatnonzero(mask & (q == best))
        return int(cand[0]) if len(cand) == 1 else int(self._rng.choice(cand))

    def select_action(self, key: tuple, epsilon: float, mask: Optional[np.ndarray] = None) -> int:
        """Epsilon-greedy; exploration is uniform over valid actions when masking."""
        if epsilon > 0 and self._rng.random() < epsilon:
            if mask is not None and mask.any():
                return int(self._rng.choice(np.flatnonzero(mask)))
            return int(self._rng.integers(N_ACTIONS))
        return self.greedy_action(key, mask)

    def update(self, key: tuple, action: int, reward: float, next_key: tuple, terminated: bool,
               next_mask: Optional[np.ndarray] = None) -> float:
        """Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') * (1 - terminated) - Q(s,a)).

        Only ``terminated`` (delivery / battery depletion = true terminal states) zeroes the bootstrap.
        Time-limit *truncation* is NOT terminal: the state after a truncated step still has value, so
        we keep bootstrapping through it (otherwise the agent would see time-out as a fake end of
        the world). Returns the TD error."""
        q = self.q_values(key)
        qn = self.q_values(next_key)
        if next_mask is not None and next_mask.any():
            nxt = float(qn[next_mask].max())
        else:
            nxt = float(qn.max())
        target = reward + self.gamma * nxt * (0.0 if terminated else 1.0)
        td = target - float(q[action])
        q[action] = q[action] + self.alpha * td
        return td

    # ------------------------------------------------------------------ Agent API
    def act(self, obs: np.ndarray, env: Optional[Any] = None, deterministic: bool = True) -> int:
        if env is None:
            raise ValueError("QLearningAgent.act needs env to build the discrete state")
        key = self.state_key(env)
        mask = env.unwrapped.action_masks() if self.use_action_mask else None
        eps = 0.0 if deterministic else self.epsilon
        return self.select_action(key, eps, mask)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str, meta: Optional[dict] = None) -> None:
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        keys = list(self.q.keys())
        kw = len(keys[0]) if keys else 12
        karr = np.array(keys, dtype=np.int32).reshape(len(keys), kw)
        qarr = (np.stack([self.q[k] for k in keys]).astype(np.float32)
                if keys else np.zeros((0, N_ACTIONS), np.float32))
        with open(path, "wb") as fh:  # file handle: numpy must not append another extension
            np.savez_compressed(fh, keys=karr, q=qarr, hparams=np.array(json.dumps(self.hparams())),
                                seed=np.array(-1 if self.seed is None else int(self.seed)))
        if meta is not None:
            with open(os.path.splitext(path)[0] + ".meta.json", "w") as fh:
                json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        path = str(path)
        with np.load(path, allow_pickle=False) as z:
            hp = json.loads(str(z["hparams"]))
            seed = int(z["seed"])
            agent = cls(seed=None if seed < 0 else seed, **hp)
            for k, row in zip(z["keys"], z["q"]):
                agent.q[tuple(int(x) for x in k)] = row.astype(np.float32).copy()
        agent.epsilon = agent.epsilon_end
        return agent
