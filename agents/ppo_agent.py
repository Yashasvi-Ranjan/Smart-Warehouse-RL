"""PPO agent: thin wrapper around Stable-Baselines3 PPO implementing the project's Agent API."""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np

from agents.base import Agent


class PPOAgent(Agent):
    name = "ppo"

    def __init__(self, model: Any, obs_rms: Optional[dict] = None):
        self.model = model
        self.obs_rms = obs_rms  # unused placeholder: observations are already normalised to [-1, 1] by the env

    def act(self, obs: np.ndarray, env: Optional[Any] = None, deterministic: bool = True) -> int:
        action, _ = self.model.predict(np.asarray(obs, np.float32), deterministic=deterministic)
        return int(action)

    def save(self, path: str, meta: Optional[dict] = None) -> None:
        path = str(path)
        if path.endswith(".zip"):
            path = path[:-4]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.model.save(path)  # SB3 appends .zip
        if meta is not None:
            with open(path + ".meta.json", "w") as fh:
                json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PPOAgent":
        from stable_baselines3 import PPO
        return cls(PPO.load(str(path), device=device))
