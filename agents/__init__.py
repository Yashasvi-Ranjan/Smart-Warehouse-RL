"""Agent factory. Heavy imports (torch / SB3) are deferred until needed."""
from __future__ import annotations

import json
import os
from typing import Any

from agents.base import Agent

ALGOS = ("random", "astar", "qlearning", "dqn", "ppo")


def make_agent(algo: str, **kwargs: Any) -> Agent:
    """Create an *untrained/stateless* agent (random, astar) or an empty learner."""
    algo = algo.lower()
    if algo == "random":
        from agents.random_agent import RandomAgent
        return RandomAgent(**kwargs)
    if algo == "astar":
        from agents.astar_agent import AStarAgent
        return AStarAgent(**kwargs)
    raise ValueError(f"make_agent only builds random/astar; load trained '{algo}' with load_agent()")


def meta_path(model_path: str) -> str:
    return os.path.splitext(model_path)[0] + ".meta.json"


def read_meta(model_path: str) -> dict:
    with open(meta_path(model_path)) as f:
        return json.load(f)


def load_agent(model_path: str) -> Agent:
    """Load any trained model. The sibling `<stem>.meta.json` names the algorithm."""
    algo = read_meta(model_path)["algo"]
    if algo == "qlearning":
        from agents.q_learning import QLearningAgent
        return QLearningAgent.load(model_path)
    if algo == "dqn":
        from agents.dqn_agent import DQNAgent
        return DQNAgent.load(model_path)
    if algo == "ppo":
        from agents.ppo_agent import PPOAgent
        return PPOAgent.load(model_path)
    raise ValueError(f"unknown algo in meta: {algo}")
