"""Tests for the PPO wrapper and trainer (tiny budgets)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from agents import load_agent
from agents.ppo_agent import PPOAgent
from environment import EnvConfig, WarehouseEnv
from training.evaluate import evaluate_model
from training.train_ppo import train


def test_ppo_tiny_run_save_load_predict(tmp_path):
    cfg = EnvConfig(battery_enabled=False, layout_seeds=[0])
    calls = []
    hp = {"total_timesteps": 600, "n_steps": 128, "batch_size": 32, "n_epochs": 2, "net_arch": [16, 16]}
    out = train(cfg, hp, seed=0, run_name="p", progress_callback=calls.append, models_dir=str(tmp_path / "m"),
                results_dir=str(tmp_path / "r"), callback_every=1, eval_every_steps=256, eval_episodes=2,
                tensorboard=True)
    assert out["model_path"].endswith(".zip") and os.path.exists(out["model_path"])
    meta = json.load(open(out["meta_path"]))
    assert meta["algo"] == "ppo" and meta["train"]["timesteps"] >= 600
    df = pd.read_csv(out["log_path"])
    assert list(df.columns) == ["episode", "timestep", "reward", "steps", "success", "collisions",
                                "energy_used", "epsilon", "loss"]
    assert len(df) >= 3 and calls[-1]["done"]
    assert os.path.isdir(tmp_path / "r" / "p" / "tb")
    agent = load_agent(out["model_path"])
    assert isinstance(agent, PPOAgent)
    env = WarehouseEnv(cfg)
    obs, _ = env.reset(seed=5)
    a = agent.act(obs, env)
    assert isinstance(a, int) and 0 <= a < 7
    assert agent.act(obs, env) == a  # deterministic
    res = evaluate_model(out["model_path"], cfg, 3, 100_000)
    assert len(res) == 3


def test_ppo_unknown_hparam_and_lr_alias(tmp_path):
    with pytest.raises(ValueError):
        train(EnvConfig(), {"bogus": 1}, 0, "x", models_dir=str(tmp_path), results_dir=str(tmp_path))
    out = train(EnvConfig(battery_enabled=False, layout_seeds=[0]),
                {"total_timesteps": 130, "n_steps": 64, "batch_size": 32, "lr": 1e-3, "net_arch": [8]}, 0, "y",
                models_dir=str(tmp_path / "m"), results_dir=str(tmp_path / "r"), eval_every_steps=0,
                tensorboard=False)
    assert json.load(open(out["meta_path"]))["hyperparams"]["learning_rate"] == 1e-3
