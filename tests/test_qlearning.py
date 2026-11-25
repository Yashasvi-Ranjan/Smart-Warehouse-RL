"""Tests for the tabular Q-learning agent and trainer."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from agents import load_agent
from agents.q_learning import QLearningAgent
from agents.random_agent import RandomAgent
from environment import EnvConfig, WarehouseEnv
from training.train_qlearning import train, train_episode_seed


def test_bellman_update_numeric():
    ag = QLearningAgent(alpha=0.5, gamma=0.9)
    s, s2 = (0,) * 12, (1,) * 12
    ag.q_values(s)[2] = 1.0
    ag.q_values(s2)[:] = [0, 4, 2, 0, 0, 0, 0]
    td = ag.update(s, 2, 3.0, s2, terminated=False)
    # target = 3 + 0.9*4 = 6.6 ; new = 1 + 0.5*(6.6-1) = 3.8
    assert td == pytest.approx(5.6, abs=1e-5)
    assert ag.q[s][2] == pytest.approx(3.8, abs=1e-5)
    # terminated => no bootstrap: target 3 ; new = 3.8 + 0.5*(3-3.8)
    ag.update(s, 2, 3.0, s2, terminated=True)
    assert ag.q[s][2] == pytest.approx(3.4, abs=1e-5)


def test_masked_bootstrap_uses_valid_actions_only():
    ag = QLearningAgent(alpha=1.0, gamma=1.0)
    s, s2 = (0,) * 12, (1,) * 12
    ag.q_values(s2)[:] = [0, 100, 2, 0, 0, 0, 0]
    mask = np.array([1, 0, 1, 0, 0, 0, 0], bool)
    ag.update(s, 0, 0.0, s2, False, mask)
    assert ag.q[s][0] == pytest.approx(2.0)


def test_epsilon_schedule():
    ag = QLearningAgent(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_episodes=100)
    assert ag.epsilon_at(0) == 1.0
    assert ag.epsilon_at(50) == pytest.approx(0.55)
    assert ag.epsilon_at(100) == pytest.approx(0.1)
    assert ag.epsilon_at(10_000) == pytest.approx(0.1)


def test_tie_breaking_and_masking():
    ag = QLearningAgent(seed=0)
    k = (0,) * 12
    ag.q_values(k)[:] = [1, 1, 0, 1, 0, 0, 0]
    picks = {ag.greedy_action(k) for _ in range(200)}
    assert picks == {0, 1, 3}
    mask = np.array([0, 1, 1, 1, 1, 1, 1], bool)
    assert {ag.greedy_action(k, mask) for _ in range(200)} == {1, 3}
    # exploration respects mask
    assert {ag.select_action(k, 1.0, mask) for _ in range(300)} <= {1, 2, 3, 4, 5, 6}
    # unseen state: zeros, all tie
    assert {ag.greedy_action((9,) * 12) for _ in range(300)} == set(range(7))


def test_act_uses_epsilon_only_when_stochastic():
    env = WarehouseEnv(EnvConfig())
    obs, _ = env.reset(seed=0)
    ag = QLearningAgent(seed=0)
    key = ag.state_key(env)
    ag.q_values(key)[3] = 5.0
    ag.epsilon = 1.0
    assert all(ag.act(obs, env, deterministic=True) == 3 for _ in range(50))
    assert len({ag.act(obs, env, deterministic=False) for _ in range(100)}) > 1


def test_save_load_roundtrip(tmp_path):
    env = WarehouseEnv(EnvConfig(layout_seeds=[0]))
    ag = QLearningAgent(alpha=0.3, battery_buckets=4, use_action_mask=True, seed=1)
    rng = np.random.default_rng(0)
    for s in range(30):
        env.reset(seed=s)
        ag.q_values(ag.state_key(env))[:] = rng.normal(size=7).astype(np.float32)
    p = tmp_path / "m.npz"
    ag.save(str(p), meta={"algo": "qlearning", "run_name": "m"})
    assert os.path.exists(tmp_path / "m.meta.json")
    assert json.load(open(tmp_path / "m.meta.json"))["algo"] == "qlearning"
    ag2 = QLearningAgent.load(str(p))
    assert ag2.hparams() == ag.hparams()
    assert set(ag2.q) == set(ag.q)
    for k in ag.q:
        np.testing.assert_array_equal(ag.q[k], ag2.q[k])
    assert isinstance(load_agent(str(p)), QLearningAgent)
    for s in range(30):
        obs, _ = env.reset(seed=s)
        assert ag.act(obs, env) == ag2.act(obs, env) or len(np.unique(ag.q[ag.state_key(env)])) < 7


def test_loaded_greedy_identical_when_no_ties(tmp_path):
    env = WarehouseEnv(EnvConfig(layout_seeds=[0]))
    ag = QLearningAgent(seed=1)
    rng = np.random.default_rng(1)
    for s in range(40):
        env.reset(seed=s)
        ag.q_values(ag.state_key(env))[:] = rng.normal(size=7)
    ag.save(str(tmp_path / "a.npz"))
    ag2 = QLearningAgent.load(str(tmp_path / "a.npz"))
    for s in range(40):
        obs, _ = env.reset(seed=s)
        assert ag.act(obs, env) == ag2.act(obs, env)


def _small_train(tmp_path, name, seed=0, episodes=1500, **kw):
    cfg = EnvConfig(layout_seeds=[0], battery_enabled=False, max_steps=80)
    return cfg, train(cfg, {"episodes": episodes, "epsilon_decay_episodes": 900, "alpha": 0.3}, seed, name,
                      models_dir=str(tmp_path / "models"), results_dir=str(tmp_path / "res"), **kw)


def test_train_outputs_and_reproducible(tmp_path):
    cfg, o1 = _small_train(tmp_path, "r1", episodes=300)
    _, o2 = _small_train(tmp_path, "r2", episodes=300)
    a, b = QLearningAgent.load(o1["model_path"]), QLearningAgent.load(o2["model_path"])
    assert set(a.q) == set(b.q)
    assert all(np.array_equal(a.q[k], b.q[k]) for k in a.q)
    _, o3 = _small_train(tmp_path, "r3", seed=1, episodes=300)
    c = QLearningAgent.load(o3["model_path"])
    assert set(c.q) != set(a.q) or any(not np.array_equal(a.q[k], c.q[k]) for k in a.q)
    df = pd.read_csv(o1["log_path"])
    assert list(df.columns) == ["episode", "timestep", "reward", "steps", "success", "collisions",
                                "energy_used", "epsilon", "loss"]
    assert len(df) == 300
    meta = json.load(open(o1["meta_path"]))
    assert meta["algo"] == "qlearning" and meta["env_config"]["battery_enabled"] is False
    assert set(o1) >= {"model_path", "meta_path", "log_path", "wall_seconds"}


def test_train_seeds_disjoint_from_eval():
    assert train_episode_seed(0, 0) > 100000 + 10_000_000
    assert train_episode_seed(0, 9_999_999) < train_episode_seed(1, 0)


def test_stop_event_and_callback(tmp_path):
    import threading
    ev = threading.Event()
    calls = []

    def cb(d):
        calls.append(d)
        if d["episode"] >= 200:
            ev.set()
    cfg = EnvConfig(layout_seeds=[0], battery_enabled=False)
    out = train(cfg, {"episodes": 5000}, 0, "stop", progress_callback=cb, stop_event=ev,
                models_dir=str(tmp_path / "m"), results_dir=str(tmp_path / "r"), callback_every=100)
    assert os.path.exists(out["model_path"])
    assert len(pd.read_csv(out["log_path"])) < 1000
    assert calls[-1]["done"] and {"episode", "timestep", "reward_ma", "success_ma", "epsilon", "loss"} <= set(calls[-1])


def test_tiny_training_beats_random(tmp_path):
    cfg, out = _small_train(tmp_path, "learn", episodes=1500)
    ag = QLearningAgent.load(out["model_path"])
    rnd = RandomAgent(seed=0)

    def succ(agent):
        env = WarehouseEnv(cfg)
        n = 0
        for s in range(100000, 100040):
            obs, _ = env.reset(seed=s)
            agent.reset()
            while True:
                obs, r, t, tr, info = env.step(agent.act(obs, env, True))
                if t or tr:
                    n += info["episode_stats"]["success"]
                    break
        return n / 40
    assert succ(ag) > succ(rnd) + 0.3
