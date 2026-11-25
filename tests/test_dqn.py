"""Tests for the replay buffer, DQN agent, DQN trainer and evaluate.py."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from agents import load_agent
from agents.dqn_agent import DQNAgent
from agents.replay_buffer import ReplayBuffer
from environment import EnvConfig, WarehouseEnv
from training.evaluate import compare_agents, evaluate_model
from training.train_dqn import train


def tiny_env_cfg(**kw):
    return EnvConfig(battery_enabled=False, layout_seeds=[0], **kw)


# ----------------------------------------------------------------- replay buffer
def test_buffer_add_sample_wraparound():
    b = ReplayBuffer(5, 3, 7, seed=0)
    for i in range(8):
        b.add(np.full(3, i), i % 7, float(i), np.full(3, i + 1), i == 7, None)
    assert len(b) == 5 and b.pos == 3
    assert sorted(b.rewards.tolist()) == [3, 4, 5, 6, 7]  # oldest 3 overwritten
    batch = b.sample(16)
    assert batch["obs"].shape == (16, 3) and batch["obs"].dtype == torch.float32
    assert batch["actions"].dtype == torch.int64
    assert torch.equal(batch["next_obs"][:, 0], batch["obs"][:, 0] + 1)
    assert (batch["terminated"][batch["rewards"] == 7] == 1).all()


def test_buffer_seeded_and_empty():
    def mk():
        b = ReplayBuffer(50, 2, 7, seed=3)
        for i in range(50):
            b.add([i, i], 0, i, [i, i], False)
        return b
    assert torch.equal(mk().sample(10)["rewards"], mk().sample(10)["rewards"])
    with pytest.raises(ValueError):
        ReplayBuffer(5, 2).sample(1)


# ----------------------------------------------------------------- Bellman target
def _agent(**kw):
    return DQNAgent(obs_dim=2, hidden=[4], gamma=0.9, seed=0, **kw)


def _set_target(agent, table):
    """Make target_net return fixed Q-values for 1 specific input regardless of weights."""
    with torch.no_grad():
        for p in agent.target_net.parameters():
            p.zero_()
        agent.target_net.net[-1].bias.copy_(torch.tensor(table, dtype=torch.float32))


def test_bellman_target_terminated_vs_truncated():
    ag = _agent()
    _set_target(ag, [1, 5, 2, 0, 0, 0, 3])
    b = ReplayBuffer(4, 2, 7, seed=0)
    b.add([0, 0], 0, 1.0, [0, 0], False)   # normal / truncated step: bootstraps
    b.add([0, 0], 0, 1.0, [0, 0], True)    # terminated: no bootstrap
    batch = {k: v[:2] for k, v in {kk: torch.from_numpy(vv[:2]) for kk, vv in
             {"obs": b.obs, "actions": b.actions, "rewards": b.rewards, "next_obs": b.next_obs,
              "terminated": b.terminated, "next_mask": b.next_mask}.items()}.items()}
    y = ag.compute_targets(batch)
    assert y[0].item() == pytest.approx(1.0 + 0.9 * 5)
    assert y[1].item() == pytest.approx(1.0)


def test_masked_target_and_double_dqn():
    ag = _agent(use_action_mask=True)
    _set_target(ag, [1, 5, 2, 0, 0, 0, 3])
    mask = torch.tensor([[True, False, True, False, False, False, True]])
    batch = {"obs": torch.zeros(1, 2), "actions": torch.zeros(1, dtype=torch.int64), "rewards": torch.zeros(1),
             "next_obs": torch.zeros(1, 2), "terminated": torch.zeros(1), "next_mask": mask}
    assert ag.compute_targets(batch).item() == pytest.approx(0.9 * 3)
    # double DQN: online net picks the action, target evaluates it
    dd = _agent(double_dqn=True)
    _set_target(dd, [1, 5, 2, 0, 0, 0, 3])
    with torch.no_grad():
        for p in dd.q_net.parameters():
            p.zero_()
        dd.q_net.net[-1].bias.copy_(torch.tensor([0, 0, 9, 0, 0, 0, 0.0]))
    assert dd.compute_targets(batch).item() == pytest.approx(0.9 * 2)


# ----------------------------------------------------------------- target net / epsilon
def test_hard_and_soft_target_update():
    ag = _agent(target_update_every=3)
    ag.q_net.net[0].weight.data.fill_(1.0)
    ag.grad_steps = 2
    assert not torch.equal(ag.q_net.net[0].weight, ag.target_net.net[0].weight)
    ag.grad_steps = 3
    ag._update_target()
    assert torch.equal(ag.q_net.net[0].weight, ag.target_net.net[0].weight)
    sg = _agent(tau=0.1)
    sg.q_net.net[0].weight.data.fill_(1.0)
    sg.target_net.net[0].weight.data.fill_(0.0)
    sg._update_target()
    assert torch.allclose(sg.target_net.net[0].weight, torch.full_like(sg.target_net.net[0].weight, 0.1))


def test_epsilon_decay():
    ag = DQNAgent(obs_dim=2, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=100)
    assert ag.epsilon_at(0) == 1.0
    assert ag.epsilon_at(50) == pytest.approx(0.55)
    assert ag.epsilon_at(100) == pytest.approx(0.1) and ag.epsilon_at(10_000) == pytest.approx(0.1)


def test_masking_in_action_selection():
    ag = DQNAgent(obs_dim=3, hidden=[8], seed=0, use_action_mask=True)
    obs = np.zeros(3, np.float32)
    mask = np.array([0, 0, 1, 0, 0, 0, 0], bool)
    assert ag.greedy_action(obs, mask) == 2
    assert all(ag.select_action(obs, 1.0, mask) == 2 for _ in range(20))


def test_learn_step_reduces_loss_on_fixed_batch():
    ag = DQNAgent(obs_dim=2, hidden=[16], lr=1e-2, seed=0, target_update_every=10_000)
    b = ReplayBuffer(32, 2, 7, seed=0)
    for i in range(32):
        b.add([i / 32, 0.5], i % 3, 1.0, [0, 0], True)
    batch = b.sample(32)
    l0 = ag.learn_on_batch(batch)
    for _ in range(100):
        l = ag.learn_on_batch(batch)
    assert l < l0 * 0.2


# ----------------------------------------------------------------- persistence
def test_save_load_same_greedy_actions_and_resume(tmp_path):
    ag = DQNAgent(obs_dim=43, hidden=[16, 16], seed=1, learning_starts=10, batch_size=8)
    env = WarehouseEnv(tiny_env_cfg())
    obs, _ = env.reset(seed=0)
    for _ in range(40):
        a = ag.select_action(obs, 0.5)
        nobs, r, te, tr, info = env.step(a)
        ag.observe(obs, a, r, nobs, te, info["action_mask"])
        obs = nobs
        if te or tr:
            obs, _ = env.reset(seed=1)
    p = str(tmp_path / "m.pt")
    ag.save(p)
    ag2 = DQNAgent.load(p)
    assert ag2.env_steps == ag.env_steps and ag2.grad_steps == ag.grad_steps and ag2.epsilon == ag.epsilon
    for s in range(5):
        o, _ = env.reset(seed=50 + s)
        assert ag.act(o, env) == ag2.act(o, env)
        assert np.allclose(ag.q_values(o), ag2.q_values(o))
    # resume: optimizer state restored => identical next update
    assert len(ag2.optimizer.state_dict()["state"]) == len(ag.optimizer.state_dict()["state"]) > 0
    batch = ag.buffer.sample(8)
    ag2.buffer = None
    l1, l2 = ag.learn_on_batch(batch), ag2.learn_on_batch(batch)
    assert l1 == pytest.approx(l2, rel=1e-6)


# ----------------------------------------------------------------- trainer
FAST = dict(episodes=12, learning_starts=50, batch_size=16, hidden=[16], epsilon_decay_steps=500,
            target_update_every=50)


def test_train_outputs_and_reproducible(tmp_path):
    cfg = tiny_env_cfg()
    calls, outs = [], []
    for i in range(2):
        out = train(cfg, dict(FAST), seed=3, run_name=f"r{i}", progress_callback=calls.append,
                    models_dir=str(tmp_path / "m"), results_dir=str(tmp_path / "r"), callback_every=4,
                    eval_every=6, eval_episodes=3, tensorboard=(i == 0))
        outs.append(out)
    assert os.path.exists(outs[0]["model_path"]) and os.path.exists(outs[0]["meta_path"])
    df = pd.read_csv(outs[0]["log_path"])
    assert list(df.columns) == ["episode", "timestep", "reward", "steps", "success", "collisions",
                                "energy_used", "epsilon", "loss"]
    assert len(df) == 12 and df["epsilon"].iloc[-1] < df["epsilon"].iloc[0]
    meta = json.load(open(outs[0]["meta_path"]))
    assert meta["algo"] == "dqn" and meta["hyperparams"]["hidden"] == [16]
    assert calls[-1]["done"] and {"episode", "timestep", "reward_ma", "success_ma", "epsilon", "loss"} <= set(calls[0])
    assert os.path.isdir(tmp_path / "r" / "r0" / "tb")
    # same seed, same config => identical final (latest) weights
    a = torch.load(tmp_path / "r" / "r0" / "checkpoints" / "latest.pt", weights_only=False)["q_net"]
    b = torch.load(tmp_path / "r" / "r1" / "checkpoints" / "latest.pt", weights_only=False)["q_net"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert pd.read_csv(outs[1]["log_path"])["reward"].tolist() == df["reward"].tolist()


def test_stop_event_and_resume(tmp_path):
    import threading
    ev = threading.Event()
    cfg = tiny_env_cfg()
    n = []

    def cb(d):
        n.append(d)
        if d["episode"] >= 4:
            ev.set()
    out = train(cfg, dict(FAST, episodes=50), 0, "s", cb, ev, str(tmp_path / "m"), str(tmp_path / "r"),
                callback_every=2, eval_every=0, tensorboard=False)
    df = pd.read_csv(out["log_path"])
    assert 4 <= len(df) < 50 and os.path.exists(out["model_path"])
    out2 = train(cfg, dict(FAST, episodes=3), 0, "s2", None, None, str(tmp_path / "m"), str(tmp_path / "r"),
                 eval_every=0, tensorboard=False,
                 resume_from=str(tmp_path / "r" / "s" / "checkpoints" / "latest.pt"))
    assert pd.read_csv(out2["log_path"])["episode"].iloc[0] == len(df)


def test_unknown_hparam_raises(tmp_path):
    with pytest.raises(ValueError):
        train(tiny_env_cfg(), {"bogus": 1}, 0, "x", models_dir=str(tmp_path), results_dir=str(tmp_path))


def test_dqn_learns_trivial_setup_beats_random(tmp_path):
    """No battery, single layout, small budget: greedy DQN should clearly beat the random policy."""
    cfg = tiny_env_cfg()
    hp = dict(episodes=600, learning_starts=300, batch_size=64, hidden=[64, 64], epsilon_decay_steps=10000,
              target_update_every=200, use_action_mask=True)
    out = train(cfg, hp, seed=0, run_name="learn", models_dir=str(tmp_path / "m"),
                results_dir=str(tmp_path / "r"), eval_every=0, tensorboard=False)
    df = pd.read_csv(out["log_path"])
    assert df["success"].iloc[-100:].mean() > df["success"].iloc[:100].mean()
    d = evaluate_model(out["model_path"], cfg, 20, 100_000)
    r = evaluate_model("random", cfg, 20, 100_000)
    assert d["success"].mean() >= 0.6
    assert d["reward"].mean() > r["reward"].mean() + 50


def test_evaluate_end_to_end(tmp_path):
    cfg = tiny_env_cfg()
    csv = str(tmp_path / "e.csv")
    df = compare_agents(["random", "astar"], cfg, 4, 100_000, out_csv=csv)
    assert set(df["label"]) == {"random", "astar"} and len(df) == 8
    a, b = df[df.label == "random"], df[df.label == "astar"]
    assert a["seed"].tolist() == b["seed"].tolist() and a["layout_seed"].tolist() == b["layout_seed"].tolist()
    assert len(pd.read_csv(csv)) == 8
    assert b["success"].all()
