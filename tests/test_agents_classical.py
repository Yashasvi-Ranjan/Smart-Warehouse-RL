"""Tests for RandomAgent and AStarAgent."""
import numpy as np
import pytest

from agents import make_agent
from agents.astar_agent import AStarAgent
from agents.random_agent import RandomAgent
from environment import EnvConfig, WarehouseEnv


def rollout(env, agent, seed, options=None):
    obs, info = env.reset(seed=seed, options=options)
    agent.reset()
    while True:
        obs, r, t, tr, info = env.step(agent.act(obs, env, True))
        if t or tr:
            return info["episode_stats"]


def test_random_seedable():
    env = WarehouseEnv(EnvConfig())
    obs, _ = env.reset(seed=1)
    a = [RandomAgent(seed=3).act(obs, env) for _ in range(1)]
    s1 = RandomAgent(seed=5)
    s2 = RandomAgent(seed=5)
    assert [s1.act(obs, env) for _ in range(50)] == [s2.act(obs, env) for _ in range(50)]
    assert a[0] in range(7)


def test_random_unmasked_covers_all_actions():
    env = WarehouseEnv(EnvConfig())
    obs, _ = env.reset(seed=1)
    ag = RandomAgent(seed=0)
    assert {ag.act(obs, env) for _ in range(300)} == set(range(7))


def test_random_masked_only_valid():
    env = WarehouseEnv(EnvConfig())
    obs, info = env.reset(seed=1)
    ag = RandomAgent(seed=0, use_action_mask=True)
    for _ in range(100):
        mask = env.unwrapped.action_masks()
        assert mask[ag.act(obs, env)]
        obs, r, t, tr, info = env.step(ag.act(obs, env))
        if t or tr:
            obs, info = env.reset()


def test_factory():
    assert isinstance(make_agent("random", seed=1), RandomAgent)
    assert isinstance(make_agent("astar"), AStarAgent)


@pytest.mark.parametrize("over", [
    {}, {"battery_enabled": False},
    {"battery_capacity": 24, "battery_init_range": (0.4, 0.6)}])
def test_astar_succeeds_no_collisions(over):
    env = WarehouseEnv(EnvConfig(**over))
    ag = AStarAgent()
    n_rech = 0
    for s in range(100000, 100320):
        st = rollout(env, ag, s)
        assert st["success"], (s, st)
        assert st["collisions"] == 0 and st["invalid_actions"] == 0
        if st["recharges"] == 0:
            assert st["steps"] == st["optimal_steps"]
        n_rech += st["recharges"] > 0
    if over.get("battery_capacity") == 24:
        assert n_rech > 20  # tight battery really forces detours


def test_astar_optimal_vs_bfs_and_min_task_steps():
    cfg = EnvConfig(battery_capacity=24, battery_init_range=(0.4, 0.6))
    env = WarehouseEnv(cfg)
    ag = AStarAgent()
    for s in range(100000, 100060):
        env.reset(seed=s)
        e = env.unwrapped
        m = e.warehouse_map
        exp = m.min_task_steps(e.robot_pos, e.pickup_pos, e.delivery_pos, e.battery, cfg.battery_capacity,
                               cfg.recharge_amount, True)
        st = rollout(env, ag, s)
        assert st["steps"] == exp  # A* over (cell, battery) is exactly optimal


def test_astar_detour_when_battery_low():
    env = WarehouseEnv(EnvConfig())
    env.reset(seed=3, options={"layout_seed": 0})
    e = env.unwrapped
    m = e.warehouse_map
    start = m.start_cells[0]
    ag = AStarAgent()
    obs, _ = env.reset(seed=3, options={"layout_seed": 0, "start_pos": start, "battery": 12})
    done = False
    while not done:
        obs, r, t, tr, info = env.step(ag.act(obs, env))
        done = t or tr
    st = info["episode_stats"]
    assert st["success"] and st["recharges"] >= 1 and st["collisions"] == 0


def test_astar_unreachable_safe():
    env = WarehouseEnv(EnvConfig())
    env.reset(seed=3, options={"layout_seed": 0, "battery": 1,
                               "start_pos": WarehouseEnv(EnvConfig()).reset(seed=0)[1]["robot_pos"]})
    ag = AStarAgent()
    obs = env.unwrapped._obs_mask(False)[0]
    # battery 1, far from everything: may be unsolvable -> must not raise and must return a valid int
    a = ag.act(obs, env)
    assert a in range(7)
