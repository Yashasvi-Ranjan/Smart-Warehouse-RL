import numpy as np
import pytest

from environment import EnvConfig, WarehouseEnv, generate_layout
from environment.warehouse_map import FLOOR, SHELF, WALL
from conftest import plan_task, make_env

CFGS = [EnvConfig(), EnvConfig(grid_h=7, grid_w=8, battery_capacity=40, n_pickup_stations=2,
                               n_delivery_stations=2), EnvConfig(grid_h=16, grid_w=20, battery_capacity=80,
                                                                  n_pickup_stations=6, n_delivery_stations=4)]


@pytest.mark.parametrize("cfg", CFGS)
def test_layout_structure(cfg):
    for seed in range(60):
        m = generate_layout(cfg, seed)
        g = m.grid
        assert g.shape == (cfg.grid_h, cfg.grid_w)
        assert (g[0] == WALL).all() and (g[-1] == WALL).all() and (g[:, 0] == WALL).all() and (g[:, -1] == WALL).all()
        assert (g == SHELF).sum() >= 6
        assert m.is_connected()
        assert len(m.pickup_stations) == cfg.n_pickup_stations
        assert len(m.delivery_stations) == cfg.n_delivery_stations
        assert len(m.chargers) == cfg.n_chargers
        cells = m.pickup_stations + m.delivery_stations + m.chargers
        assert len(set(cells)) == len(cells)
        for r, c in cells:
            assert g[r, c] == FLOOR
        for r, c in m.pickup_stations:
            assert any(g[r + dr, c + dc] == SHELF for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))


def test_same_seed_same_layout_different_seeds_differ():
    cfg = EnvConfig()
    a, b = generate_layout(cfg, 5), generate_layout(cfg, 5)
    assert a.signature() == b.signature()
    sigs = {generate_layout(cfg, s).signature() for s in range(100)}
    assert len(sigs) >= 95
    grids = {generate_layout(cfg, s).grid.tobytes() for s in range(100)}
    assert len(grids) >= 20  # shelf arrangement itself varies, not just stations
    assert generate_layout(cfg, 0).signature() != generate_layout(cfg, 1).signature()
    with pytest.raises(ValueError):
        generate_layout(cfg, -1)


def test_300_seeds_connected_and_tasks_solvable():
    """Random-layout mode: every reset yields a connected layout and a task with a valid plan."""
    env = make_env()
    for s in range(300):
        env.reset(seed=s)
        e = env.unwrapped
        assert e.warehouse_map.is_connected()
        m = e.warehouse_map
        assert m.min_task_steps(e.robot_pos, e.pickup_pos, e.delivery_pos, e.battery,
                                e.config.battery_capacity, e.config.recharge_amount) <= e.config.max_steps
        assert len(plan_task(env)) <= e.config.max_steps
        assert e.robot_pos not in m.pickup_stations + m.delivery_stations + m.chargers


def test_min_task_steps_matches_independent_planner_with_recharge():
    # tight batteries force charger detours
    cfg = dict(battery_capacity=22, battery_init_range=(0.3, 0.5), n_chargers=2, max_steps=200)
    env = make_env(**cfg)
    detours = 0
    for s in range(120):
        env.reset(seed=s)
        e = env.unwrapped
        plan = plan_task(env)
        n = e.warehouse_map.min_task_steps(e.robot_pos, e.pickup_pos, e.delivery_pos, e.battery, 22, 20)
        assert n == len(plan)
        if 6 in plan:
            detours += 1
        # execute the plan
        total = 0.0
        for a in plan:
            _, r, te, tr, info = env.step(a)
            total += r
        assert te and info["episode_stats"]["success"]
        assert info["episode_stats"]["steps"] == n
    assert detours > 0  # the recharge path of the solver is exercised


def test_unsolvable_task_resampled_or_raises():
    env = make_env(max_steps=3)
    with pytest.raises(RuntimeError):
        env.reset(seed=0)
