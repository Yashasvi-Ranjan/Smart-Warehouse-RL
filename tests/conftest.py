"""Shared pytest setup/helpers for the environment tests (headless, repo root on sys.path)."""
import os
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from environment import EnvConfig, WarehouseEnv

UP, DOWN, LEFT, RIGHT, PICKUP, DELIVER, RECHARGE = range(7)
DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


def make_env(layout_seed=0, render_mode=None, **over):
    cfg = EnvConfig(**over)
    return WarehouseEnv(cfg, render_mode=render_mode)


def reset_at(env, start=None, pickup=None, delivery=None, battery=None, layout_seed=0, seed=0):
    """Reset with a forced task (defaults: first start cell / first pickup / first delivery of the layout)."""
    from environment.warehouse_map import get_layout
    m = get_layout(env.unwrapped.config, layout_seed)
    opts = {"layout_seed": layout_seed,
            "start_pos": start if start is not None else m.start_cells[0],
            "pickup_pos": pickup if pickup is not None else m.pickup_stations[0],
            "delivery_pos": delivery if delivery is not None else m.delivery_stations[0]}
    if battery is not None:
        opts["battery"] = battery
    return env.reset(seed=seed, options=opts)


def path_actions(m, a, b):
    """Action list of a BFS shortest path from cell a to cell b (uses env public map helpers)."""
    path = m.shortest_path(a, b)
    out = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        out.append({v: k for k, v in DELTAS.items()}[(r1 - r0, c1 - c0)])
    return out


def plan_task(env):
    """Optimal action list for the current task under the battery rules (BFS on (cell, battery, phase)).
    Independent implementation from WarehouseMap.min_task_steps (used to cross-check it)."""
    e = env.unwrapped
    m, cfg = e.warehouse_map, e.config
    s = e.robot_pos
    start = (s, e.battery, 1 if e.carrying else 0)
    parent = {start: None}
    q = deque([start])
    goal = None
    while q:
        st = q.popleft()
        (r, c), b, ph = st
        if ph == 0 and (r, c) == e.pickup_pos:
            nxt = [((r, c), b, 1, PICKUP)]
        else:
            nxt = []
        if ph == 1 and (r, c) == e.delivery_pos:
            goal = st
            break
        for a, (dr, dc) in DELTAS.items():
            rr, cc = r + dr, c + dc
            if m.is_blocked(rr, cc):
                continue
            nb = b - 1 if cfg.battery_enabled else b
            if cfg.battery_enabled and (b < 1 or (nb == 0 and (rr, cc) not in m.chargers)):
                continue
            nxt.append(((rr, cc), nb, ph, a))
        if cfg.battery_enabled and (r, c) in m.chargers and b < cfg.battery_capacity:
            nxt.append(((r, c), min(cfg.battery_capacity, b + cfg.recharge_amount), ph, RECHARGE))
        for cell, nb, nph, a in nxt:
            ns = (cell, nb, nph)
            if ns not in parent:
                parent[ns] = (st, a)
                q.append(ns)
    assert goal is not None, "task unsolvable"
    acts = [DELIVER]
    st = goal
    while parent[st] is not None:
        st, a = parent[st]
        acts.append(a)
    return acts[::-1]


@pytest.fixture
def env():
    e = make_env()
    yield e
    e.close()
