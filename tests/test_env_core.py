import copy

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import environment  # noqa: F401  (registers the id)
from environment import EnvConfig, WarehouseEnv
from conftest import (DELIVER, DELTAS, DOWN, LEFT, PICKUP, RECHARGE, RIGHT, UP, make_env, path_actions,
                      plan_task, reset_at)

INFO_KEYS = {"action_mask", "event", "layout_seed", "robot_pos", "carrying", "battery", "steps"}


def blocked_dir(m, pos):
    for a, (dr, dc) in DELTAS.items():
        if m.is_blocked(pos[0] + dr, pos[1] + dc):
            return a
    raise AssertionError


def free_dir(m, pos):
    for a, (dr, dc) in DELTAS.items():
        if not m.is_blocked(pos[0] + dr, pos[1] + dc):
            return a


def test_check_env_and_registration():
    check_env(make_env(), skip_render_check=True)
    e = gym.make("SmartWarehouse-v0", config={"max_steps": 50}, render_mode="rgb_array")
    obs, info = e.reset(seed=1)
    assert obs.shape == (43,)
    assert e.unwrapped.config.max_steps == 50
    e.close()


def test_signatures_and_spaces(env):
    obs, info = env.reset(seed=3)
    assert env.observation_space.shape == (43,) and env.observation_space.dtype == np.float32
    assert env.action_space.n == 7
    assert obs.dtype == np.float32 and env.observation_space.contains(obs)
    assert INFO_KEYS <= set(info) and info["event"] == "reset" and info["steps"] == 0
    assert info["action_mask"].dtype == np.bool_ and info["action_mask"].shape == (7,)
    rng = np.random.default_rng(0)
    for _ in range(300):
        out = env.step(int(rng.integers(7)))
        assert len(out) == 5
        obs, r, te, tr, info = out
        assert env.observation_space.contains(obs) and isinstance(r, float)
        assert INFO_KEYS <= set(info) and isinstance(info["battery"], int)
        assert ("episode_stats" in info) == (te or tr)
        if te or tr:
            env.reset()
    f = env.unwrapped.OBS_FIELDS
    assert f["local_window"] == slice(19, 43) and f["goal_delta"] == slice(17, 19)
    assert len(env.reset(seed=1)[0][f["local_window"]]) == 24
    env24 = make_env(view_radius=3)
    assert env24.reset(seed=0)[0].shape == (19 + 48,)


def test_obs_contents(env):
    reset_at(env, start=None)
    e = env.unwrapped
    m = e.warehouse_map
    obs, _ = reset_at(env, start=e.pickup_pos)
    H1, W1 = 9, 9
    r, c = e.pickup_pos
    assert np.allclose(obs[0:2], [r / H1, c / W1])
    assert np.allclose(obs[2:4], [r / H1, c / W1])
    assert np.allclose(obs[4:6], [e.delivery_pos[0] / H1, e.delivery_pos[1] / W1])
    assert obs[6] == 0 and obs[14] == 1 and obs[15] == 0 and obs[16] == 0
    assert np.isclose(obs[7], e.battery / 60)
    near = min(range(len(m.chargers)), key=lambda k: (abs(m.chargers[k][0] - r) + abs(m.chargers[k][1] - c), k))
    assert np.allclose(obs[8:10], [m.chargers[near][0] / H1, m.chargers[near][1] / W1])
    for k, a in enumerate((UP, DOWN, LEFT, RIGHT)):
        dr, dc = DELTAS[a]
        assert obs[10 + k] == float(m.is_blocked(r + dr, c + dc))
    assert np.allclose(obs[17:19], 0)
    k = 19
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if (dr, dc) != (0, 0):
                assert obs[k] == float(m.is_blocked(r + dr, c + dc))
                k += 1
    obs2, *_ = env.step(PICKUP)
    assert obs2[6] == 1 and obs2[14] == 0
    d = e.delivery_pos
    assert np.allclose(obs2[17:19], [(d[0] - r) / H1, (d[1] - c) / W1])
    # discrete state
    ds = e.discrete_state()
    assert len(ds) == 12 and ds[:8] == (r, c, r, c, d[0], d[1], 1, min(5, int(e.battery / 60 * 6)))
    assert ds[8:] == tuple(int(x) for x in obs2[10:14])
    assert e.discrete_state(B=3)[7] == min(2, int(e.battery / 60 * 3))
    # charger obs
    reset_at(env, start=m.chargers[0])
    assert env.unwrapped.discrete_state()[:2] == m.chargers[0]


def test_moves_and_collision(env):
    reset_at(env, start=(1, 1), battery=40)
    e = env.unwrapped
    m = e.warehouse_map
    # (1,1) has wall above and left
    for a in (UP, LEFT):
        obs, r, te, tr, info = env.step(a)
        assert e.robot_pos == (1, 1) and e.battery == 40
        assert info["event"] == "collision" and r == pytest.approx(-0.2 - 5) and not te
        assert not info["action_mask"][a]
    a = RIGHT if not m.is_blocked(1, 2) else DOWN
    obs, r, te, tr, info = env.step(a)
    assert e.robot_pos != (1, 1) and e.battery == 39 and info["event"] == "move"
    assert r == pytest.approx(-0.2)
    # every move direction
    for a in (UP, DOWN, LEFT, RIGHT):
        reset_at(env, start=(4, 4) if not m.is_blocked(4, 4) else (4, 1), battery=40)
        p = e.robot_pos
        dr, dc = DELTAS[a]
        env.step(a)
        expect = (p[0] + dr, p[1] + dc) if not m.is_blocked(p[0] + dr, p[1] + dc) else p
        assert e.robot_pos == expect


def test_pickup_rules(env):
    reset_at(env)
    e = env.unwrapped
    m = e.warehouse_map
    # not on pickup -> invalid
    off = next(p for p in m.start_cells if p != e.pickup_pos)
    reset_at(env, start=off)
    _, r, te, tr, info = env.step(PICKUP)
    assert info["event"] == "invalid_pickup" and r == pytest.approx(-2.2) and not e.carrying
    # on an inactive pickup station -> invalid too
    other = next(p for p in m.pickup_stations if p != e.pickup_pos)
    reset_at(env, start=other)
    assert env.step(PICKUP)[4]["event"] == "invalid_pickup"
    # on active pickup: success once only
    reset_at(env, start=e.pickup_pos)
    _, r, te, tr, info = env.step(PICKUP)
    assert info["event"] == "pickup" and r == pytest.approx(19.8) and e.carrying and not te
    _, r, te, tr, info = env.step(PICKUP)
    assert info["event"] == "invalid_pickup" and r == pytest.approx(-2.2)


def test_deliver_terminates_with_reward(env):
    reset_at(env, battery=60)
    e = env.unwrapped
    m = e.warehouse_map
    reset_at(env, start=e.pickup_pos, battery=60)
    # deliver without carrying at delivery cell is invalid; deliver carrying elsewhere invalid
    assert env.step(DELIVER)[4]["event"] == "invalid_deliver"
    env.step(PICKUP)
    assert env.step(DELIVER)[4]["event"] == "invalid_deliver"
    total = 0.0
    acts = path_actions(m, e.pickup_pos, e.delivery_pos)
    for a in acts:
        _, r, te, tr, info = env.step(a)
        assert not te and info["event"] == "move"
    assert e.robot_pos == e.delivery_pos
    _, r, te, tr, info = env.step(DELIVER)
    assert te and not tr and info["event"] == "deliver"
    assert r == pytest.approx(-0.2 + 100 + 10)
    s = info["episode_stats"]
    assert s["success"] and s["picked_up"] and not s["battery_depleted"] and not s["timeout"]
    with pytest.raises(RuntimeError):
        env.step(UP)
    env.reset()
    env.step(UP)  # fine after reset


def test_battery_depletion():
    env = make_env()
    reset_at(env, start=(4, 1), battery=1)
    e = env.unwrapped
    m = e.warehouse_map
    # battery 1 -> move onto non-charger leaves 0 -> depleted
    a = free_dir(m, (4, 1))
    tgt = (4 + DELTAS[a][0], 1 + DELTAS[a][1])
    assert tgt not in m.chargers
    _, r, te, tr, info = env.step(a)
    assert te and info["event"] == "depleted" and r == pytest.approx(-50.2)
    assert info["episode_stats"]["battery_depleted"] and not info["episode_stats"]["success"]
    assert info["action_mask"].any()
    # move attempted at battery 0 (off charger) -> depleted without moving
    reset_at(env, start=(4, 1), battery=0)
    _, r, te, tr, info = env.step(free_dir(m, (4, 1)))
    assert te and info["event"] == "depleted" and e.robot_pos == (4, 1) and r == pytest.approx(-50.2)
    # battery 1, move onto a charger -> alive with battery 0, can recharge
    ch = m.chargers[0]
    nb = next(((ch[0] + DELTAS[a][0], ch[1] + DELTAS[a][1], a) for a in DELTAS
               if not m.is_blocked(ch[0] + DELTAS[a][0], ch[1] + DELTAS[a][1])))
    back = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}[nb[2]]
    reset_at(env, start=(nb[0], nb[1]), battery=1)
    _, r, te, tr, info = env.step(back)
    assert not te and e.battery == 0 and e.robot_pos == ch and info["event"] == "move"
    assert info["action_mask"].tolist() == [False] * 4 + [False, False, True]
    _, r, te, tr, info = env.step(RECHARGE)
    assert e.battery == 20 and info["event"] == "recharge"


def test_recharge_rules(env):
    reset_at(env, start=None, battery=10)
    e = env.unwrapped
    m = e.warehouse_map
    reset_at(env, start=m.chargers[0], battery=10)
    _, r, te, tr, info = env.step(RECHARGE)
    assert e.battery == 30 and info["event"] == "recharge" and r == pytest.approx(-0.2)
    for _ in range(2):
        env.step(RECHARGE)
    assert e.battery == 60  # clipped at capacity
    _, r, te, tr, info = env.step(RECHARGE)
    assert info["event"] == "invalid_recharge" and r == pytest.approx(-2.2) and e.battery == 60
    off = next(p for p in m.start_cells)
    reset_at(env, start=off, battery=10)
    _, r, te, tr, info = env.step(RECHARGE)
    assert info["event"] == "invalid_recharge" and e.battery == 10 and r == pytest.approx(-2.2)
    assert info["episode_stats"]["recharges"] if te else True


def test_battery_disabled():
    env = make_env(battery_enabled=False)
    obs, info = env.reset(seed=2)
    e = env.unwrapped
    assert obs[7] == 1.0 and e.discrete_state()[7] == 0
    m = e.warehouse_map
    reset_at(env, start=m.chargers[0])
    assert env.step(RECHARGE)[4]["event"] == "invalid_recharge"
    assert not env.action_masks()[6]
    b0 = e.battery
    reset_at(env, start=(4, 1))
    for i in range(60):
        _, r, te, tr, info = env.step(UP if i % 2 else DOWN)
        assert not te and info["battery"] == b0 and info["event"] != "depleted"
        assert env.unwrapped.observation_space.contains(_) and _[7] == 1.0
    _, r, te, tr, info = env.step(PICKUP)
    assert info["event"] == "invalid_pickup"
    reset_at(env)
    m = e.warehouse_map
    p = path_actions(m, e.robot_pos, e.pickup_pos) + [PICKUP]
    obs_r = 0.0
    p += path_actions(m, e.pickup_pos, e.delivery_pos) + [DELIVER]
    for a in p:
        _, r, te, tr, info = env.step(a)
        obs_r += r
    assert te and info["episode_stats"]["success"] and info["episode_stats"]["battery_enabled"] is False
    assert r == pytest.approx(-0.2 + 100)  # no battery bonus when disabled


def test_truncation_vs_termination():
    env = make_env(max_steps=5)
    reset_at(env, start=(4, 1))
    for i in range(4):
        _, r, te, tr, info = env.step(PICKUP)
        assert not te and not tr and "episode_stats" not in info
    _, r, te, tr, info = env.step(PICKUP)
    assert tr and not te and info["event"] == "timeout" and info["action_event"] == "invalid_pickup"
    assert info["episode_stats"]["timeout"] and not info["episode_stats"]["success"]
    assert info["episode_stats"]["steps"] == 5
    with pytest.raises(RuntimeError):
        env.step(0)
    # terminating on the last step is termination, not truncation
    reset_at(env, start=env.unwrapped.pickup_pos)
    e = env.unwrapped
    e2 = make_env(max_steps=2)
    reset_at(e2, start=e.pickup_pos, delivery=e.pickup_pos)
    e2.step(PICKUP)
    _, r, te, tr, info = e2.step(DELIVER)
    assert te and not tr and info["episode_stats"]["success"]


def test_step_before_reset_and_bad_action(env):
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=0)
    for bad in (-1, 7):
        with pytest.raises(ValueError):
            env.step(bad)
    env.step(np.int64(3))


def test_determinism():
    def run(seed):
        env = make_env()
        obs, info = env.reset(seed=seed)
        rng = np.random.default_rng(99)
        trace = [obs.copy()]
        e = env.unwrapped
        head = (e.warehouse_map.signature(), e.robot_pos, e.pickup_pos, e.delivery_pos, e.battery)
        for _ in range(150):
            obs, r, te, tr, info = env.step(int(rng.integers(7)))
            trace.append((obs.copy(), r, te, tr, info["event"]))
            if te or tr:
                break
        return head, trace
    h1, t1 = run(11)
    h2, t2 = run(11)
    assert h1 == h2 and len(t1) == len(t2)
    for a, b in zip(t1[1:], t2[1:]):
        assert np.array_equal(a[0], b[0]) and a[1:] == b[1:]
    # re-reset the same env with the same seed
    env = make_env()
    env.reset(seed=11)
    sig = env.unwrapped.warehouse_map.signature()
    env.step(0)
    env.reset(seed=11)
    assert env.unwrapped.warehouse_map.signature() == sig == h1[0]
    assert env.unwrapped.robot_pos == h1[1]
    heads = {run(s)[0][0] for s in range(10)}
    assert len(heads) >= 9  # different seeds -> different layouts


def test_layout_seed_pool_and_options():
    env = make_env(layout_seeds=[3, 4])
    seen = set()
    for s in range(40):
        _, info = env.reset(seed=s)
        seen.add(info["layout_seed"])
        assert env.unwrapped.warehouse_map.layout_seed == info["layout_seed"]
    assert seen == {3, 4}
    _, info = env.reset(seed=0, options={"layout_seed": 9})
    assert info["layout_seed"] == 9
    _, info2 = env.reset(seed=0, options={"layout_seed": 9})
    assert env.unwrapped.warehouse_map is not None and info2["layout_seed"] == 9
    # random layout mode: many distinct layout seeds; same reset seed -> same layout seed
    r = make_env()
    ls = {r.reset(seed=s)[1]["layout_seed"] for s in range(30)}
    assert len(ls) > 25
    assert r.reset(seed=5)[1]["layout_seed"] == r.reset(seed=5)[1]["layout_seed"]
    # same layout + same seed -> same task regardless of mode (determinism of task)
    a = make_env(layout_seeds=[7]); b = make_env(layout_seeds=[7])
    a.reset(seed=4); b.reset(seed=4)
    assert a.unwrapped.robot_pos == b.unwrapped.robot_pos


def test_action_masks_correct():
    env = make_env(reward="default")
    rng = np.random.default_rng(1)
    checked = 0
    for ep in range(6):
        env.reset(seed=ep, options={"battery": 8, "start_pos": None} if False else None)
        for t in range(120):
            mask = env.action_masks()
            assert mask.shape == (7,) and mask.dtype == np.bool_ and mask.any()
            for a in range(7):
                twin = copy.deepcopy(env)
                b0 = twin.unwrapped.battery
                _, r, te, tr, info = twin.step(a)
                ev = info["action_event"]
                # a legal move that drains the last battery unit onto a non-charger is "valid" per the contract
                bad = ev in ("collision", "invalid_pickup", "invalid_deliver", "invalid_recharge") or (
                    ev == "depleted" and b0 == 0)
                assert mask[a] == (not bad), (a, ev, mask)
                checked += 1
            _, r, te, tr, info = env.step(int(rng.choice(np.flatnonzero(mask)) if rng.random() < 0.9
                                              else rng.integers(7)))
            assert np.array_equal(info["action_mask"], env.action_masks())
            if te or tr:
                break
    assert checked > 100


def test_episode_stats_hand_trajectory():
    env = make_env()
    reset_at(env, battery=60)
    e = env.unwrapped
    m = e.warehouse_map
    pk, dl = e.pickup_pos, e.delivery_pos
    reset_at(env, start=pk, battery=60)
    L = len(path_actions(m, pk, dl))
    wall = blocked_dir(m, pk)
    seq = [PICKUP, PICKUP, RECHARGE, wall] + path_actions(m, pk, dl) + [DELIVER]
    total = 0.0
    for a in seq:
        _, r, te, tr, info = env.step(a)
        total += r
    s = info["episode_stats"]
    n = len(seq)
    expect = n * -0.2 + 20 - 2 * 2 - 5 + 100 + 10
    assert te and s["success"] and s["picked_up"]
    assert s["steps"] == n == L + 5 and s["collisions"] == 1 and s["invalid_actions"] == 2
    assert s["energy_used"] == L and s["recharges"] == 0 and not s["battery_depleted"] and not s["timeout"]
    assert s["reward"] == pytest.approx(expect) and total == pytest.approx(expect)
    assert s["optimal_steps"] == 0 + 1 + L + 1
    assert s["layout_seed"] == 0 and s["battery_enabled"] is True
    assert set(s) == {"reward", "steps", "success", "picked_up", "collisions", "invalid_actions", "energy_used",
                      "recharges", "battery_depleted", "timeout", "optimal_steps", "layout_seed",
                      "battery_enabled"}


def test_optimal_scripted_policy_solves_and_return(capsys):
    """BFS-optimal scripted policy (only public helpers) solves every episode; report typical return."""
    for cfg_over in ({}, {"battery_enabled": False}, {"battery_capacity": 24, "battery_init_range": (0.4, 0.6)}):
        env = make_env(**cfg_over)
        rets, lens, gaps = [], [], []
        for s in range(200):
            env.reset(seed=1000 + s)
            plan = plan_task(env)
            for a in plan:
                _, r, te, tr, info = env.step(a)
            st = info["episode_stats"]
            assert te and st["success"] and st["collisions"] == 0 and st["invalid_actions"] == 0
            rets.append(st["reward"]); lens.append(st["steps"])
            gaps.append(st["steps"] - st["optimal_steps"])
        print(f"\n[optimal policy {cfg_over or 'default'}] mean return {np.mean(rets):.2f} "
              f"(min {min(rets):.2f}, max {max(rets):.2f}) mean steps {np.mean(lens):.1f} "
              f"mean extra steps over battery-free optimum {np.mean(gaps):.2f}")
        if not cfg_over:
            assert 100 < np.mean(rets) < 135


def test_render_rgb_array_headless():
    env = make_env(render_mode="rgb_array")
    env.reset(seed=0)
    img = env.render()
    assert img.shape == (480, 480, 3) and img.dtype == np.uint8
    assert img.std() > 5
    reset_at(env, start=env.unwrapped.warehouse_map.pickup_stations[0])
    env.step(PICKUP)
    img2 = env.render()
    assert img2.shape == img.shape and not np.array_equal(img, img2)
    st = env.unwrapped.get_render_state()
    assert st["carrying"] and st["robot_pos"] == env.unwrapped.robot_pos and st["grid"].shape == (10, 10)
    env.close()
    assert make_env().render() is None


def test_render_human_headless():
    env = make_env(render_mode="human")
    env.reset(seed=0)
    env.step(0)
    env.close()


def test_throughput_floor():
    import time
    env = make_env()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    acts = rng.integers(0, 7, 20000).tolist()
    t = time.perf_counter()
    for a in acts:
        _, _, te, tr, _ = env.step(a)
        if te or tr:
            env.reset()
    sps = len(acts) / (time.perf_counter() - t)
    assert sps > 20000, sps
