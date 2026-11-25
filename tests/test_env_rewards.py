import numpy as np
import pytest

from environment import REWARD_PRESETS, RewardConfig
from environment.rewards import event_reward, potential
from conftest import DELIVER, PICKUP, RECHARGE, UP, make_env, path_actions, plan_task, reset_at


def test_preset_tables():
    d = REWARD_PRESETS["default"]
    assert (d.deliver, d.pickup, d.battery_bonus, d.step, d.collision, d.invalid, d.depleted) == (
        100, 20, 10, -0.2, -5, -2, -50)
    s = REWARD_PRESETS["sparse"]
    assert (s.pickup, s.collision, s.invalid, s.battery_bonus, s.deliver, s.depleted, s.step) == (0, 0, 0, 0, 100, -50, -0.2)
    assert REWARD_PRESETS["high_move_penalty"].step == -1.0
    assert REWARD_PRESETS["low_collision_penalty"].collision == -0.5
    assert REWARD_PRESETS["no_pickup_bonus"].pickup == 0
    assert REWARD_PRESETS["shaped"].shaping and not d.shaping
    assert event_reward(d, "collision") == pytest.approx(-5.2)
    assert event_reward(d, "deliver", True, 5) == pytest.approx(109.8)
    assert event_reward(d, "deliver", False, 5) == pytest.approx(99.8)


@pytest.mark.parametrize("name", ["sparse", "high_move_penalty", "low_collision_penalty", "no_pickup_bonus"])
def test_presets_in_env(name):
    cfg = REWARD_PRESETS[name]
    env = make_env(reward=name)
    reset_at(env, battery=60)
    e = env.unwrapped
    m = e.warehouse_map
    pk, dl = e.pickup_pos, e.delivery_pos
    reset_at(env, start=pk, battery=60)
    _, r, *_ = env.step(PICKUP)
    assert r == pytest.approx(cfg.step + cfg.pickup)
    _, r, *_ = env.step(PICKUP)
    assert r == pytest.approx(cfg.step + cfg.invalid)
    for a in path_actions(m, pk, dl):
        _, r, *_ = env.step(a)
        assert r == pytest.approx(cfg.step)
    _, r, te, *_ = env.step(DELIVER)
    assert te and r == pytest.approx(cfg.step + cfg.deliver + cfg.battery_bonus)


def test_shaping_is_potential_based_and_telescopes():
    for gamma_cfg in (None, 1.0):
        reward = "shaped" if gamma_cfg is None else {"preset": "shaped", "gamma": 1.0}
        gamma = 0.99 if gamma_cfg is None else 1.0
        sh, base = make_env(reward=reward), make_env(reward="default")
        rng = np.random.default_rng(5)
        for ep in range(25):
            sh.reset(seed=ep)
            base.reset(seed=ep)
            e, m = sh.unwrapped, sh.unwrapped.warehouse_map
            def phi():
                d = m.distance(e.robot_pos, e.delivery_pos if e.carrying else e.pickup_pos)
                leg = 0 if e.carrying else m.distance(e.pickup_pos, e.delivery_pos)
                return -(d + leg)
            assert phi() == e.potential() == potential(
                m.distance(e.robot_pos, e.delivery_pos if e.carrying else e.pickup_pos),
                m.distance(e.pickup_pos, e.delivery_pos), e.carrying)
            phi0, disc_sum, t = phi(), 0.0, 0
            # mix random actions with a plan prefix so pickups/deliveries are exercised
            plan = plan_task(sh)
            seq = plan if ep % 2 == 0 else [int(rng.integers(7)) for _ in range(80)]
            for a in seq:
                p_old = phi()
                _, rs, te, tr, _i = sh.step(a)
                _, rb, *_ = base.step(a)
                p_new = 0.0 if te else phi()
                assert rs - rb == pytest.approx(gamma * p_new - p_old, abs=1e-9)
                disc_sum += gamma ** t * (rs - rb)
                t += 1
                if te or tr:
                    break
            end = 0.0 if te else phi()
            assert disc_sum == pytest.approx(gamma ** t * end - phi0, abs=1e-8)


def test_shaping_preserves_ranking_gamma1():
    """With gamma=1 the shaped return of any terminated-by-delivery episode = base return - Phi(s0)."""
    sh = make_env(reward={"preset": "shaped", "gamma": 1.0})
    base = make_env()
    for s in range(20):
        sh.reset(seed=s); base.reset(seed=s)
        phi0 = sh.unwrapped.potential()
        plan = plan_task(sh)
        rs = sum(sh.step(a)[1] for a in plan)
        rb = sum(base.step(a)[1] for a in plan)
        assert rs == pytest.approx(rb - phi0)


def test_pickup_reward_only_once():
    env = make_env()
    reset_at(env)
    pk = env.unwrapped.pickup_pos
    reset_at(env, start=pk, battery=60)
    total = sum(env.step(PICKUP)[1] for _ in range(10))
    assert total == pytest.approx(19.8 + 9 * -2.2)
    # leaving and returning cannot re-trigger (no drop action): pickup stays invalid
    e = env.unwrapped
    assert e.carrying
    assert not env.action_masks()[4]


def test_recharge_spam_gains_nothing():
    env = make_env()
    reset_at(env)
    ch = env.unwrapped.warehouse_map.chargers[0]
    reset_at(env, start=ch, battery=5)
    e = env.unwrapped
    total = 0.0
    for _ in range(30):
        _, r, te, tr, info = env.step(RECHARGE)
        assert r <= 0 and e.battery <= 60
        total += r
    assert e.battery == 60 and total < 0
    assert info["episode_stats"]["recharges"] if (te or tr) else True
    # shaped preset: recharge/no-op never yields positive net reward over a long spam either
    sh = make_env(reward="shaped")
    reset_at(sh, start=ch, battery=5)
    tot = sum(sh.step(RECHARGE)[1] for _ in range(30))
    # shaping bonus from (gamma-1)*Phi is bounded; invalid-action penalties dominate
    assert tot < 0


def test_collision_spam_negative():
    env = make_env()
    reset_at(env, start=(1, 1), battery=60)
    assert sum(env.step(UP)[1] for _ in range(5)) == pytest.approx(5 * -5.2)
