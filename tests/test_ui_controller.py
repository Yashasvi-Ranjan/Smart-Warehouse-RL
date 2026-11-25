"""Headless tests of the non-UI SimulationController (no window needed)."""
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from dashboard.controller import SimulationController, algo_availability, list_models  # noqa: E402


@pytest.fixture()
def ctrl():
    return SimulationController(layout_seed=0)


@pytest.mark.parametrize("algo", ["random", "astar"])
def test_episode_metrics_match_env_stats(ctrl, algo):
    assert ctrl.set_agent(algo)
    ctrl.reset(task_seed=123)
    stats = ctrl.run_episode()
    assert ctrl.done and stats is not None
    m = ctrl.metrics()
    assert m["reward"] == pytest.approx(stats["reward"])
    assert m["steps"] == stats["steps"]
    assert m["collisions"] == stats["collisions"]
    assert m["invalid_actions"] == stats["invalid_actions"]
    assert m["energy_used"] == stats["energy_used"]
    assert ctrl.history[-1]["steps"] == stats["steps"]


def test_live_counters_agree_with_final_stats():
    c = SimulationController(layout_seed=1)
    c.set_agent("astar")
    c.reset(task_seed=5)
    live = []
    while not c.done:
        live.append((c.collisions, c.invalid))
        c.step()
    assert c.stats["success"] is True
    assert c.package_state == "delivered"
    assert c.collisions == c.stats["collisions"]


def test_reset_same_seed_is_deterministic(ctrl):
    ctrl.set_agent("astar")
    ctrl.reset(task_seed=9)
    a = ctrl.run_episode()
    ctrl.reset(task_seed=9)
    b = ctrl.run_episode()
    assert a == b


def test_configure_battery_and_reward(ctrl):
    ctrl.configure(battery=False, reward="sparse")
    assert ctrl.env_config.battery_enabled is False
    assert ctrl.env_config.reward.step == pytest.approx(-0.2)
    ctrl.set_agent("astar")
    assert ctrl.run_episode()["battery_enabled"] is False


def test_random_layout_mode(ctrl):
    ctrl.configure(layout_seed=None)
    ctrl.reset(task_seed=1)
    s1 = ctrl.env.layout_seed
    ctrl.reset(task_seed=2)
    assert ctrl.env.layout_seed != s1


def test_unavailable_agent_reports_reason(tmp_path):
    c = SimulationController(models_dir=str(tmp_path))
    assert c.set_agent("dqn") is False
    assert "unavailable" in c.agent_error
    assert c.agent_key == "random"  # previous agent kept
    av = algo_availability(str(tmp_path))
    assert av["random"][0] and not av["qlearning"][0] and av["qlearning"][1]


def test_load_saved_qlearning_model():
    models = [m for m in list_models() if m["algo"] == "qlearning"]
    if not models:
        pytest.skip("no saved Q-learning models")
    c = SimulationController(layout_seed=0)
    assert c.set_agent("qlearning", models[0]["path"]), c.agent_error
    assert c.model_meta["algo"] == "qlearning"
    stats = c.run_episode()
    assert stats["steps"] >= 1
    assert c.metrics()["reward"] == pytest.approx(stats["reward"])


def test_list_models_reads_meta():
    for m in list_models():
        assert os.path.exists(m["path"])
        assert m["algo"] in ("qlearning", "dqn", "ppo")
