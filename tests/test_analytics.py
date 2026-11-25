"""Analytics tests. Frames marked FIXTURE are synthetic and exist only inside tests."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from agents.base import Agent
from analytics import export, plots
from analytics.comparison import (battery_comparison, compare_agents, compare_runs, generalization_table,
                                  paired_comparison, reward_config_comparison)
from analytics.metrics import (EPISODE_COLUMNS, aggregate_over_seeds, bootstrap_ci, evaluate_agent,
                               evaluate_generalization, load_train_log, moving_average, run_episode,
                               summarize, wilson_interval)
from environment import EnvConfig, WarehouseEnv


class StubAgent(Agent):
    """Test-only deterministic agent: always tries to move right, then up."""
    name = "stub"

    def __init__(self):
        self.resets = 0
        self.envs = 0

    def reset(self):
        self.resets += 1

    def act(self, obs, env=None, deterministic=True):
        self.envs += env is not None
        return 3 if obs[13] < 0.5 else 0


def _cfg(**kw):
    return EnvConfig(max_steps=30, **kw)


def fixture_df():
    """FIXTURE: hand-built episode frame with known aggregates."""
    return pd.DataFrame({
        "agent": "x", "label": "x", "seed": [0, 1, 2, 3], "layout_seed": 0,
        "reward": [100.0, 50.0, -10.0, 0.0], "success": [True, True, False, False],
        "steps": [10, 20, 30, 30], "optimal_steps": [8.0, 10.0, 8.0, 9.0],
        "collisions": [0, 2, 4, 6], "invalid_actions": [0, 0, 1, 1], "energy_used": [10, 30, 25, 5],
        "recharges": 0, "picked_up": True, "battery_depleted": [False, False, True, False],
        "timeout": [False, False, False, True]})


def fixture_log(n=60, seed=0):
    """FIXTURE: synthetic training log for plot smoke tests."""
    r = np.random.RandomState(seed)
    return pd.DataFrame({"episode": np.arange(n), "timestep": np.arange(n) * 20, "reward": np.linspace(-50, 80, n) + r.randn(n),
                         "steps": 50, "success": (r.rand(n) < np.linspace(0, 1, n)).astype(int),
                         "collisions": 1, "energy_used": 20, "epsilon": np.linspace(1, 0.05, n),
                         "loss": np.where(np.arange(n) < 10, np.nan, 0.5)})


def test_evaluate_agent_columns_and_determinism():
    a = StubAgent()
    d1 = evaluate_agent(a, _cfg(), range(5), layout_seeds=[0, 1])
    d2 = evaluate_agent(StubAgent(), _cfg(), range(5), layout_seeds=[0, 1])
    assert list(d1.columns) == EPISODE_COLUMNS
    assert len(d1) == 5 and a.resets == 5 and a.envs > 0
    pd.testing.assert_frame_equal(d1, d2)
    assert set(d1["layout_seed"]) <= {0, 1}
    assert evaluate_agent(a, _cfg(), [0], label="L")["label"].iloc[0] == "L"


def test_run_episode_trajectory():
    env = WarehouseEnv(_cfg())
    st, traj = run_episode(StubAgent(), env, 3, record_trajectory=True)
    assert traj[0]["t"] == -1 and len(traj) == st["steps"] + 1
    st2, none = run_episode(StubAgent(), env, 3)
    assert none is None and st2 == st


def test_summarize_hand_computed():
    s = summarize(fixture_df())
    assert s["n"] == 4 and s["n_success"] == 2
    assert s["reward_mean"] == pytest.approx(35.0)
    assert s["reward_std"] == pytest.approx(np.std([100, 50, -10, 0], ddof=1))
    assert s["success_rate"] == 0.5
    assert s["steps_per_delivery"] == 15.0
    assert s["collisions_mean"] == 3.0
    assert s["energy_per_delivery"] == 20.0
    assert s["path_efficiency"] == pytest.approx((0.8 + 0.5) / 2)
    assert s["battery_depleted_rate"] == 0.25 and s["timeout_rate"] == 0.25
    assert 0 < s["success_ci_low"] < 0.5 < s["success_ci_high"] < 1


def test_summarize_empty_and_no_success():
    s = summarize(fixture_df().iloc[0:0])
    assert s["n"] == 0 and np.isnan(s["reward_mean"])
    s = summarize(fixture_df().iloc[2:])
    assert s["success_rate"] == 0 and np.isnan(s["steps_per_delivery"])


def test_aggregate_over_seeds():
    a = fixture_df()
    b = fixture_df().copy()
    b["reward"] = b["reward"] + 10
    agg = aggregate_over_seeds([a, summarize(b)])
    assert agg["n_runs"] == 2 and agg["reward_mean"] == pytest.approx(40.0)
    assert agg["reward_mean_std_runs"] == pytest.approx(np.std([35, 45], ddof=1))
    assert aggregate_over_seeds([a])["reward_mean_std_runs"] == 0.0
    with pytest.raises(ValueError):
        aggregate_over_seeds([])


def test_helpers():
    assert moving_average([1, 2, 3, 4], 2).tolist() == [1, 1.5, 2.5, 3.5]
    assert moving_average([], 5).size == 0
    lo, hi = wilson_interval(5, 10)
    assert lo < 0.5 < hi and np.isnan(wilson_interval(0, 0)[0])
    lo, hi = bootstrap_ci(np.arange(100.0))
    assert 40 < lo < 49.5 < hi < 60


def test_compare_and_generalization():
    agents = {"a": StubAgent(), "b": StubAgent()}
    long_df, summ = compare_agents(agents, _cfg(), range(4), layout_seeds=[0])
    assert list(summ.index) == ["a", "b"] and len(long_df) == 8
    ra, rb = (long_df[long_df.label == k].reset_index(drop=True) for k in "ab")
    pd.testing.assert_frame_equal(ra.drop(columns="label"), rb.drop(columns="label"))  # identical seeds => identical
    out = evaluate_generalization(StubAgent(), _cfg(), [0], [1000, 1001], range(3))
    assert set(out["summary"]) == {"train", "test", "gap"} and len(out["test"]) == 3
    _, table = generalization_table(agents, _cfg(), [0], [1000], range(3))
    assert set(table["split"]) == {"train", "test", "gap"}
    fig = plots.plot_generalization(table)
    assert fig is not None


def test_compare_runs_and_other_helpers():
    long_df, summ = compare_runs({"a": {0: StubAgent(), 1: StubAgent()}}, _cfg(), range(3), [0])
    assert summ.loc["a", "n_runs"] == 2 and set(long_df.train_seed) == {0, 1}
    _, s = reward_config_comparison({"sparse": StubAgent(), "default": StubAgent()}, _cfg(), range(3), layout_seeds=[0])
    assert list(s.index) == ["sparse", "default"]
    _, s = battery_comparison({"on": (StubAgent(), _cfg()), "off": (StubAgent(), _cfg(battery_enabled=False))}, range(3), [0])
    assert set(s.index) == {"on", "off"}


def test_paired_comparison():
    a, b = fixture_df(), fixture_df().copy()
    b["reward"] = b["reward"] - 5
    r = paired_comparison(a, b)
    assert r["mean_diff"] == 5 and r["wins_a"] == 4 and r["p_value"] <= 0.2
    with pytest.raises(ValueError):
        paired_comparison(a, b.assign(seed=b.seed + 100))


def test_plots_return_figures(tmp_path):
    from matplotlib.figure import Figure
    logs = {"qlearning": [fixture_log(seed=0), fixture_log(seed=1)], "dqn": fixture_log(seed=2)}
    figs = [plots.plot_training_curves(logs, window=10)]
    s = pd.DataFrame([summarize(fixture_df()), summarize(fixture_df())], index=["random", "ppo"])
    figs.append(plots.plot_comparison_bars(s))
    figs.append(plots.plot_comparison_bars(pd.DataFrame([aggregate_over_seeds([fixture_df(), fixture_df()])], index=["dqn"])))
    figs.append(plots.plot_episode_length_distribution(pd.concat([fixture_df().assign(label="a"), fixture_df().assign(label="b")])))
    for f in figs:
        assert isinstance(f, Figure)
    p = plots.save_figure(figs[0], str(tmp_path / "sub" / "c.png"))
    assert os.path.getsize(p) > 1000
    html = plots.plot_interactive_training(logs, 10, path=str(tmp_path / "c.html"))
    assert os.path.exists(tmp_path / "c.html") and html is not None


def test_plots_empty_raise():
    with pytest.raises(ValueError):
        plots.plot_training_curves({})
    with pytest.raises(ValueError):
        plots.plot_comparison_bars(pd.DataFrame())
    with pytest.raises(ValueError):
        plots.plot_episode_length_distribution(pd.DataFrame())


def test_export_roundtrip_and_empty(tmp_path):
    df = fixture_df()
    p = export.export_results(df, str(tmp_path / "e.csv"))
    pd.testing.assert_frame_equal(pd.read_csv(p)[["seed", "reward"]], df[["seed", "reward"]])
    sp = export.export_summary(pd.DataFrame([summarize(df)], index=["x"]), str(tmp_path / "s.json"))
    assert json.load(open(sp))[0]["n"] == 4
    export.export_summary(summarize(df), str(tmp_path / "s.csv"))
    export.export_run_metadata({"a": np.float64(1.0)}, str(tmp_path / "m.json"))
    assert json.load(open(tmp_path / "m.json"))["a"] == 1.0
    with pytest.raises(ValueError):
        export.export_results(df.iloc[0:0], str(tmp_path / "x.csv"))


def test_discover_runs_and_tensorboard(tmp_path):
    res, mod = tmp_path / "results", tmp_path / "models"
    assert export.discover_runs(str(res), str(mod)).empty
    (res / "run1").mkdir(parents=True)
    (mod / "qlearning").mkdir(parents=True)
    fixture_log(20).to_csv(res / "run1" / "train_log.csv", index=False)
    (mod / "qlearning" / "run1.npz").write_bytes(b"x")
    (mod / "qlearning" / "run1.meta.json").write_text(json.dumps(
        {"algo": "qlearning", "run_name": "run1", "reward": {"preset": "default"}, "env_config": {"battery_enabled": True},
         "train": {"seed": 3, "episodes": 20, "wall_seconds": 1.5}, "created": "2026-01-01"}))
    idx = export.discover_runs(str(res), str(mod))
    assert len(idx) == 1 and idx.loc[0, "algo"] == "qlearning" and idx.loc[0, "seed"] == 3
    assert idx.loc[0, "final_reward_ma"] == pytest.approx(fixture_log(20)["reward"].mean())
    assert load_train_log("run1", results_dir=str(res)).shape[0] == 20
    with pytest.raises(FileNotFoundError):
        load_train_log("nope", results_dir=str(res))
    out = export.write_tensorboard(str(res / "run1" / "train_log.csv"), str(tmp_path / "tb"))
    assert any(f.startswith("events.out.tfevents") for f in os.listdir(out))
