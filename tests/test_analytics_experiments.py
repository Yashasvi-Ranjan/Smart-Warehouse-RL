"""Smoke test of the experiment orchestrator with a tiny budget in a temp dir."""
import importlib.util
import os

import pandas as pd
import pytest

from experiments import run_experiments as rx

pytestmark = pytest.mark.skipif(importlib.util.find_spec("training.train_qlearning") is None,
                                reason="Q-learning trainer not available")


def test_run_name_sharing():
    class A:  # minimal args stand-in
        quick = False; algos = None; seeds = 1; eval_episodes = 3; episodes_scale = 1.0
        models_dir = "m"; results_dir = "r"
    from environment import load_config
    S = rx.Settings(1, A(), load_config("config.yaml"))
    assert S.run_name("dqn", 2) == "dqn_main_s2"
    assert S.run_name("dqn", 2, battery=False) == "dqn_nobat_s2"
    assert S.run_name("dqn", 0, preset="shaped") == "dqn_rw-shaped_s0"


def test_exp1_and_exp4_tiny(tmp_path):
    base = ["--quick", "--seeds", "1", "--eval-episodes", "4", "--episodes-scale", "0.02", "--algos", "random", "qlearning",
            "--models-dir", str(tmp_path / "models"), "--results-dir", str(tmp_path / "res")]
    assert rx.main(["--exp", "1"] + base) == 0
    out = tmp_path / "res" / "exp1_quick"
    ep = pd.read_csv(out / "episodes.csv")
    assert set(ep["label"]) == {"random", "qlearning"}
    assert (ep.groupby("label")["seed"].apply(lambda s: sorted(s)).apply(tuple).nunique()) == 1  # identical eval seeds
    assert (out / "summary.csv").exists() and (out / "summary.json").exists() and (out / "comparison_bars.png").exists()
    n_models = len(os.listdir(tmp_path / "models" / "qlearning"))
    assert rx.main(["--exp", "4"] + base) == 0
    assert (tmp_path / "res" / "exp4_quick" / "summary.csv").exists()
    # the with-battery model was reused (cached): only the no-battery run (+meta) was added
    assert len(os.listdir(tmp_path / "models" / "qlearning")) == n_models + 4  # + nobat and tightbat (model+meta)
    s = pd.read_csv(tmp_path / "res" / "exp4_quick" / "summary.csv")
    assert "tight_battery" in set(s["condition"]) and "recharges_mean" in s.columns
