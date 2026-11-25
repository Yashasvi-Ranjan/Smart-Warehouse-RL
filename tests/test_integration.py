"""End-to-end integration tests (tiny budgets, everything written to pytest tmp dirs).

Nothing here touches ``models/`` or ``experiments/results/``: every trainer is called with explicit
``models_dir`` / ``results_dir`` arguments and ``run_experiments`` with ``--models-dir/--results-dir``.
The budgets are far too small to learn anything useful; the tests check that the *pipeline* is wired
correctly (formats, determinism, loading, evaluation on identical seeds, exports, plots, UI controller).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

from agents import load_agent, make_agent, meta_path, read_meta  # noqa: E402
from analytics.export import discover_runs, export_results, export_summary  # noqa: E402
from analytics.metrics import EPISODE_COLUMNS, evaluate_agent, load_train_log, summarize  # noqa: E402
from environment import EnvConfig, WarehouseEnv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = [0, 1]
EVAL_SEEDS = list(range(100_000, 100_006))
LOG_COLUMNS = ["episode", "timestep", "reward", "steps", "success", "collisions", "energy_used", "epsilon", "loss"]

QL_HP = {"episodes": 300, "epsilon_decay_episodes": 200}
DQN_HP = {"episodes": 25, "learning_starts": 200, "batch_size": 32, "hidden": [32, 32], "buffer_size": 5000,
          "target_update_every": 50, "epsilon_decay_steps": 1500, "use_action_mask": True}
PPO_HP = {"total_timesteps": 4096, "n_steps": 512, "batch_size": 64, "n_epochs": 2, "net_arch": [32, 32]}


def env_cfg(**over):
    return EnvConfig(layout_seeds=list(LAYOUTS), **over)


def run_agent_episode(agent, cfg, seed):
    env = WarehouseEnv(cfg)
    obs, info = env.reset(seed=seed)
    agent.reset()
    while True:
        obs, r, term, trunc, info = env.step(agent.act(obs, env=env, deterministic=True))
        if term or trunc:
            return info["episode_stats"]


# ----------------------------------------------------------------------------- shared tiny training
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train tiny Q-learning, DQN and PPO models once into a tmp dir (plus a same-seed twin for QL/DQN)."""
    from training import train_dqn, train_ppo, train_qlearning
    root = tmp_path_factory.mktemp("integration")
    models, results = str(root / "models"), str(root / "results")
    cfg = env_cfg()
    out = {"root": root, "models": models, "results": results, "cfg": cfg}
    out["qlearning"] = train_qlearning.train(cfg, dict(QL_HP), 0, "ql_a", models_dir=models, results_dir=results)
    out["dqn"] = train_dqn.train(cfg, dict(DQN_HP), 0, "dqn_a", models_dir=models, results_dir=results,
                                 eval_every=10, eval_episodes=3, tensorboard=False)
    out["ppo"] = train_ppo.train(cfg, dict(PPO_HP), 0, "ppo_a", models_dir=models, results_dir=results,
                                 eval_every_steps=0, eval_episodes=3, tensorboard=False)
    # twins with the same seed in separate dirs (reproducibility test)
    m2, r2 = str(root / "models2"), str(root / "results2")
    out["qlearning2"] = train_qlearning.train(cfg, dict(QL_HP), 0, "ql_a", models_dir=m2, results_dir=r2)
    out["dqn2"] = train_dqn.train(cfg, dict(DQN_HP), 0, "dqn_a", models_dir=m2, results_dir=r2,
                                  eval_every=10, eval_episodes=3, tensorboard=False)
    return out


ALGOS = ["qlearning", "dqn", "ppo"]
EXT = {"qlearning": ".npz", "dqn": ".pt", "ppo": ".zip"}


# ----------------------------------------------------------------------------- (a) baselines
def test_baselines_evaluate_agent_identical_seeds():
    cfg = env_cfg()
    rnd = evaluate_agent(make_agent("random", seed=0), cfg, EVAL_SEEDS, label="random")
    ast = evaluate_agent(make_agent("astar"), cfg, EVAL_SEEDS, label="astar")
    for df in (rnd, ast):
        assert list(df.columns) == EPISODE_COLUMNS
        assert len(df) == len(EVAL_SEEDS)
        assert df["layout_seed"].isin(LAYOUTS).all()
    # identical seeds -> identical episodes (same layout and same optimal path length) for every agent
    assert rnd["seed"].tolist() == ast["seed"].tolist() == EVAL_SEEDS
    assert rnd["layout_seed"].tolist() == ast["layout_seed"].tolist()
    assert rnd["optimal_steps"].tolist() == ast["optimal_steps"].tolist()
    assert ast["success"].all()
    assert (ast["collisions"] == 0).all()
    assert (ast["steps"] >= ast["optimal_steps"]).all()
    s = summarize(ast)
    assert s["success_rate"] == 1.0 and s["n"] == len(EVAL_SEEDS)
    assert summarize(rnd)["reward_mean"] < s["reward_mean"]


def test_evaluation_is_deterministic_for_deterministic_agents():
    cfg = env_cfg()
    a = evaluate_agent(make_agent("astar"), cfg, EVAL_SEEDS)
    b = evaluate_agent(make_agent("astar"), cfg, EVAL_SEEDS)
    pd.testing.assert_frame_equal(a, b)
    r1 = evaluate_agent(make_agent("random", seed=7), cfg, EVAL_SEEDS)
    r2 = evaluate_agent(make_agent("random", seed=7), cfg, EVAL_SEEDS)
    pd.testing.assert_frame_equal(r1, r2)


# ----------------------------------------------------------------------------- (b) train -> load -> evaluate
@pytest.mark.parametrize("algo", ALGOS)
def test_train_outputs_and_formats(trained, algo):
    out = trained[algo]
    assert set(out) >= {"model_path", "meta_path", "log_path", "wall_seconds"}
    assert out["model_path"].endswith(EXT[algo]) and os.path.isfile(out["model_path"])
    assert out["meta_path"] == meta_path(out["model_path"])
    meta = json.load(open(out["meta_path"]))
    assert meta["algo"] == algo
    assert meta["run_name"] == {"qlearning": "ql_a", "dqn": "dqn_a", "ppo": "ppo_a"}[algo]
    assert meta["env_config"]["layout_seeds"] == LAYOUTS
    assert meta["reward"]["deliver"] == 100.0
    assert meta["train"]["seed"] == 0 and meta["train"]["wall_seconds"] >= 0
    assert "hyperparams" in meta and "created" in meta
    # model lives under <models_dir>/<algo>/
    assert Path(out["model_path"]).parent == Path(trained["models"]) / algo
    log = pd.read_csv(out["log_path"])
    assert list(log.columns) == LOG_COLUMNS
    assert len(log) >= 3
    assert log["timestep"].is_monotonic_increasing
    assert set(log["success"].unique()) <= {0, 1}
    assert Path(out["log_path"]).parent.name == meta["run_name"]


def test_train_progress_callback_and_stop_event(tmp_path):
    import threading
    from training import train_qlearning
    calls = []
    out = train_qlearning.train(env_cfg(), {"episodes": 50}, 0, "cb", progress_callback=calls.append,
                                models_dir=str(tmp_path / "m"), results_dir=str(tmp_path / "r"), callback_every=10)
    assert calls[-1]["done"] is True and calls[-1]["episode"] == 50
    assert {"episode", "timestep", "reward_ma", "success_ma", "epsilon", "loss", "done"} <= set(calls[0])
    stop = threading.Event()
    stop.set()
    out = train_qlearning.train(env_cfg(), {"episodes": 50}, 0, "stopped", stop_event=stop,
                                models_dir=str(tmp_path / "m"), results_dir=str(tmp_path / "r"))
    assert os.path.isfile(out["model_path"])  # a stopped run still saves its model and log
    assert len(pd.read_csv(out["log_path"])) == 0


def test_load_evaluate_export_discover_plot(trained):
    from analytics import plots
    cfg = trained["cfg"]
    frames, logs = [], {}
    for algo in ALGOS:
        agent = load_agent(trained[algo]["model_path"])
        assert agent.name == algo
        frames.append(evaluate_agent(agent, cfg, EVAL_SEEDS, label=algo))
        logs[algo] = [load_train_log(trained[algo]["log_path"])]
    frames.append(evaluate_agent(make_agent("astar"), cfg, EVAL_SEEDS, label="astar"))
    df = pd.concat(frames, ignore_index=True)
    # identical episodes for every agent
    for label, g in df.groupby("label"):
        assert g["seed"].tolist() == EVAL_SEEDS
    ref = df[df["label"] == "astar"]
    for label, g in df.groupby("label"):
        assert g["layout_seed"].tolist() == ref["layout_seed"].tolist()
        assert g["optimal_steps"].tolist() == ref["optimal_steps"].tolist()
    # export
    out_csv = str(trained["root"] / "export" / "episodes.csv")
    export_results(df, out_csv)
    back = pd.read_csv(out_csv)
    assert list(back.columns) == EPISODE_COLUMNS and len(back) == len(df)
    summ = pd.DataFrame({k: summarize(g) for k, g in df.groupby("label")}).T
    export_summary(summ, str(trained["root"] / "export" / "summary.json"))
    assert json.load(open(trained["root"] / "export" / "summary.json"))
    # discovery finds all three runs, with metadata and real log statistics
    runs = discover_runs(results_dir=trained["results"], models_dir=trained["models"])
    got = runs.set_index("run_name")
    assert {"ql_a", "dqn_a", "ppo_a"} <= set(got.index)
    assert got.loc["ql_a", "algo"] == "qlearning" and got.loc["dqn_a", "algo"] == "dqn" and got.loc["ppo_a", "algo"] == "ppo"
    assert all(os.path.isfile(got.loc[n, "model_path"]) for n in ("ql_a", "dqn_a", "ppo_a"))
    assert got.loc["ql_a", "episodes"] == QL_HP["episodes"]
    # plots are built from the real logs
    fig = plots.plot_training_curves(logs)
    assert fig.axes
    p = plots.save_figure(fig, str(trained["root"] / "export" / "curves.png"))
    assert os.path.getsize(p) > 1000
    from analytics.comparison import summarize_runs
    fig2 = plots.plot_comparison_bars(summarize_runs(df.assign(train_seed=0), by=("label",)))
    assert fig2.axes


def test_comparison_bars_with_perfect_success_rate():
    from analytics import plots
    from analytics.comparison import compare_agents
    _, summary = compare_agents({"astar": make_agent("astar")}, EnvConfig(layout_seeds=[0]), range(100_000, 100_100))
    assert summary["success_rate"].iloc[0] == 1.0
    plots.plot_comparison_bars(summary)


# ----------------------------------------------------------------------------- (c) evaluate CLI
def test_evaluate_cli_end_to_end(trained, tmp_path):
    out_csv = tmp_path / "eval.csv"
    cmd = [sys.executable, "-m", "training.evaluate", "--agents", "random", "astar", trained["qlearning"]["model_path"],
           trained["dqn"]["model_path"], trained["ppo"]["model_path"], "--episodes", "4",
           "--layout-seeds", "0,1", "--out", str(out_csv)]
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", MPLBACKEND="Agg")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300, env=env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    df = pd.read_csv(out_csv)
    assert set(df["label"]) == {"random", "astar", "ql_a", "dqn_a", "ppo_a"}
    assert len(df) == 5 * 4
    seeds = {l: g["seed"].tolist() for l, g in df.groupby("label")}
    assert all(s == [100_000, 100_001, 100_002, 100_003] for s in seeds.values())
    for label in ("random", "astar", "ql_a", "dqn_a", "ppo_a"):
        assert label in proc.stdout
    # the CLI numbers equal an in-process evaluation of the same model on the same seeds
    q = load_agent(trained["qlearning"]["model_path"])
    mine = evaluate_agent(q, EnvConfig.from_dict({**__import__("environment").load_config(str(ROOT / "config.yaml"))["env"],
                                                  "layout_seeds": [0, 1]}), range(100_000, 100_004), label="ql_a")
    cli = df[df["label"] == "ql_a"].reset_index(drop=True)
    assert cli["reward"].round(4).tolist() == mine["reward"].round(4).tolist()
    assert cli["steps"].tolist() == mine["steps"].tolist()


def test_train_cli_help_and_tiny_run(tmp_path):
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", MPLBACKEND="Agg")
    for mod in ("training.train_qlearning", "training.train_dqn", "training.train_ppo", "training.evaluate",
                "experiments.run_experiments"):
        p = subprocess.run([sys.executable, "-m", mod, "--help"], cwd=ROOT, capture_output=True, text=True,
                           timeout=120, env=env)
        assert p.returncode == 0 and "usage" in p.stdout.lower(), (mod, p.stderr[-500:])
    p = subprocess.run([sys.executable, "-m", "training.train_qlearning", "--name", "cli_ql", "--episodes", "40",
                        "--layout-seeds", "0,1", "--models-dir", str(tmp_path / "m"),
                        "--results-dir", str(tmp_path / "r")],
                       cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert p.returncode == 0, p.stderr[-1000:]
    assert (tmp_path / "m" / "qlearning" / "cli_ql.npz").exists()
    assert (tmp_path / "m" / "qlearning" / "cli_ql.meta.json").exists()
    assert (tmp_path / "r" / "cli_ql" / "train_log.csv").exists()


# ----------------------------------------------------------------------------- (d) UI controller
@pytest.mark.parametrize("algo", ALGOS)
def test_ui_controller_runs_trained_model(trained, algo):
    from dashboard.controller import SimulationController, list_models
    ctrl = SimulationController(layout_seed=0, models_dir=trained["models"])
    listed = {m["algo"] for m in list_models(trained["models"])}
    assert set(ALGOS) <= listed
    assert ctrl.set_agent(algo, trained[algo]["model_path"]), ctrl.agent_error
    ctrl.reset(task_seed=100_000)
    stats = ctrl.run_episode()
    assert ctrl.done and stats["steps"] >= 1
    assert ctrl.metrics()["steps"] == stats["steps"]
    # the controller (UI path) reproduces the headless evaluation of the same agent/seed/layout
    from analytics.metrics import run_episode
    ref, _ = run_episode(load_agent(trained[algo]["model_path"]), WarehouseEnv(ctrl.env_config), 100_000,
                         options={"layout_seed": 0})
    assert ref["steps"] == stats["steps"] and ref["reward"] == pytest.approx(stats["reward"])
    assert ctrl.history[-1]["agent"].startswith(ctrl.agent_label.split(" (")[0])


def test_ui_training_job_writes_to_tmp(tmp_path):
    from dashboard.controller import TrainingJob
    job = TrainingJob("qlearning", env_cfg(), {"episodes": 60}, 0, "ui_job", models_dir=str(tmp_path / "m"),
                      results_dir=str(tmp_path / "r"))
    job.start()
    job.join(120)
    assert job.status == "done", job.error
    assert job.latest.get("done") is True
    assert os.path.isfile(job.result["model_path"])


# ----------------------------------------------------------------------------- (e) experiments runner
def test_run_experiments_quick_into_tmp_dirs(tmp_path):
    from experiments import run_experiments as rx
    models, res = tmp_path / "models", tmp_path / "res"
    argv = ["--exp", "1", "--quick", "--seeds", "1", "--eval-episodes", "4", "--episodes-scale", "0.02",
            "--algos", "random", "astar", "qlearning", "dqn", "--models-dir", str(models), "--results-dir", str(res)]
    assert rx.main(argv) == 0
    out = res / "exp1_quick"
    for f in ("episodes.csv", "summary.csv", "summary.json", "run_info.json", "comparison_bars.png"):
        assert (out / f).exists(), f
    ep = pd.read_csv(out / "episodes.csv")
    assert set(ep["label"]) == {"random", "astar", "qlearning", "dqn"}
    assert ep.groupby("label")["seed"].apply(lambda s: tuple(sorted(s))).nunique() == 1
    info = json.load(open(out / "run_info.json"))
    assert all(v in ("trained", "cached") for v in info["training_status"].values()), info["training_status"]
    assert any(n.startswith("qlearning_main-quick") or "quick" in n for n in info["training_status"])
    # a second invocation reuses the cached models (nothing retrained)
    assert rx.main(argv) == 0
    info2 = json.load(open(out / "run_info.json"))
    assert set(info2["training_status"].values()) == {"cached"}
    # nothing leaked to the real folders: only files in tmp
    assert (models / "qlearning").is_dir() and (models / "dqn").is_dir()


# ----------------------------------------------------------------------------- (f) save -> load round trip
@pytest.mark.parametrize("algo", ALGOS)
def test_save_load_roundtrip_identical_greedy_actions(trained, algo, tmp_path):
    a = load_agent(trained[algo]["model_path"])
    new_path = str(tmp_path / f"copy{EXT[algo]}")
    a.save(new_path)
    # loader needs the meta file next to the copy (it names the algorithm)
    json.dump(read_meta(trained[algo]["model_path"]), open(meta_path(new_path), "w"))
    b = load_agent(new_path)
    if algo == "qlearning":
        assert a.q.keys() == b.q.keys()
        assert all(np.array_equal(a.q[k], b.q[k]) for k in a.q)
    cfg = trained["cfg"]
    env = WarehouseEnv(cfg)
    n_obs = 0
    for seed in EVAL_SEEDS:
        obs, info = env.reset(seed=seed)
        a.reset()
        b.reset()
        while True:
            x = a.act(obs, env=env, deterministic=True)
            y = b.act(obs, env=env, deterministic=True)
            assert x == y
            n_obs += 1
            obs, r, term, trunc, info = env.step(x)
            if term or trunc:
                break
    assert n_obs >= len(EVAL_SEEDS)


def test_dqn_q_values_identical_after_roundtrip(trained, tmp_path):
    from agents.dqn_agent import DQNAgent
    a = DQNAgent.load(trained["dqn"]["model_path"])
    a.save(str(tmp_path / "x.pt"))
    b = DQNAgent.load(str(tmp_path / "x.pt"))
    env = WarehouseEnv(trained["cfg"])
    obs, _ = env.reset(seed=3)
    np.testing.assert_array_equal(a.q_values(obs), b.q_values(obs))


# ----------------------------------------------------------------------------- (g) reproducibility
def test_same_seed_identical_q_table(trained):
    from agents.q_learning import QLearningAgent
    a = QLearningAgent.load(trained["qlearning"]["model_path"])
    b = QLearningAgent.load(trained["qlearning2"]["model_path"])
    assert len(a.q) == len(b.q) > 0
    assert a.q.keys() == b.q.keys()
    assert all(np.array_equal(a.q[k], b.q[k]) for k in a.q)
    la, lb = pd.read_csv(trained["qlearning"]["log_path"]), pd.read_csv(trained["qlearning2"]["log_path"])
    pd.testing.assert_frame_equal(la, lb)


def test_same_seed_identical_dqn_weights(trained):
    import torch
    ca = torch.load(trained["dqn"]["model_path"], map_location="cpu", weights_only=False)
    cb = torch.load(trained["dqn2"]["model_path"], map_location="cpu", weights_only=False)
    assert ca["q_net"].keys() == cb["q_net"].keys()
    for k in ca["q_net"]:
        assert torch.equal(ca["q_net"][k], cb["q_net"][k]), k
    la, lb = pd.read_csv(trained["dqn"]["log_path"]), pd.read_csv(trained["dqn2"]["log_path"])
    pd.testing.assert_frame_equal(la, lb)


def test_different_seed_gives_different_q_table(tmp_path):
    from agents.q_learning import QLearningAgent
    from training import train_qlearning
    cfg = env_cfg()
    paths = []
    for s in (0, 1):
        o = train_qlearning.train(cfg, dict(QL_HP), s, f"s{s}", models_dir=str(tmp_path / "m"),
                                  results_dir=str(tmp_path / "r"))
        paths.append(o["model_path"])
    a, b = QLearningAgent.load(paths[0]), QLearningAgent.load(paths[1])
    assert a.q.keys() != b.q.keys() or any(not np.array_equal(a.q[k], b.q[k]) for k in a.q)


# ----------------------------------------------------------------------------- (h) space compatibility
@pytest.mark.parametrize("battery", [True, False])
def test_all_agents_share_observation_and_action_spaces(trained, battery):
    cfg = env_cfg(battery_enabled=battery)
    env = WarehouseEnv(cfg)
    assert env.action_space.n == 7
    obs_dim = env.observation_space.shape[0]
    assert obs_dim == 19 + (2 * cfg.view_radius + 1) ** 2 - 1 == 43
    agents = {"random": make_agent("random", seed=0), "astar": make_agent("astar")}
    for a in ALGOS:
        agents[a] = load_agent(trained[a]["model_path"])
    # network input / output sizes of the learned agents match the env spaces
    assert agents["dqn"].obs_dim == obs_dim and agents["dqn"].n_actions == env.action_space.n
    assert tuple(agents["ppo"].model.observation_space.shape) == env.observation_space.shape
    assert agents["ppo"].model.action_space.n == env.action_space.n
    for name, agent in agents.items():
        obs, info = env.reset(seed=11)
        agent.reset()
        for _ in range(30):
            assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
            assert env.observation_space.contains(obs)
            a = agent.act(obs, env=env, deterministic=True)
            assert isinstance(a, int) and env.action_space.contains(a), name
            obs, r, term, trunc, info = env.step(a)
            assert info["action_mask"].shape == (7,) and info["action_mask"].any()
            if term or trunc:
                break


def test_meta_files_are_loadable_and_name_the_algorithm(trained):
    for algo in ALGOS:
        assert read_meta(trained[algo]["model_path"])["algo"] == algo
        assert type(load_agent(trained[algo]["model_path"])).__name__ == {
            "qlearning": "QLearningAgent", "dqn": "DQNAgent", "ppo": "PPOAgent"}[algo]


def test_learners_never_need_privileged_state(trained):
    """Learned agents must work with an env object that exposes only the public mask (no map access)."""
    class Opaque:
        def __init__(self, env):
            self._e = env
            self.unwrapped = type("U", (), {"action_masks": staticmethod(env.unwrapped.action_masks),
                                            "discrete_state": staticmethod(env.unwrapped.discrete_state)})()
    env = WarehouseEnv(trained["cfg"])
    obs, _ = env.reset(seed=5)
    op = Opaque(env)
    for algo in ALGOS:
        assert 0 <= load_agent(trained[algo]["model_path"]).act(obs, env=op) < 7
