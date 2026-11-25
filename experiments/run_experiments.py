"""Experiment orchestration: ``python -m experiments.run_experiments --exp {1,2,3,4,all} [--seeds N] [--quick]``.

Pipeline per experiment: (1) train N independent training seeds per learner through the trainers'
``train()`` API (skipped when the model for that run name already exists, unless ``--force``; modules that
do not exist yet are skipped with a warning), (2) evaluate every agent on *identical* held-out episode seeds
(``splits.eval_seed_base`` ..), (3) aggregate per training seed first, then mean +- std across training
runs, (4) write ``experiments/results/exp<k>/{episodes.csv,summary.csv,summary.json,*.png}`` and print a
markdown table. Quick runs write to ``exp<k>_quick`` and use ``-quick`` run-name suffixes so they never
mix with full runs.

Run names are shared across experiments when the training setup is identical (e.g. ``qlearning_main_s0``
is used by exp1, exp3, exp2/default and exp4/with_battery), so each model is trained once.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from typing import Optional

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ALGO_MODULES = {"qlearning": "training.train_qlearning", "dqn": "training.train_dqn", "ppo": "training.train_ppo"}
MODEL_EXT = {"qlearning": ".npz", "dqn": ".pt", "ppo": ".zip"}
BUDGET_KEY = {"qlearning": "episodes", "dqn": "episodes", "ppo": "total_timesteps"}
BASELINES = ("random", "astar")
CONFIG_DIR = os.path.join(ROOT, "experiments", "configs")


# ----------------------------------------------------------------------------- settings
class Settings:
    """Resolved run settings for one experiment (config.yaml + exp yaml + CLI)."""

    def __init__(self, exp: int, args, base_cfg: dict):
        with open(os.path.join(CONFIG_DIR, f"exp{exp}.yaml"), encoding="utf-8") as f:
            ec = yaml.safe_load(f)
        self.exp, self.ec, self.base = exp, ec, base_cfg
        self.quick = bool(args.quick)
        q = ec.get("quick", {}) if self.quick else {}
        self.algorithms = [a for a in ec["algorithms"] if args.algos is None or a in args.algos]
        self.n_seeds = int(args.seeds if args.seeds is not None else q.get("train_seeds", ec.get("train_seeds", 3)))
        self.eval_episodes = int(args.eval_episodes if args.eval_episodes is not None
                                 else q.get("eval_episodes", ec.get("eval_episodes", 100)))
        self.scale = float(args.episodes_scale)
        hp = {a: dict(v) for a, v in (ec.get("hparams") or {}).items()}
        for a, v in (q.get("hparams") or {}).items():
            hp.setdefault(a, {}).update(v)
        for a, v in hp.items():
            k = BUDGET_KEY.get(a)
            if k in v:
                v[k] = max(1, int(round(v[k] * self.scale)))
        self.hparams = hp
        splits = base_cfg["splits"]
        self.default_train_layouts = list(splits["train_layout_seeds"])
        self.train_layouts = list(ec.get("train_layouts") or self.default_train_layouts)
        self.eval_layouts = list(ec.get("eval_layouts") or self.train_layouts)
        self.test_layouts = list(ec.get("test_layouts") or splits["test_layout_seeds"])
        self.eval_seeds = list(range(int(splits["eval_seed_base"]), int(splits["eval_seed_base"]) + self.eval_episodes))
        self.preset = ec.get("reward_preset", base_cfg.get("reward", {}).get("preset", "default"))
        self.battery = bool(ec.get("battery_enabled", base_cfg["env"].get("battery_enabled", True)))
        self.models_dir = os.path.join(ROOT, args.models_dir) if not os.path.isabs(args.models_dir) else args.models_dir
        res = os.path.join(ROOT, args.results_dir) if not os.path.isabs(args.results_dir) else args.results_dir
        self.results_root = res
        self.out_dir = os.path.join(res, f"exp{exp}" + ("_quick" if self.quick else ""))
        self.suffix = ("-quick" if self.quick else "") + (f"-x{self.scale:g}" if self.scale != 1 else "")
        if self.train_layouts != self.default_train_layouts:
            self.suffix += "-L" + hashlib.md5(json.dumps(self.train_layouts).encode()).hexdigest()[:6]
        self.train_seeds = list(range(self.n_seeds))

    def env_config(self, preset=None, battery=None, layouts=None, over=None):
        from environment import EnvConfig
        env = dict(self.base["env"])
        env.update(reward=preset or self.preset, battery_enabled=self.battery if battery is None else battery,
                   layout_seeds=list(layouts if layouts is not None else self.train_layouts))
        env.update(over or {})
        return EnvConfig.from_dict(env)

    def run_name(self, algo: str, seed: int, preset: Optional[str] = None, battery: Optional[bool] = None,
                 tag: Optional[str] = None) -> str:
        if tag:
            return f"{algo}_{tag}{self.suffix}_s{seed}"
        preset = preset or self.preset
        battery = self.battery if battery is None else battery
        if battery:
            tag = "main" if preset == "default" else f"rw-{preset}"
        else:
            tag = "nobat" if preset == "default" else f"nobat-rw-{preset}"
        return f"{algo}_{tag}{self.suffix}_s{seed}"

    def model_path(self, algo: str, run_name: str) -> str:
        return os.path.join(self.models_dir, algo, run_name + MODEL_EXT[algo])

    def train_job(self, algo: str, seed: int, preset=None, battery=None, tag=None, over=None) -> dict:
        rn = self.run_name(algo, seed, preset, battery, tag)
        return {"algo": algo, "run_name": rn, "seed": seed, "hparams": dict(self.hparams.get(algo, {})),
                "env_config": self.env_config(preset, battery, over=over).to_dict(), "models_dir": self.models_dir,
                "results_dir": self.results_root, "model_path": self.model_path(algo, rn)}


# ----------------------------------------------------------------------------- training
def _train_job(job: dict) -> dict:
    """Top-level (picklable, spawn-safe) worker: train one run through the trainer API."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    res = {"run_name": job["run_name"], "algo": job["algo"], "ok": False, "error": None}
    t0 = time.time()
    try:
        if job.get("single_thread"):
            try:
                import torch
                torch.set_num_threads(1)
            except ImportError:
                pass
        from environment import EnvConfig
        mod = importlib.import_module(ALGO_MODULES[job["algo"]])
        mod.train(EnvConfig.from_dict(job["env_config"]), job["hparams"] or None, job["seed"], job["run_name"],
                  models_dir=job["models_dir"], results_dir=job["results_dir"])
        res["ok"] = True
    except Exception as e:  # keep the whole experiment alive; reported in the summary
        res["error"] = f"{type(e).__name__}: {e}"
    res["seconds"] = time.time() - t0
    return res


def ensure_trained(jobs: list, workers: int = 1, force: bool = False) -> dict:
    """Train all jobs whose model is missing. Returns ``{run_name: 'trained'|'cached'|'skipped: ...'|'failed: ...'}``."""
    status, todo = {}, []
    for j in jobs:
        if importlib.util.find_spec(ALGO_MODULES[j["algo"]]) is None:
            status[j["run_name"]] = f"skipped: module {ALGO_MODULES[j['algo']]} not found"
        elif os.path.exists(j["model_path"]) and not force:
            status[j["run_name"]] = "cached"
        else:
            todo.append(j)
    for algo in sorted({j["algo"] for j in jobs if str(status.get(j["run_name"], "")).startswith("skipped")}):
        warnings.warn(f"algorithm '{algo}' skipped: {ALGO_MODULES[algo]} is not available yet")
    if todo:
        print(f"[train] {len(todo)} run(s) to train ({workers} worker(s))", flush=True)
        if workers > 1:
            for j in todo:
                j["single_thread"] = True
            with mp.get_context("spawn").Pool(min(workers, len(todo))) as pool:
                results = list(pool.imap_unordered(_train_job, todo))
        else:
            results = [_train_job(j) for j in todo]
        for r in results:
            if r["ok"]:
                status[r["run_name"]] = "trained"
                print(f"[train] {r['run_name']} done in {r['seconds']:.0f}s", flush=True)
            else:
                status[r["run_name"]] = f"failed: {r['error']}"
                warnings.warn(f"training {r['run_name']} failed: {r['error']}")
    return status


# ----------------------------------------------------------------------------- evaluation
def _load(algo: str, S: Settings, run_name: str, seed: int, cache: dict):
    from agents import load_agent, make_agent
    if algo in BASELINES:
        key = (algo, seed)
        if key not in cache:
            try:
                cache[key] = make_agent(algo, seed=seed) if algo == "random" else make_agent(algo)
            except ImportError as e:
                warnings.warn(f"baseline {algo} unavailable: {e}")
                cache[key] = None
        return cache[key]
    if run_name not in cache:
        path = S.model_path(algo, run_name)
        try:
            cache[run_name] = load_agent(path) if os.path.exists(path) else None
        except Exception as e:
            warnings.warn(f"could not load {path}: {e}")
            cache[run_name] = None
    return cache[run_name]


def evaluate_entries(entries: list, S: Settings) -> pd.DataFrame:
    """entries: dicts with condition, algo, train_seed, run_name, env_cfg. All use the same eval seeds."""
    from analytics.metrics import evaluate_agent
    frames, cache = [], {}
    for e in entries:
        agent = _load(e["algo"], S, e["run_name"], e["train_seed"], cache)
        if agent is None:
            warnings.warn(f"no agent for {e['algo']} / {e['run_name']}: omitted from results")
            continue
        d = evaluate_agent(agent, e["env_cfg"], S.eval_seeds, label=e["algo"])
        d.insert(0, "condition", e["condition"])
        d["train_seed"] = e["train_seed"]
        d["run_name"] = e["run_name"]
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _entries(S, condition, algo, env_cfg, preset=None, battery=None, train_battery=None, train_preset=None, tag=None):
    """Entries (one per training seed for learners, one for baselines)."""
    if algo in BASELINES:
        return [{"condition": condition, "algo": algo, "train_seed": 0, "run_name": algo, "env_cfg": env_cfg}]
    return [{"condition": condition, "algo": algo, "train_seed": s, "env_cfg": env_cfg,
             "run_name": S.run_name(algo, s, train_preset, train_battery, tag)} for s in S.train_seeds]


def _jobs_for(S, algos, preset=None, battery=None, tag=None, over=None):
    return [S.train_job(a, s, preset, battery, tag, over) for a in algos if a not in BASELINES for s in S.train_seeds]


# ----------------------------------------------------------------------------- experiments
def build_exp(exp: int, S: Settings, workers: int, force: bool):
    """Returns (entries, status). Training happens here."""
    learners = [a for a in S.algorithms if a not in BASELINES]
    entries, jobs = [], []
    if exp == 1:
        env = S.env_config(layouts=S.eval_layouts)
        jobs = _jobs_for(S, S.algorithms)
        for a in S.algorithms:
            entries += _entries(S, "main", a, env)
    elif exp == 2:
        presets = S.ec["reward_presets"]
        eval_env = S.env_config(preset=S.ec.get("eval_reward", "default"), layouts=S.eval_layouts)
        for p in presets:
            jobs += _jobs_for(S, learners, preset=p)
            for a in learners:
                entries += _entries(S, p, a, eval_env, train_preset=p)
    elif exp == 3:
        jobs = _jobs_for(S, S.algorithms)
        for split, lay in (("train_layouts", S.train_layouts), ("unseen_layouts", S.test_layouts)):
            env = S.env_config(layouts=lay)
            for a in S.algorithms:
                entries += _entries(S, split, a, env)
    elif exp == 4:
        jobs = _jobs_for(S, learners, battery=True) + _jobs_for(S, learners, battery=False)
        tight = dict(S.ec.get("tight_battery") or {"battery_capacity": 24, "battery_init_range": [0.4, 0.6]})
        jobs += _jobs_for(S, learners, tag="tightbat", over=tight)
        env_tight = S.env_config(battery=True, layouts=S.eval_layouts, over=tight)
        env_on, env_off = S.env_config(battery=True, layouts=S.eval_layouts), S.env_config(battery=False, layouts=S.eval_layouts)
        for a in S.algorithms:
            entries += _entries(S, "tight_battery", a, env_tight, tag="tightbat")
            entries += _entries(S, "with_battery", a, env_on, train_battery=True)
            if a in BASELINES:
                entries += _entries(S, "no_battery", a, env_off)
            else:
                entries += _entries(S, "no_battery", a, env_off, train_battery=False)
                entries += _entries(S, "no_battery_agent_in_battery_env", a, env_on, train_battery=False)
    status = ensure_trained(jobs, workers, force)
    return entries, status


def _md_table(summary: pd.DataFrame) -> str:
    cols = [("reward_mean", "reward", 1), ("success_rate", "success", 3), ("steps_per_delivery", "steps/deliv", 1),
            ("collisions_mean", "collisions", 2), ("energy_per_delivery", "energy/deliv", 1),
            ("path_efficiency", "path eff", 2),
            ("battery_depleted_rate", "depleted", 3), ("recharges_mean", "recharges", 2)]
    idx = summary.index.names
    head = list(idx) + [c[1] for c in cols] + ["runs"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for key, r in summary.iterrows():
        key = key if isinstance(key, tuple) else (key,)
        cells = [str(k) for k in key]
        for c, _, nd in cols:
            m, s = r.get(c), r.get(c + "_std_runs", float("nan"))
            cells.append("n/a" if pd.isna(m) else f"{m:.{nd}f}" + ("" if pd.isna(s) or r["n_runs"] < 2 else f" ± {s:.{nd}f}"))
        cells.append(str(int(r["n_runs"])))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _plots(exp: int, S: Settings, long_df: pd.DataFrame, summary: pd.DataFrame, out: str) -> list:
    from analytics import plots
    from analytics.metrics import load_train_log
    saved = []

    def save(fig, name):
        saved.append(plots.save_figure(fig, os.path.join(out, name)))

    flat = summary.reset_index()
    if exp == 1:
        save(plots.plot_comparison_bars(summary.droplevel("condition")), "comparison_bars.png")
        save(plots.plot_episode_length_distribution(long_df), "episode_lengths.png")
    elif exp == 2:
        for a in sorted(set(flat["label"])):
            s = flat[flat["label"] == a].set_index("condition")
            save(plots.plot_comparison_bars(s), f"{a}_reward_presets.png")
    elif exp == 3:
        t = flat.rename(columns={"label": "agent", "condition": "split"})
        t["split"] = t["split"].map({"train_layouts": "train", "unseen_layouts": "test"})
        save(plots.plot_generalization(t, "success_rate", "Success rate"), "generalization_success.png")
        save(plots.plot_generalization(t, "reward_mean", "Mean episode reward"), "generalization_reward.png")
    elif exp == 4:
        s = flat.assign(name=flat["label"] + " / " + flat["condition"]).set_index("name")
        save(plots.plot_comparison_bars(s), "battery_bars.png")
        save(plots.plot_comparison_bars(s, metrics=("success", "energy", "recharges", "depleted")), "battery_management.png")
    logs = {}
    for a in S.algorithms:
        for rn in sorted(set(long_df.loc[long_df["label"] == a, "run_name"])) if "run_name" in long_df else []:
            if exp in (1, 3) or "main" in rn:
                try:
                    logs.setdefault(a, []).append(load_train_log(rn, results_dir=S.results_root))
                except FileNotFoundError:
                    pass
    if logs:
        save(plots.plot_training_curves(logs), "training_curves.png")
    return saved


def run_experiment(exp: int, args, base_cfg: dict) -> Optional[pd.DataFrame]:
    from analytics.comparison import summarize_runs
    from analytics.export import export_results, export_summary
    S = Settings(exp, args, base_cfg)
    print(f"\n=== Experiment {exp}: {S.ec.get('title')} | algos={S.algorithms} train_seeds={S.n_seeds} "
          f"eval_episodes={S.eval_episodes} quick={S.quick} ===", flush=True)
    entries, status = build_exp(exp, S, args.workers, args.force)
    long_df = evaluate_entries(entries, S)
    if long_df.empty:
        print(f"[exp{exp}] no agents could be evaluated; nothing written.")
        return None
    summary = summarize_runs(long_df, by=("condition", "label"))
    os.makedirs(S.out_dir, exist_ok=True)
    export_results(long_df, os.path.join(S.out_dir, "episodes.csv"))
    export_summary(summary, os.path.join(S.out_dir, "summary.csv"))
    export_summary(summary, os.path.join(S.out_dir, "summary.json"))
    with open(os.path.join(S.out_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump({"experiment": exp, "title": S.ec.get("title"), "quick": S.quick, "train_seeds": S.train_seeds,
                   "eval_seeds": [S.eval_seeds[0], S.eval_seeds[-1]], "train_layouts": S.train_layouts,
                   "eval_layouts": S.eval_layouts, "test_layouts": S.test_layouts if exp == 3 else None,
                   "hparams": S.hparams, "training_status": status}, f, indent=2)
    try:
        saved = _plots(exp, S, long_df, summary, S.out_dir)
    except Exception as e:
        warnings.warn(f"plotting failed: {type(e).__name__}: {e}")
        saved = []
    print(_md_table(summary))
    print(f"[exp{exp}] wrote {S.out_dir} ({len(saved)} figures)")
    bad = {k: v for k, v in status.items() if v.startswith(("failed", "skipped"))}
    for k, v in bad.items():
        print(f"[exp{exp}] WARNING {k}: {v}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the Smart Warehouse RL experiments")
    ap.add_argument("--exp", default="all", choices=["1", "2", "3", "4", "all"])
    ap.add_argument("--seeds", type=int, default=None, help="independent training seeds per learner")
    ap.add_argument("--quick", action="store_true", help="tiny budgets (smoke run); writes exp<k>_quick")
    ap.add_argument("--episodes-scale", type=float, default=1.0, help="multiply training budgets")
    ap.add_argument("--workers", type=int, default=1, help="parallel training processes (spawn)")
    ap.add_argument("--force", action="store_true", help="retrain even if the model exists")
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--algos", nargs="*", default=None, help="restrict algorithms")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--results-dir", default="experiments/results")
    args = ap.parse_args(argv)
    from environment import load_config
    base = load_config(args.config)
    for e in (1, 2, 3, 4) if args.exp == "all" else (int(args.exp),):
        run_experiment(e, args, base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
