"""Evaluate / compare any agents on identical seeds.

``python -m training.evaluate --agents random astar models/qlearning/qlearning_fixed_s0.npz models/dqn/x.pt
 --episodes 100 --layout-seeds 0 --out experiments/results/eval.csv``

Each agent plays the SAME episodes (``seed_base .. seed_base+n-1``, default 100000+), deterministic policy.
Evaluation seeds are disjoint from training seeds (>= 1e9) and from model-selection seeds (5e8).
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Optional, Sequence

import pandas as pd

from agents import load_agent, make_agent
from analytics.metrics import evaluate_agent, summarize
from environment import EnvConfig


def resolve_agent(spec: Any):
    """Agent instance | 'random' | 'astar' | path to a model file (.npz / .pt / .zip)."""
    if not isinstance(spec, str):
        return spec, getattr(spec, "name", type(spec).__name__)
    if spec.lower() in ("random", "astar"):
        return make_agent(spec.lower()), spec.lower()
    if not os.path.exists(spec):
        raise FileNotFoundError(f"model file not found: {spec}")
    return load_agent(spec), os.path.splitext(os.path.basename(spec))[0]


def evaluate_model(model_path_or_agent_name: Any, env_config: Optional[EnvConfig] = None, n_episodes: int = 100,
                   seed_base: int = 100_000, layout_seeds: Optional[Sequence[int]] = None,
                   out_csv: Optional[str] = None, label: Optional[str] = None) -> pd.DataFrame:
    """Per-episode results of one agent (see analytics.metrics.EPISODE_COLUMNS). ``label`` defaults to the
    model file stem (or 'random'/'astar'). Optionally writes the rows to ``out_csv``."""
    agent, default_label = resolve_agent(model_path_or_agent_name)
    df = evaluate_agent(agent, env_config, range(seed_base, seed_base + int(n_episodes)),
                        layout_seeds=layout_seeds, label=label or default_label)
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


def compare_agents(specs: Sequence[Any], env_config: Optional[EnvConfig] = None, n_episodes: int = 100,
                   seed_base: int = 100_000, layout_seeds: Optional[Sequence[int]] = None,
                   out_csv: Optional[str] = None) -> pd.DataFrame:
    """Evaluate several agents on identical seeds; returns the concatenated per-episode frame."""
    frames = [evaluate_model(s, env_config, n_episodes, seed_base, layout_seeds) for s in specs]
    df = pd.concat(frames, ignore_index=True)
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in df.groupby("label", sort=False):
        s = summarize(g)
        rows.append({"label": label, "n": len(g), "success": s["success_rate"], "reward": s["reward_mean"],
                     "reward_std": s["reward_std"], "steps_succ": s["steps_per_delivery"],
                     "collisions": s["collisions_mean"]})
    return pd.DataFrame(rows)


def main(argv=None) -> pd.DataFrame:
    from environment.config import load_config
    p = argparse.ArgumentParser(description="Evaluate agents on identical seeds")
    p.add_argument("--agents", nargs="+", required=True, help="random | astar | path/to/model")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=None)
    p.add_argument("--layout-seeds", default=None, help="comma list | 'train' | 'test' | omit = env config")
    p.add_argument("--no-battery", action="store_true")
    p.add_argument("--reward-preset", default=None)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    env_d = dict(cfg.get("env", {}))
    if "reward" in cfg:
        env_d["reward"] = cfg["reward"].get("preset", "default")
    if args.reward_preset:
        env_d["reward"] = args.reward_preset
    if args.no_battery:
        env_d["battery_enabled"] = False
    ls = None
    if args.layout_seeds:
        sp = cfg.get("splits", {})
        ls = (sp["train_layout_seeds"] if args.layout_seeds == "train" else sp["test_layout_seeds"]
              if args.layout_seeds == "test" else [int(x) for x in args.layout_seeds.split(",")])
    base = args.seed_base if args.seed_base is not None else cfg.get("splits", {}).get("eval_seed_base", 100_000)
    df = compare_agents(args.agents, EnvConfig.from_dict(env_d), args.episodes, base, ls, args.out)
    print(summary_table(df).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df


if __name__ == "__main__":
    main()
