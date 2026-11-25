"""Train the tabular Q-learning agent.  CLI: ``python -m training.train_qlearning``.

Episode seeds: episode ``i`` of training run ``seed`` uses
``env.reset(seed = TRAIN_SEED_OFFSET + train_seed_base + seed * SEED_STRIDE + i)`` with
``TRAIN_SEED_OFFSET = 1_000_000_000`` and ``SEED_STRIDE = 10_000_000``. Evaluation seeds start at
``eval_seed_base`` (100000) and are expected to stay far below 1e9, hence disjoint from every training
seed (runs differ by 1e7, so up to 10M episodes per run never overlap).
Reproducible: the agent RNG is ``default_rng(seed)`` and env seeds are a pure function of (seed, i).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable, Optional

import numpy as np

from agents.q_learning import QLearningAgent
from environment import EnvConfig, WarehouseEnv

TRAIN_SEED_OFFSET = 1_000_000_000
SEED_STRIDE = 10_000_000
LOG_COLUMNS = ["episode", "timestep", "reward", "steps", "success", "collisions", "energy_used",
               "epsilon", "loss"]

# Defaults used when neither hparams nor config.yaml provide a value (tuned, see report).
DEFAULT_HPARAMS = {"episodes": 20000, "alpha": 0.2, "gamma": 0.99, "epsilon_start": 1.0,
                   "epsilon_end": 0.05, "epsilon_decay_episodes": 12000, "battery_buckets": 6}


def train_episode_seed(seed: int, episode: int, train_seed_base: int = 0) -> int:
    return TRAIN_SEED_OFFSET + int(train_seed_base) + int(seed) * SEED_STRIDE + int(episode)


def train(env_config: EnvConfig, hparams: Optional[dict] = None, seed: int = 0, run_name: str = "qlearning",
          progress_callback: Optional[Callable[[dict], None]] = None,
          stop_event: Optional[threading.Event] = None,
          models_dir: str = "models", results_dir: str = "experiments/results",
          use_action_mask: bool = False, train_seed_base: int = 0, callback_every: int = 100,
          ma_window: int = 100) -> dict:
    hp = dict(DEFAULT_HPARAMS)
    if hparams:
        unknown = set(hparams) - set(DEFAULT_HPARAMS) - {"use_action_mask"}
        if unknown:
            raise ValueError(f"unknown Q-learning hyperparameters: {sorted(unknown)}")
        hp.update(hparams)
    use_action_mask = bool(hp.pop("use_action_mask", use_action_mask) or use_action_mask)
    episodes = int(hp.pop("episodes"))
    agent = QLearningAgent(use_action_mask=use_action_mask, seed=seed, **hp)
    env = WarehouseEnv(env_config)
    run_dir = os.path.join(results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(models_dir, "qlearning"), exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.csv")
    model_path = os.path.join(models_dir, "qlearning", f"{run_name}.npz")
    meta_path = os.path.join(models_dir, "qlearning", f"{run_name}.meta.json")

    rew_ma, suc_ma = deque(maxlen=ma_window), deque(maxlen=ma_window)
    t0 = time.time()
    timestep = 0
    ep_done = 0
    state_key, update = agent.state_key, agent.update
    with open(log_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(LOG_COLUMNS)
        for ep in range(episodes):
            if stop_event is not None and stop_event.is_set():
                break
            eps = agent.epsilon_at(ep)
            obs, info = env.reset(seed=train_episode_seed(seed, ep, train_seed_base))
            key = state_key(env)
            mask = info["action_mask"] if use_action_mask else None
            ep_r, td_abs, n = 0.0, 0.0, 0
            while True:
                a = agent.select_action(key, eps, mask)
                obs, r, term, trunc, info = env.step(a)
                nkey = state_key(env)
                nmask = info["action_mask"] if use_action_mask else None
                td_abs += abs(update(key, a, r, nkey, term, nmask))
                key, mask = nkey, nmask
                ep_r += r
                n += 1
                if term or trunc:
                    break
            timestep += n
            st = info["episode_stats"]
            rew_ma.append(ep_r)
            suc_ma.append(float(st["success"]))
            w.writerow([ep, timestep, round(ep_r, 4), st["steps"], int(st["success"]), st["collisions"],
                        st["energy_used"], round(eps, 5), round(td_abs / n, 5)])
            ep_done = ep + 1
            if progress_callback is not None and ep_done % callback_every == 0:
                progress_callback({"episode": ep_done, "timestep": timestep, "reward_ma": float(np.mean(rew_ma)),
                                   "success_ma": float(np.mean(suc_ma)), "epsilon": eps,
                                   "loss": td_abs / n, "done": False})
    wall = time.time() - t0
    agent.epsilon = agent.epsilon_end
    meta = {"algo": "qlearning", "run_name": run_name, "env_config": env_config.to_dict(),
            "reward": env_config.reward.to_dict(),
            "hyperparams": {**agent.hparams(), "episodes_planned": episodes, "train_seed_base": train_seed_base,
                            "train_seed_offset": TRAIN_SEED_OFFSET, "seed_stride": SEED_STRIDE},
            "train": {"seed": seed, "episodes": ep_done, "timesteps": timestep, "wall_seconds": wall,
                      "n_states": len(agent.q)},
            "created": datetime.now().isoformat(timespec="seconds")}
    agent.save(model_path, meta=meta)
    if progress_callback is not None:
        progress_callback({"episode": ep_done, "timestep": timestep,
                           "reward_ma": float(np.mean(rew_ma)) if rew_ma else 0.0,
                           "success_ma": float(np.mean(suc_ma)) if suc_ma else 0.0,
                           "epsilon": agent.epsilon_at(max(ep_done - 1, 0)), "loss": None, "done": True})
    with open(os.path.join(run_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return {"model_path": model_path, "meta_path": meta_path, "log_path": log_path, "wall_seconds": wall}


def main(argv=None) -> dict:
    from environment.config import load_config
    p = argparse.ArgumentParser(description="Train tabular Q-learning on the Smart Warehouse env")
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--reward-preset", default=None)
    p.add_argument("--no-battery", action="store_true")
    p.add_argument("--layout-seeds", default=None, help="comma separated, e.g. 0,1,2 ('train' = config split)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--epsilon-decay-episodes", type=int, default=None)
    p.add_argument("--mask", action="store_true", help="use action masking")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--results-dir", default="experiments/results")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    env_d = dict(cfg.get("env", {}))
    if "reward" in cfg:
        env_d["reward"] = cfg["reward"].get("preset", "default") if isinstance(cfg["reward"], dict) else cfg["reward"]
    if args.reward_preset:
        env_d["reward"] = args.reward_preset
    if args.no_battery:
        env_d["battery_enabled"] = False
    if args.layout_seeds:
        env_d["layout_seeds"] = (cfg["splits"]["train_layout_seeds"] if args.layout_seeds == "train"
                                 else [int(x) for x in args.layout_seeds.split(",")])
    env_config = EnvConfig.from_dict(env_d)
    hp = {k: v for k, v in cfg.get("qlearning", {}).items()}
    for k in ("episodes", "alpha", "gamma", "epsilon_decay_episodes"):
        if getattr(args, k) is not None:
            hp[k] = getattr(args, k)
    name = args.name or f"qlearning_s{args.seed}"
    base = cfg.get("splits", {}).get("train_seed_base", 0)

    def cb(d):
        if d["done"] or d["episode"] % 1000 == 0:
            print(f"ep {d['episode']:>6} steps {d['timestep']:>9} reward_ma {d['reward_ma']:8.2f} "
                  f"success_ma {d['success_ma']:.2f} eps {d['epsilon']:.3f}", flush=True)

    out = train(env_config, hp, args.seed, name, progress_callback=cb, models_dir=args.models_dir,
                results_dir=args.results_dir, use_action_mask=args.mask, train_seed_base=base)
    print(out)
    return out


if __name__ == "__main__":
    main()
