"""Train PPO (Stable-Baselines3) on the Smart Warehouse env.  CLI: ``python -m training.train_ppo``.

Seeds: env episode i uses ``reset(seed=TRAIN_SEED_OFFSET + base + seed*SEED_STRIDE + i)`` (>= 1e9, disjoint from
eval seeds) via a small wrapper that supplies the seed on every reset; SB3 itself is seeded with ``seed``.
Reproducibility: PPO on CPU with one torch thread is repeatable in practice for the same seed/versions, but we do
not guarantee bit-for-bit equality across machines / torch versions (and not on MPS).
The env is NOT wrapped in ``Monitor`` (the env already emits ``info["episode_stats"]``).
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

import gymnasium as gym
import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from environment import EnvConfig, WarehouseEnv
from training.train_dqn import HELD_OUT_BASE, build_env_config, greedy_eval, seed_everything
from training.train_qlearning import SEED_STRIDE, TRAIN_SEED_OFFSET, train_episode_seed

LOG_COLUMNS = ["episode", "timestep", "reward", "steps", "success", "collisions", "energy_used",
               "epsilon", "loss"]
DEFAULT_HPARAMS = {"total_timesteps": 300_000, "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,
                   "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
                   "net_arch": [128, 128]}


class SeededEpisodes(gym.Wrapper):
    """Gives every episode a deterministic training seed (spaces unchanged, info passes through)."""

    def __init__(self, env, seed: int, train_seed_base: int = 0):
        super().__init__(env)
        self._seed, self._base, self.episode = seed, train_seed_base, 0

    def reset(self, *, seed=None, options=None):
        s = train_episode_seed(self._seed, self.episode, self._base) if seed is None else seed
        self.episode += 1
        return self.env.reset(seed=s, options=options)


def make_callback(log_writer, progress_callback, stop_event, callback_every, ma_window, state,
                  eval_env=None, eval_every_steps=0, held_out=None):
    from stable_baselines3.common.callbacks import BaseCallback

    class EpisodeLogger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rew, self.suc = deque(maxlen=ma_window), deque(maxlen=ma_window)
            self.ep = 0
            self.next_eval = eval_every_steps
            self.last_loss = None

        def _on_step(self) -> bool:
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                st = info.get("episode_stats")
                if not done or st is None:
                    continue
                self.rew.append(st["reward"])
                self.suc.append(float(st["success"]))
                log_writer.writerow([self.ep, self.num_timesteps, round(st["reward"], 4), st["steps"],
                                     int(st["success"]), st["collisions"], st["energy_used"], "",
                                     "" if self.last_loss is None else round(self.last_loss, 5)])
                self.ep += 1
                state["episodes"] = self.ep
                state["timesteps"] = self.num_timesteps
                if progress_callback is not None and self.ep % callback_every == 0:
                    progress_callback({"episode": self.ep, "timestep": self.num_timesteps,
                                       "reward_ma": float(np.mean(self.rew)), "success_ma": float(np.mean(self.suc)),
                                       "epsilon": None, "loss": self.last_loss, "done": False})
            if eval_every_steps and self.num_timesteps >= self.next_eval:
                self.next_eval += eval_every_steps
                ev = greedy_eval(PPOAgent(self.model), eval_env, held_out)
                self.logger.record("eval/success", ev["success"])
                self.logger.record("eval/reward", ev["reward"])
                state["evals"].append({"timestep": self.num_timesteps, **ev})
                if (ev["success"], ev["reward"]) > (state["best"]["success"], state["best"]["reward"]):
                    state["best"] = {**ev, "timestep": self.num_timesteps}
                    self.model.save(state["best_path"])
            v = self.model.logger.name_to_value.get("train/loss")
            if v is not None:
                self.last_loss = float(v)
            if stop_event is not None and stop_event.is_set():
                return False
            return True

        def get_ma(self):
            return (float(np.mean(self.rew)) if self.rew else 0.0, float(np.mean(self.suc)) if self.suc else 0.0)

    return EpisodeLogger()


def train(env_config: EnvConfig, hparams: Optional[dict] = None, seed: int = 0, run_name: str = "ppo",
          progress_callback: Optional[Callable[[dict], None]] = None,
          stop_event: Optional[threading.Event] = None,
          models_dir: str = "models", results_dir: str = "experiments/results",
          device: str = "cpu", train_seed_base: int = 0, callback_every: int = 50, ma_window: int = 100,
          eval_every_steps: int = 25_000, eval_episodes: int = 30, select: str = "best",
          tensorboard: bool = True) -> dict:
    from stable_baselines3 import PPO
    hp = dict(DEFAULT_HPARAMS)
    if hparams:
        hparams = dict(hparams)
        if "lr" in hparams:
            hparams["learning_rate"] = hparams.pop("lr")  # alias
        unknown = set(hparams) - set(DEFAULT_HPARAMS)
        if unknown:
            raise ValueError(f"unknown PPO hyperparameters: {sorted(unknown)}")
        hp.update(hparams)
    total = int(hp.pop("total_timesteps"))
    net_arch = [int(x) for x in hp.pop("net_arch")]
    seed_everything(seed)
    run_dir = os.path.join(results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(models_dir, "ppo"), exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.csv")
    model_path = os.path.join(models_dir, "ppo", f"{run_name}.zip")
    meta_path = os.path.join(models_dir, "ppo", f"{run_name}.meta.json")
    best_path = os.path.join(run_dir, "best_model")  # SB3 adds .zip

    env = SeededEpisodes(WarehouseEnv(env_config), seed, train_seed_base)
    eval_env = WarehouseEnv(env_config)
    model = PPO("MlpPolicy", env, seed=seed, device=device, verbose=0,
                policy_kwargs={"net_arch": net_arch},
                tensorboard_log=os.path.join(run_dir, "tb") if tensorboard else None, **hp)
    state = {"episodes": 0, "timesteps": 0, "evals": [], "best": {"success": -1.0, "reward": -1e18, "timestep": -1},
             "best_path": best_path}
    held_out = [HELD_OUT_BASE + k for k in range(eval_episodes)]
    t0 = time.time()
    with open(log_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(LOG_COLUMNS)
        cb = make_callback(w, progress_callback, stop_event, callback_every, ma_window, state, eval_env,
                           eval_every_steps, held_out)
        model.learn(total_timesteps=total, callback=cb, progress_bar=False, tb_log_name="tb")
        rma, sma = cb.get_ma()
    wall = time.time() - t0
    final_ev = greedy_eval(PPOAgent(model), eval_env, held_out)
    ship, chosen = model, "last"
    best = state["best"]
    if select == "best" and best["timestep"] >= 0 and (best["success"], best["reward"]) > (
            final_ev["success"], final_ev["reward"]):
        ship, chosen = PPO.load(best_path, device=device), f"best@{best['timestep']}"
    meta = {"algo": "ppo", "run_name": run_name, "env_config": env_config.to_dict(),
            "reward": env_config.reward.to_dict(),
            "hyperparams": {**hp, "total_timesteps": total, "net_arch": net_arch, "train_seed_base": train_seed_base,
                            "train_seed_offset": TRAIN_SEED_OFFSET, "seed_stride": SEED_STRIDE,
                            "reward_scale": 1.0, "device": device},
            "train": {"seed": seed, "episodes": state["episodes"], "timesteps": int(model.num_timesteps),
                      "wall_seconds": wall, "selected_model": chosen,
                      "held_out_eval": {"seed_base": HELD_OUT_BASE, "n": eval_episodes, "best": best,
                                        "final": final_ev}},
            "created": datetime.now().isoformat(timespec="seconds")}
    PPOAgent(ship).save(model_path, meta=meta)
    if progress_callback is not None:
        progress_callback({"episode": state["episodes"], "timestep": int(model.num_timesteps), "reward_ma": rma,
                           "success_ma": sma, "epsilon": None, "loss": None, "done": True})
    with open(os.path.join(run_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return {"model_path": model_path, "meta_path": meta_path, "log_path": log_path, "wall_seconds": wall}


def main(argv=None) -> dict:
    from environment.config import load_config
    p = argparse.ArgumentParser(description="Train PPO (SB3) on the Smart Warehouse env")
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--reward-preset", default=None)
    p.add_argument("--no-battery", action="store_true")
    p.add_argument("--layout-seeds", default=None)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-epochs", type=int, default=None)
    p.add_argument("--ent-coef", type=float, default=None)
    p.add_argument("--net-arch", default=None, help="comma separated, e.g. 128,128")
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-every-steps", type=int, default=25_000)
    p.add_argument("--select", choices=["best", "last"], default="best")
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--results-dir", default="experiments/results")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    env_config = build_env_config(args, cfg)
    hp = dict(cfg.get("ppo", {}))
    for src, dst in (("timesteps", "total_timesteps"), ("lr", "learning_rate"), ("gamma", "gamma"),
                     ("n_steps", "n_steps"), ("batch_size", "batch_size"), ("n_epochs", "n_epochs"),
                     ("ent_coef", "ent_coef")):
        if getattr(args, src) is not None:
            hp[dst] = getattr(args, src)
    if args.net_arch:
        hp["net_arch"] = [int(x) for x in args.net_arch.split(",")]
    name = args.name or f"ppo_s{args.seed}"
    base = cfg.get("splits", {}).get("train_seed_base", 0)

    def cb(d):
        if d["done"] or d["episode"] % 500 == 0:
            print(f"ep {d['episode']:>6} steps {d['timestep']:>8} reward_ma {d['reward_ma']:8.2f} "
                  f"success_ma {d['success_ma']:.2f}", flush=True)

    out = train(env_config, hp, args.seed, name, progress_callback=cb, models_dir=args.models_dir,
                results_dir=args.results_dir, device=args.device, train_seed_base=base,
                eval_every_steps=args.eval_every_steps, select=args.select, tensorboard=not args.no_tensorboard)
    print(out)
    return out


if __name__ == "__main__":
    main()
