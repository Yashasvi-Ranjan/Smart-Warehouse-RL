"""Train the from-scratch DQN agent.  CLI: ``python -m training.train_dqn``.

Seeds: training episode i of run ``seed`` uses ``env.reset(seed=TRAIN_SEED_OFFSET + base + seed*SEED_STRIDE + i)``
(same scheme as Q-learning, >= 1e9, disjoint from the evaluation seeds 100000+).  Model selection uses a small
*held-out* set of seeds (``HELD_OUT_BASE + k``) drawn from the TRAINING distribution (same ``layout_seeds`` /
random-layout mode as training); the test seeds / unseen layouts are never touched during training.

Reproducibility (CPU): numpy/random/torch are seeded, ``torch.use_deterministic_algorithms(True)`` is on, and
torch uses one thread, so two runs with the same seed and config give identical weights. MPS (``--device mps``)
is not guaranteed bit-reproducible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import torch

from agents.dqn_agent import DQNAgent
from environment import EnvConfig, WarehouseEnv
from training.train_qlearning import SEED_STRIDE, TRAIN_SEED_OFFSET, train_episode_seed

LOG_COLUMNS = ["episode", "timestep", "reward", "steps", "success", "collisions", "energy_used",
               "epsilon", "loss"]
HELD_OUT_BASE = 500_000_000  # held-out model-selection seeds (training distribution), < 1e9 and > eval seeds

DEFAULT_HPARAMS = {"episodes": 3000, "gamma": 0.99, "lr": 1e-3, "batch_size": 64, "buffer_size": 100_000,
                   "learning_starts": 1000, "target_update_every": 500, "tau": None, "epsilon_start": 1.0,
                   "epsilon_end": 0.05, "epsilon_decay_steps": 60_000, "hidden": [128, 128],
                   "double_dqn": False, "grad_clip": 10.0, "loss": "huber", "train_freq": 1,
                   "use_action_mask": False}
TRAINER_KEYS = {"episodes"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:  # pragma: no cover
        pass


def greedy_eval(agent, env, seeds) -> dict:
    """Greedy rollouts on ``seeds`` (no learning). Returns mean success / reward."""
    succ, rew = [], []
    for s in seeds:
        obs, info = env.reset(seed=int(s))
        while True:
            a = agent.act(obs, env=env, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                st = info["episode_stats"]
                succ.append(float(st["success"]))
                rew.append(st["reward"])
                break
    return {"success": float(np.mean(succ)), "reward": float(np.mean(rew))}


def train(env_config: EnvConfig, hparams: Optional[dict] = None, seed: int = 0, run_name: str = "dqn",
          progress_callback: Optional[Callable[[dict], None]] = None,
          stop_event: Optional[threading.Event] = None,
          models_dir: str = "models", results_dir: str = "experiments/results",
          device: str = "cpu", train_seed_base: int = 0, callback_every: int = 50, ma_window: int = 100,
          eval_every: int = 100, eval_episodes: int = 30, checkpoint_every: int = 200,
          select: str = "best", resume_from: Optional[str] = None, start_episode: int = 0,
          tensorboard: bool = True) -> dict:
    """Train DQN. ``select``: 'best' (save best-by-held-out-greedy-eval as the final model) or 'last'."""
    hp = dict(DEFAULT_HPARAMS)
    if hparams:
        unknown = set(hparams) - set(DEFAULT_HPARAMS)
        if unknown:
            raise ValueError(f"unknown DQN hyperparameters: {sorted(unknown)}")
        hp.update({k: v for k, v in hparams.items()})
    episodes = int(hp.pop("episodes"))
    seed_everything(seed)
    env = WarehouseEnv(env_config)
    eval_env = WarehouseEnv(env_config)
    obs_dim = env.observation_space.shape[0]
    if resume_from:
        agent = DQNAgent.load(resume_from, device=device)
        start_episode = int(getattr(agent, "loaded_extra", {}).get("episode", start_episode))
        agent.q_net.train()
    else:
        agent = DQNAgent(obs_dim=obs_dim, seed=seed, device=device, **hp)
    agent.make_buffer()
    use_mask = agent.use_action_mask

    run_dir = os.path.join(results_dir, run_name)
    ck_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    os.makedirs(os.path.join(models_dir, "dqn"), exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.csv")
    model_path = os.path.join(models_dir, "dqn", f"{run_name}.pt")
    meta_path = os.path.join(models_dir, "dqn", f"{run_name}.meta.json")
    writer = None
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(run_dir, "tb"))

    held_out = [HELD_OUT_BASE + k for k in range(eval_episodes)]
    best = {"success": -1.0, "reward": -1e18, "episode": -1}
    rew_ma, suc_ma = deque(maxlen=ma_window), deque(maxlen=ma_window)
    evals = []
    t0 = time.time()
    timestep = agent.env_steps
    ep_done = start_episode
    last_loss = None
    best_path = os.path.join(ck_dir, "best.pt")
    with open(log_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(LOG_COLUMNS)
        for ep in range(start_episode, start_episode + episodes):
            if stop_event is not None and stop_event.is_set():
                break
            obs, info = env.reset(seed=train_episode_seed(seed, ep, train_seed_base))
            mask = info["action_mask"] if use_mask else None
            ep_r, losses = 0.0, []
            while True:
                eps = agent.epsilon_at(agent.env_steps)
                a = agent.select_action(obs, eps, mask)
                nobs, r, term, trunc, info = env.step(a)
                nmask = info["action_mask"]
                loss = agent.observe(obs, a, r, nobs, term, nmask)
                if loss is not None:
                    losses.append(loss)
                obs, mask = nobs, (nmask if use_mask else None)
                ep_r += r
                if term or trunc:
                    break
            st = info["episode_stats"]
            ep_done = ep + 1
            timestep = agent.env_steps
            eps_now = agent.epsilon
            mloss = float(np.mean(losses)) if losses else None
            if mloss is not None:
                last_loss = mloss
            rew_ma.append(ep_r)
            suc_ma.append(float(st["success"]))
            w.writerow([ep, timestep, round(ep_r, 4), st["steps"], int(st["success"]), st["collisions"],
                        st["energy_used"], round(eps_now, 5), "" if mloss is None else round(mloss, 5)])
            if writer is not None:
                writer.add_scalar("train/reward", ep_r, ep)
                writer.add_scalar("train/success", float(st["success"]), ep)
                writer.add_scalar("train/epsilon", eps_now, ep)
                if mloss is not None:
                    writer.add_scalar("train/loss", mloss, ep)
            if eval_every and ep_done % eval_every == 0:
                agent.q_net.eval()
                ev = greedy_eval(agent, eval_env, held_out)
                agent.q_net.train()
                evals.append({"episode": ep_done, **ev})
                if writer is not None:
                    writer.add_scalar("eval/success", ev["success"], ep_done)
                    writer.add_scalar("eval/reward", ev["reward"], ep_done)
                if (ev["success"], ev["reward"]) > (best["success"], best["reward"]):
                    best = {**ev, "episode": ep_done}
                    agent.save(best_path, extra={"episode": ep_done})
            if checkpoint_every and ep_done % checkpoint_every == 0:
                agent.save(os.path.join(ck_dir, "latest.pt"), extra={"episode": ep_done})
            if progress_callback is not None and ep_done % callback_every == 0:
                progress_callback({"episode": ep_done, "timestep": timestep, "reward_ma": float(np.mean(rew_ma)),
                                   "success_ma": float(np.mean(suc_ma)), "epsilon": eps_now,
                                   "loss": last_loss, "done": False})
    wall = time.time() - t0
    agent.save(os.path.join(ck_dir, "latest.pt"), extra={"episode": ep_done})
    # final eval of the last weights, then pick the model to ship
    agent.q_net.eval()
    final_ev = greedy_eval(agent, eval_env, held_out)
    final_agent = agent
    chosen = "last"
    if select == "best" and best["episode"] >= 0 and (best["success"], best["reward"]) > (
            final_ev["success"], final_ev["reward"]):
        final_agent = DQNAgent.load(best_path, device=device)
        chosen = f"best@{best['episode']}"
    meta = {"algo": "dqn", "run_name": run_name, "env_config": env_config.to_dict(),
            "reward": env_config.reward.to_dict(),
            "hyperparams": {**agent.hparams(), "episodes_planned": episodes, "train_seed_base": train_seed_base,
                            "train_seed_offset": TRAIN_SEED_OFFSET, "seed_stride": SEED_STRIDE,
                            "reward_scale": 1.0, "device": str(device)},
            "train": {"seed": seed, "episodes": ep_done, "timesteps": timestep, "wall_seconds": wall,
                      "grad_steps": agent.grad_steps, "selected_model": chosen,
                      "held_out_eval": {"seed_base": HELD_OUT_BASE, "n": eval_episodes,
                                        "best": best, "final": final_ev}},
            "created": datetime.now().isoformat(timespec="seconds")}
    final_agent.save(model_path, meta=meta, extra={"episode": ep_done})
    if progress_callback is not None:
        progress_callback({"episode": ep_done, "timestep": timestep,
                           "reward_ma": float(np.mean(rew_ma)) if rew_ma else 0.0,
                           "success_ma": float(np.mean(suc_ma)) if suc_ma else 0.0,
                           "epsilon": agent.epsilon, "loss": last_loss, "done": True})
    with open(os.path.join(run_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    if writer is not None:
        writer.close()
    return {"model_path": model_path, "meta_path": meta_path, "log_path": log_path, "wall_seconds": wall}


def build_env_config(args, cfg: dict) -> EnvConfig:
    """Shared by train_dqn / train_ppo CLIs."""
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
    return EnvConfig.from_dict(env_d)


def main(argv=None) -> dict:
    from environment.config import load_config
    p = argparse.ArgumentParser(description="Train DQN on the Smart Warehouse env")
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--reward-preset", default=None)
    p.add_argument("--no-battery", action="store_true")
    p.add_argument("--layout-seeds", default=None, help="comma separated, 'train' = config split; omit = config")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--buffer-size", type=int, default=None)
    p.add_argument("--learning-starts", type=int, default=None)
    p.add_argument("--target-update-every", type=int, default=None)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--epsilon-end", type=float, default=None)
    p.add_argument("--epsilon-decay-steps", type=int, default=None)
    p.add_argument("--hidden", default=None, help="comma separated, e.g. 128,128")
    p.add_argument("--train-freq", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--loss", choices=["huber", "mse"], default=None)
    p.add_argument("--double", action="store_true", help="Double DQN")
    p.add_argument("--mask", action="store_true", help="use action masking")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--select", choices=["best", "last"], default="best")
    p.add_argument("--resume", default=None, help="checkpoint .pt to resume from")
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--results-dir", default="experiments/results")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    env_config = build_env_config(args, cfg)
    hp = dict(cfg.get("dqn", {}))
    for k in ("episodes", "lr", "gamma", "batch_size", "buffer_size", "learning_starts", "target_update_every",
              "tau", "epsilon_end", "epsilon_decay_steps", "train_freq", "grad_clip", "loss"):
        if getattr(args, k) is not None:
            hp[k] = getattr(args, k)
    if args.hidden:
        hp["hidden"] = [int(x) for x in args.hidden.split(",")]
    if args.double:
        hp["double_dqn"] = True
    if args.mask:
        hp["use_action_mask"] = True
    name = args.name or f"dqn_s{args.seed}"
    base = cfg.get("splits", {}).get("train_seed_base", 0)

    def cb(d):
        if d["done"] or d["episode"] % 250 == 0:
            print(f"ep {d['episode']:>6} steps {d['timestep']:>8} reward_ma {d['reward_ma']:8.2f} "
                  f"success_ma {d['success_ma']:.2f} eps {d['epsilon']:.3f} loss {d['loss']}", flush=True)

    out = train(env_config, hp, args.seed, name, progress_callback=cb, models_dir=args.models_dir,
                results_dir=args.results_dir, device=args.device, train_seed_base=base,
                eval_every=args.eval_every, eval_episodes=args.eval_episodes, select=args.select,
                resume_from=args.resume, tensorboard=not args.no_tensorboard)
    print(out)
    return out


if __name__ == "__main__":
    main()
