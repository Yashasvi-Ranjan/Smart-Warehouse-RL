"""Non-UI logic: simulation controller, model discovery, background training / evaluation jobs.

Nothing in this module imports pygame, so it can be unit-tested headless.
"""
from __future__ import annotations

import glob
import importlib
import importlib.util
import json
import os
import threading
import time
import traceback
from collections import deque
from dataclasses import replace
from typing import Callable, Optional

import numpy as np

from dashboard.theme import ROOT
from environment import EnvConfig, WarehouseEnv
from environment.config import load_config
from environment.rewards import REWARD_PRESETS

MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "experiments", "results")
CONFIG_PATH = os.path.join(ROOT, "config.yaml")

ALGO_LABELS = {"random": "Random", "astar": "A* (privileged)", "qlearning": "Q-Learning", "dqn": "DQN", "ppo": "PPO"}
ACTION_NAMES = ("up", "down", "left", "right", "pickup", "deliver", "recharge")
MODEL_EXTS = (".npz", ".pt", ".zip")
TRAINER_MODULES = {"qlearning": "training.train_qlearning", "dqn": "training.train_dqn", "ppo": "training.train_ppo"}
AGENT_MODULES = {"random": "agents.random_agent", "astar": "agents.astar_agent", "qlearning": "agents.q_learning",
                 "dqn": "agents.dqn_agent", "ppo": "agents.ppo_agent"}


def load_app_config() -> dict:
    try:
        return load_config(CONFIG_PATH)
    except Exception:
        return {}


def reward_presets() -> list:
    return list(REWARD_PRESETS.keys())


def module_available(name: str) -> tuple[bool, str]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return False, f"module {name} not found"
    return True, ""


def list_models(models_dir: str = MODELS_DIR) -> list[dict]:
    """Saved models found under ``models_dir/**`` via their ``*.meta.json``, newest first."""
    out = []
    for mp in glob.glob(os.path.join(models_dir, "**", "*.meta.json"), recursive=True):
        stem = mp[: -len(".meta.json")]
        model = next((stem + e for e in MODEL_EXTS if os.path.exists(stem + e)), None)
        if model is None:
            continue
        try:
            with open(mp, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        tr = meta.get("train", {}) or {}
        rw = meta.get("reward", {}) or {}
        out.append({
            "path": model, "algo": meta.get("algo", "?"), "run_name": meta.get("run_name", os.path.basename(stem)),
            "created": meta.get("created", ""), "meta": meta,
            "budget": tr.get("episodes") or tr.get("timesteps"),
            "battery": (meta.get("env_config", {}) or {}).get("battery_enabled"),
            "layouts": (meta.get("env_config", {}) or {}).get("layout_seeds"),
            "shaping": rw.get("shaping") if isinstance(rw, dict) else None,
        })
    out.sort(key=lambda m: m["created"], reverse=True)
    return out


def model_summary(m: dict) -> str:
    bits = []
    if m.get("budget"):
        bits.append(f"{m['budget']} {'ep' if m['algo'] != 'ppo' else 'steps'}")
    if m.get("battery") is not None:
        bits.append("battery" if m["battery"] else "no battery")
    lay = m.get("layouts")
    bits.append("random layouts" if lay is None else f"layouts {','.join(map(str, lay[:4]))}{'..' if len(lay) > 4 else ''}")
    if m.get("created"):
        bits.append(str(m["created"])[:10])
    return "  ".join(bits)


def algo_availability(models_dir: str = MODELS_DIR) -> dict:
    """algo -> (ok, reason). Learners need their module (and framework) plus at least one saved model."""
    res = {}
    models = list_models(models_dir)
    for algo, mod in AGENT_MODULES.items():
        ok, why = module_available(mod)
        if ok and algo == "dqn":
            ok, why = module_available("torch")
        if ok and algo in ("qlearning", "dqn", "ppo") and not any(m["algo"] == algo for m in models):
            ok, why = False, "no saved model; train one in the Training tab"
        res[algo] = (ok, why)
    return res


def load_agent_from(algo: str, model_path: Optional[str] = None, seed: Optional[int] = None):
    """Return an agent instance (raises with a readable message when unavailable)."""
    if algo in ("random", "astar"):
        from agents import make_agent
        return make_agent(algo, seed=seed) if algo == "random" else make_agent(algo)
    if not model_path:
        raise RuntimeError("no model selected")
    from agents import load_agent
    return load_agent(model_path)


def build_env_config(base: Optional[dict] = None, battery: Optional[bool] = None, reward: Optional[str] = None,
                     layout_seeds=None, **over) -> EnvConfig:
    cfg = dict(base if base is not None else (load_app_config().get("env") or {}))
    cfg.pop("reward", None)
    if battery is not None:
        cfg["battery_enabled"] = bool(battery)
    cfg["layout_seeds"] = None if layout_seeds is None else list(layout_seeds)
    cfg.update(over)
    cfg["reward"] = reward or (load_app_config().get("reward") or {}).get("preset", "default")
    return EnvConfig.from_dict(cfg)


# ================================================================== simulation
class SimulationController:
    """Owns the env + agent for the Simulation tab. ``step()`` runs the agent's real policy through env.step."""

    HISTORY = 60

    def __init__(self, battery: bool = True, reward: str = "default", layout_seed: Optional[int] = 0,
                 base_env_cfg: Optional[dict] = None, task_seed_base: int = 100000, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.base_env_cfg = base_env_cfg
        self.battery, self.reward = battery, reward
        self.layout_seed = layout_seed      # None -> random layout derived from the task seed
        self.layout_seeds_set: Optional[list] = None  # replay of an evaluation layout set
        self.task_seed = int(task_seed_base)
        self.agent_key = "random"
        self.agent = None
        self.agent_label = "Random"
        self.model_path: Optional[str] = None
        self.model_meta: Optional[dict] = None
        self.agent_error = ""
        self.history: deque = deque(maxlen=self.HISTORY)
        self.episode_no = 0
        self.env: Optional[WarehouseEnv] = None
        self._build_env()
        self.set_agent("random")
        self.reset()

    # -------------------------------------------------- config
    def _build_env(self):
        seeds = self.layout_seeds_set
        self.env_config = build_env_config(self.base_env_cfg, self.battery, self.reward, layout_seeds=seeds)
        if self.env is not None:
            self.env.close()
        self.env = WarehouseEnv(self.env_config)

    def configure(self, battery=None, reward=None, layout_seed="keep", layout_seeds_set="keep"):
        if battery is not None:
            self.battery = bool(battery)
        if reward is not None:
            self.reward = reward
        if layout_seed != "keep":
            self.layout_seed = layout_seed
        if layout_seeds_set != "keep":
            self.layout_seeds_set = layout_seeds_set
        self._build_env()
        self.reset()

    def set_agent(self, algo: str, model_path: Optional[str] = None) -> bool:
        """Select an agent. Returns False and sets ``agent_error`` when it cannot be used."""
        self.agent_error = ""
        try:
            if algo in ("qlearning", "dqn", "ppo") and model_path is None:
                cands = [m for m in list_models(self.models_dir) if m["algo"] == algo]
                if not cands:
                    raise RuntimeError("no saved model; train one in the Training tab")
                model_path = cands[0]["path"]
            ok, why = module_available(AGENT_MODULES[algo])
            if not ok:
                raise RuntimeError(why)
            self.agent = load_agent_from(algo, model_path)
            self.model_meta = None
            if model_path:
                from agents import read_meta
                self.model_meta = read_meta(model_path)
            self.agent_key, self.model_path = algo, model_path
            self.agent_label = ALGO_LABELS[algo]
            if self.model_meta:
                self.agent_label += f" ({self.model_meta.get('run_name', os.path.basename(model_path))})"
            self.agent.reset()
            return True
        except Exception as ex:  # unavailable algorithm: keep previous agent, report reason
            self.agent_error = f"{ALGO_LABELS.get(algo, algo)} unavailable: {ex}"
            if self.agent is None:
                self.agent = load_agent_from("random")
                self.agent_key, self.agent_label = "random", "Random"
            return False

    def set_action_mask(self, on: bool):
        if hasattr(self.agent, "use_action_mask"):
            self.agent.use_action_mask = bool(on)

    # -------------------------------------------------- episode
    def reset(self, new_task: bool = False, task_seed: Optional[int] = None):
        if task_seed is not None:
            self.task_seed = int(task_seed)
        elif new_task:
            self.task_seed += 1
        opts = {"layout_seed": self.layout_seed} if (self.layout_seed is not None and self.layout_seeds_set is None) else None
        self.obs, info = self.env.reset(seed=self.task_seed, options=opts)
        self.agent.reset()
        self.info = info
        self.done = False
        self.steps = 0
        self.ep_reward = 0.0
        self.last_reward = 0.0
        self.last_action: Optional[int] = None
        self.last_event = "reset"
        self.collisions = self.invalid = self.energy = 0
        self.picked = self.delivered = False
        self.stats: Optional[dict] = None
        self.result = ""
        self.trail = [tuple(info["robot_pos"])]
        return info

    def step(self) -> dict:
        if self.done:
            raise RuntimeError("episode finished; call reset()")
        a = int(self.agent.act(self.obs, env=self.env, deterministic=True))
        prev = self.env.robot_pos
        self.obs, r, term, trunc, info = self.env.step(a)
        ev = info.get("action_event", info["event"])
        self.steps = info["steps"]
        self.last_action, self.last_reward, self.last_event = a, float(r), ev
        self.ep_reward += float(r)
        if ev == "collision":
            self.collisions += 1
        elif ev.startswith("invalid"):
            self.invalid += 1
        if tuple(info["robot_pos"]) != tuple(prev):
            self.energy += 1
            self.trail.append(tuple(info["robot_pos"]))
        if ev == "pickup":
            self.picked = True
        if ev == "deliver":
            self.delivered = True
        self.info = info
        if term or trunc:
            self.done = True
            self.stats = dict(info["episode_stats"])
            st = self.stats
            self.result = ("delivered" if st["success"] else "battery depleted" if st["battery_depleted"]
                           else "timeout" if st["timeout"] else "ended")
            self.ep_reward, self.collisions = st["reward"], st["collisions"]
            self.invalid, self.energy = st["invalid_actions"], st["energy_used"]
            self.episode_no += 1
            self.history.append({"n": self.episode_no, "result": self.result, "reward": st["reward"],
                                 "steps": st["steps"], "seed": self.task_seed, "layout": st["layout_seed"],
                                 "agent": self.agent_label})
        return info

    def run_episode(self, max_steps: int = 10_000) -> dict:
        n = 0
        while not self.done and n < max_steps:
            self.step()
            n += 1
        return self.stats

    @property
    def package_state(self) -> str:
        return "delivered" if self.delivered else "picked up" if (self.picked or self.env.carrying) else "waiting"

    def render_state(self) -> dict:
        s = self.env.get_render_state()
        s["trail"] = self.trail
        return s

    def metrics(self) -> dict:
        return {"reward": self.ep_reward, "steps": self.steps, "collisions": self.collisions,
                "invalid_actions": self.invalid, "energy_used": self.energy, "result": self.result}


# ================================================================== background jobs
class _Job:
    status = "idle"          # idle | running | stopping | done | error

    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.error = ""
        self.t0 = 0.0
        self.t1 = 0.0

    @property
    def running(self) -> bool:
        return self.status in ("running", "stopping")

    def stop(self):
        if self.running:
            self.stop_event.set()
            self.status = "stopping"

    @property
    def elapsed(self) -> float:
        return (time.time() if self.running else self.t1) - self.t0 if self.t0 else 0.0

    def join(self, timeout=None):
        if self.thread:
            self.thread.join(timeout)


class TrainingJob(_Job):
    """Runs the real ``training.train_<algo>.train`` in a daemon thread."""

    def __init__(self, algo: str, env_config: EnvConfig, hparams: dict, seed: int, run_name: str,
                 models_dir: str = MODELS_DIR, results_dir: str = RESULTS_DIR, extra: Optional[dict] = None):
        super().__init__()
        self.algo, self.env_config, self.hparams, self.seed, self.run_name = algo, env_config, hparams, seed, run_name
        self.models_dir, self.results_dir, self.extra = models_dir, results_dir, extra or {}
        self.history: list[dict] = []
        self.latest: dict = {}
        self.result: Optional[dict] = None

    def _cb(self, d: dict):
        with self.lock:
            self.latest = dict(d)
            self.history.append(dict(d))

    def start(self):
        try:
            mod = importlib.import_module(TRAINER_MODULES[self.algo])
        except Exception as ex:
            self.status, self.error = "error", f"{ALGO_LABELS[self.algo]} trainer unavailable: {ex}"
            return
        self.status, self.t0 = "running", time.time()
        self.thread = threading.Thread(target=self._run, args=(mod,), daemon=True, name=f"train-{self.algo}")
        self.thread.start()

    def _run(self, mod):
        try:
            self.result = mod.train(self.env_config, dict(self.hparams), self.seed, self.run_name,
                                    progress_callback=self._cb, stop_event=self.stop_event,
                                    models_dir=self.models_dir, results_dir=self.results_dir, **self.extra)
            self.status = "done"
        except Exception as ex:
            self.error = f"{type(ex).__name__}: {ex}"
            self.status = "error"
            self._tb = traceback.format_exc()
        finally:
            self.t1 = time.time()

    def snapshot(self):
        with self.lock:
            return list(self.history), dict(self.latest)


class EvaluationJob(_Job):
    """Evaluates several agents on identical seeds via analytics.metrics.evaluate_agent."""

    CHUNK = 5

    def __init__(self, agents: dict, env_config: EnvConfig, seeds: list, layout_seeds: Optional[list]):
        super().__init__()
        self.agents, self.env_config, self.seeds, self.layout_seeds = agents, env_config, list(seeds), layout_seeds
        self.progress = 0.0
        self.long_df = None
        self.summary = None
        self.current = ""

    def start(self):
        self.status, self.t0 = "running", time.time()
        self.thread = threading.Thread(target=self._run, daemon=True, name="evaluate")
        self.thread.start()

    def _run(self):
        import pandas as pd
        from analytics.metrics import evaluate_agent, summarize
        try:
            frames, rows = [], {}
            total = max(1, len(self.agents) * len(self.seeds))
            done = 0
            for label, agent in self.agents.items():
                self.current = label
                parts = []
                for i in range(0, len(self.seeds), self.CHUNK):
                    if self.stop_event.is_set():
                        break
                    chunk = self.seeds[i:i + self.CHUNK]
                    parts.append(evaluate_agent(agent, self.env_config, chunk, self.layout_seeds, label=label))
                    done += len(chunk)
                    self.progress = done / total
                if parts:
                    d = pd.concat(parts, ignore_index=True)
                    frames.append(d)
                    rows[label] = summarize(d)
                if self.stop_event.is_set():
                    break
            if frames:
                self.long_df = pd.concat(frames, ignore_index=True)
                s = pd.DataFrame.from_dict(rows, orient="index")
                s.index.name = "agent"
                self.summary = s
            self.status = "done"
        except Exception as ex:
            self.error = f"{type(ex).__name__}: {ex}"
            self.status = "error"
        finally:
            self.t1 = time.time()
