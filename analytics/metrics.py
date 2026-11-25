"""Episode-level evaluation and summary statistics.

Fairness rule: every agent being compared must be evaluated on the *same* ``seeds`` (and
layouts). ``env.reset(seed=s)`` fully determines layout + task (INTERFACES.md section 1), so
identical seeds give every agent identical episodes. Use seeds starting at
``splits.eval_seed_base`` (disjoint from training seeds).

No module-level mutable state: every function builds its own env and returns fresh objects.
"""
from __future__ import annotations

import dataclasses
import math
import os
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

EPISODE_COLUMNS = ["agent", "label", "seed", "layout_seed", "reward", "success", "steps",
                   "optimal_steps", "collisions", "invalid_actions", "energy_used", "recharges",
                   "picked_up", "battery_depleted", "timeout"]
TRAIN_LOG_COLUMNS = ["episode", "timestep", "reward", "steps", "success", "collisions",
                     "energy_used", "epsilon", "loss"]


# ----------------------------------------------------------------------------- helpers
def _as_env_config(env_config):
    from environment import EnvConfig
    if env_config is None:
        return EnvConfig()
    if isinstance(env_config, dict):
        return EnvConfig.from_dict(env_config)
    return env_config


def _with_layouts(env_config, layout_seeds):
    cfg = _as_env_config(env_config)
    if layout_seeds is None:
        return cfg
    return dataclasses.replace(cfg, layout_seeds=[int(s) for s in layout_seeds])


def moving_average(x: Sequence[float], window: int = 100) -> np.ndarray:
    """Trailing moving average; the first ``window-1`` points average what is available."""
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return a
    window = max(1, min(int(window), a.size))
    c = np.cumsum(np.insert(np.nan_to_num(a), 0, 0.0))
    out = np.empty(a.size)
    idx = np.arange(a.size)
    lo = np.maximum(0, idx - window + 1)
    out[:] = (c[idx + 1] - c[lo]) / (idx + 1 - lo)
    return out


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion. Returns (nan, nan) when n == 0."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values: Sequence[float], stat=np.mean, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple:
    """Percentile bootstrap CI of ``stat`` (own RandomState, so no global RNG use)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, v.size, size=(n_boot, v.size))
    boots = np.array([stat(v[i]) for i in idx])
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


# ----------------------------------------------------------------------------- episodes
def run_episode(agent, env, seed: Optional[int], options: Optional[dict] = None,
                record_trajectory: bool = False, deterministic: bool = True):
    """Run one episode. Returns ``(episode_stats, trajectory | None)``.

    ``episode_stats`` is the env's ``info["episode_stats"]``. The trajectory (only when requested)
    is a list of dicts: ``{"t", "obs", "action", "reward", "info"}`` per step, preceded by a
    ``t=-1`` entry holding the reset observation/info (action/reward None).
    """
    obs, info = env.reset(seed=seed, options=options)
    agent.reset()
    traj = [] if record_trajectory else None
    if record_trajectory:
        traj.append({"t": -1, "obs": obs, "action": None, "reward": None, "info": info})
    stats = None
    t = 0
    while True:
        action = int(agent.act(obs, env=env, deterministic=deterministic))
        obs, reward, terminated, truncated, info = env.step(action)
        if record_trajectory:
            traj.append({"t": t, "obs": obs, "action": action, "reward": float(reward), "info": info})
        t += 1
        if terminated or truncated:
            stats = info.get("episode_stats")
            break
    if stats is None:  # pragma: no cover - env contract guarantees it
        raise RuntimeError("env did not provide info['episode_stats'] on the final step")
    return dict(stats), traj


def evaluate_agent(agent, env_config, seeds: Iterable[int], layout_seeds: Optional[Sequence[int]] = None,
                   label: Optional[str] = None) -> pd.DataFrame:
    """One episode per seed (deterministic policy). Columns: ``EPISODE_COLUMNS``.

    ``layout_seeds`` (when not None) overrides ``env_config.layout_seeds``. ``agent`` is the
    agent's ``name``; ``label`` defaults to the same unless given.
    """
    from environment import WarehouseEnv
    cfg = _with_layouts(env_config, layout_seeds)
    env = WarehouseEnv(cfg)
    name = str(getattr(agent, "name", type(agent).__name__))
    label = name if label is None else str(label)
    rows = []
    try:
        for s in seeds:
            st, _ = run_episode(agent, env, int(s))
            rows.append({
                "agent": name, "label": label, "seed": int(s), "layout_seed": int(st["layout_seed"]),
                "reward": float(st["reward"]), "success": bool(st["success"]), "steps": int(st["steps"]),
                "optimal_steps": (np.nan if st.get("optimal_steps") is None else float(st["optimal_steps"])),
                "collisions": int(st["collisions"]), "invalid_actions": int(st["invalid_actions"]),
                "energy_used": int(st["energy_used"]), "recharges": int(st["recharges"]),
                "picked_up": bool(st["picked_up"]), "battery_depleted": bool(st["battery_depleted"]),
                "timeout": bool(st["timeout"]),
            })
    finally:
        env.close()
    return pd.DataFrame(rows, columns=EPISODE_COLUMNS)


def _nan_mean(a) -> float:
    a = np.asarray(a, dtype=float)
    return float(a.mean()) if a.size else float("nan")


def _nan_std(a) -> float:
    a = np.asarray(a, dtype=float)
    return float(a.std(ddof=1)) if a.size > 1 else (0.0 if a.size == 1 else float("nan"))


def summarize(df: pd.DataFrame) -> dict:
    """Aggregate an episode frame. Per-episode std (``*_std``) for episode-level spreads.

    steps_per_delivery / energy_per_delivery / path_efficiency are over *successful* episodes only
    (NaN when there are none). Empty input gives ``n == 0`` and NaN metrics (nothing is invented).
    """
    n = int(len(df))
    if n == 0:
        keys = ["reward_mean", "reward_std", "success_rate", "success_ci_low", "success_ci_high",
                "steps_per_delivery", "steps_per_delivery_std", "collisions_mean", "collisions_std",
                "energy_per_delivery", "energy_per_delivery_std", "path_efficiency", "path_efficiency_std",
                "invalid_actions_mean", "battery_depleted_rate", "timeout_rate", "recharges_mean"]
        return {**{k: float("nan") for k in keys}, "n": 0, "n_success": 0}
    succ = df[df["success"].astype(bool)]
    ns = int(len(succ))
    lo, hi = wilson_interval(ns, n)
    eff = (succ["optimal_steps"] / succ["steps"].replace(0, np.nan)).to_numpy(dtype=float) if ns else np.array([])
    eff = eff[~np.isnan(eff)]
    return {
        "n": n, "n_success": ns,
        "reward_mean": _nan_mean(df["reward"]), "reward_std": _nan_std(df["reward"]),
        "success_rate": ns / n, "success_ci_low": lo, "success_ci_high": hi,
        "steps_per_delivery": _nan_mean(succ["steps"]), "steps_per_delivery_std": _nan_std(succ["steps"]),
        "collisions_mean": _nan_mean(df["collisions"]), "collisions_std": _nan_std(df["collisions"]),
        "energy_per_delivery": _nan_mean(succ["energy_used"]), "energy_per_delivery_std": _nan_std(succ["energy_used"]),
        "path_efficiency": _nan_mean(eff), "path_efficiency_std": _nan_std(eff),
        "invalid_actions_mean": _nan_mean(df["invalid_actions"]),
        "recharges_mean": _nan_mean(df["recharges"]),
        "battery_depleted_rate": float(df["battery_depleted"].astype(bool).mean()),
        "timeout_rate": float(df["timeout"].astype(bool).mean()),
    }


# Keys that are averaged over independent training runs (standard "mean" metrics)
RUN_METRIC_KEYS = ["reward_mean", "success_rate", "steps_per_delivery", "collisions_mean",
                   "energy_per_delivery", "path_efficiency", "invalid_actions_mean",
                   "battery_depleted_rate", "timeout_rate", "recharges_mean"]


def aggregate_over_seeds(items: Iterable[Any]) -> dict:
    """Mean and std across *independent training runs*.

    ``items``: DataFrames (episode frames of one run each) and/or summary dicts. Returns the same keys as
    ``summarize`` for the means (``reward_mean``, ``success_rate``...), plus ``<key>_std_runs`` (sample
    std, ddof=1, across runs; 0.0 for one run), ``n_runs`` and ``n`` (total episodes).
    """
    summaries = [summarize(x) if isinstance(x, pd.DataFrame) else dict(x) for x in items]
    if not summaries:
        raise ValueError("aggregate_over_seeds needs at least one run")
    out: dict = {"n_runs": len(summaries), "n": int(sum(s.get("n", 0) for s in summaries))}
    for k in RUN_METRIC_KEYS:
        vals = np.array([s.get(k, np.nan) for s in summaries], dtype=float)
        vals = vals[~np.isnan(vals)]
        out[k] = float(vals.mean()) if vals.size else float("nan")
        out[k + "_std_runs"] = _nan_std(vals) if vals.size else float("nan")
    return out


def evaluate_generalization(agent, env_config, train_layout_seeds: Sequence[int],
                            test_layout_seeds: Sequence[int], seeds: Iterable[int],
                            label: Optional[str] = None) -> dict:
    """Evaluate on training layouts and on unseen layouts with the same episode seeds.

    Returns ``{"train": df, "test": df, "summary": {"train": {...}, "test": {...},
    "gap": {metric: test - train}}}``.
    """
    seeds = list(seeds)
    tr = evaluate_agent(agent, env_config, seeds, layout_seeds=train_layout_seeds, label=label)
    te = evaluate_agent(agent, env_config, seeds, layout_seeds=test_layout_seeds, label=label)
    s_tr, s_te = summarize(tr), summarize(te)
    gap = {k: s_te[k] - s_tr[k] for k in RUN_METRIC_KEYS}
    return {"train": tr, "test": te, "summary": {"train": s_tr, "test": s_te, "gap": gap}}


# ----------------------------------------------------------------------------- training logs
def load_train_log(run_name_or_path: str, results_dir: str = "experiments/results") -> pd.DataFrame:
    """Load ``train_log.csv`` from a csv path, a run directory or a run name under ``results_dir``.

    Raises FileNotFoundError when it does not exist (never returns invented data).
    """
    p = str(run_name_or_path)
    cands = [p, os.path.join(p, "train_log.csv"), os.path.join(results_dir, p, "train_log.csv")]
    for c in cands:
        if os.path.isfile(c):
            df = pd.read_csv(c)
            missing = [col for col in ("episode", "reward", "success") if col not in df.columns]
            if missing:
                raise ValueError(f"{c} lacks columns {missing}")
            return df
    raise FileNotFoundError(f"no train_log.csv for {run_name_or_path!r} (tried {cands})")
