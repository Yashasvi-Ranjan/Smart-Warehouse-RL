"""Export and discovery of real result files (no data is ever fabricated)."""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from typing import Optional, Union

import numpy as np
import pandas as pd

from analytics.metrics import load_train_log

INDEX_COLUMNS = ["run_name", "algo", "seed", "episodes", "timesteps", "final_reward_ma", "final_success_ma",
                 "wall_seconds", "reward_preset", "battery_enabled", "created", "model_path", "log_path"]


def _mk(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def export_results(df: pd.DataFrame, path: str) -> str:
    """Write an episode-level frame to CSV. Raises ValueError on an empty frame."""
    if df is None or len(df) == 0:
        raise ValueError("export_results: nothing to export (empty frame)")
    _mk(path)
    df.to_csv(path, index=False)
    return path


def export_summary(summary: Union[pd.DataFrame, dict], path: str) -> str:
    """Write a summary (DataFrame -> CSV with index, or dict) to ``.csv`` or ``.json`` by extension."""
    if summary is None or len(summary) == 0:
        raise ValueError("export_summary: nothing to export (empty summary)")
    _mk(path)
    if path.lower().endswith(".json"):
        if isinstance(summary, pd.DataFrame):
            payload = json.loads(summary.reset_index().to_json(orient="records"))
        else:
            payload = _jsonable(summary)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    else:
        df = summary if isinstance(summary, pd.DataFrame) else pd.DataFrame([summary])
        df.to_csv(path, index=isinstance(summary, pd.DataFrame))
    return path


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def export_run_metadata(meta: dict, path: str) -> str:
    """Write run metadata JSON (adds an ISO timestamp under ``exported`` if absent)."""
    meta = dict(meta)
    meta.setdefault("exported", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    _mk(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(meta), f, indent=2)
    return path


def discover_runs(results_dir: str = "experiments/results", models_dir: str = "models", ma_window: int = 100) -> pd.DataFrame:
    """Index of runs: ``results_dir/*/train_log.csv`` joined with ``models_dir/**/*.meta.json`` by run name.

    Columns: ``INDEX_COLUMNS``. Returns an empty frame (with those columns) when nothing exists.
    """
    metas = {}
    for mp in glob.glob(os.path.join(models_dir, "**", "*.meta.json"), recursive=True):
        try:
            with open(mp, encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, ValueError):
            continue
        name = m.get("run_name") or os.path.basename(mp)[: -len(".meta.json")]
        stem = mp[: -len(".meta.json")]
        model = next((stem + e for e in (".npz", ".pt", ".zip") if os.path.exists(stem + e)), None)
        metas[name] = (m, model)
    rows = []
    names = set(metas)
    for lp in glob.glob(os.path.join(results_dir, "*", "train_log.csv")):
        names.add(os.path.basename(os.path.dirname(lp)))
    for name in sorted(names):
        m, model = metas.get(name, ({}, None))
        lp = os.path.join(results_dir, name, "train_log.csv")
        row = {c: None for c in INDEX_COLUMNS}
        row.update(run_name=name, algo=m.get("algo"), model_path=model,
                   log_path=lp if os.path.isfile(lp) else None, created=m.get("created"))
        tr = m.get("train", {}) or {}
        row["seed"] = tr.get("seed")
        row["wall_seconds"] = tr.get("wall_seconds")
        row["episodes"] = tr.get("episodes")
        row["timesteps"] = tr.get("timesteps")
        rw = m.get("reward", {}) or {}
        row["reward_preset"] = rw.get("preset") if isinstance(rw, dict) else rw
        row["battery_enabled"] = (m.get("env_config", {}) or {}).get("battery_enabled")
        if row["log_path"]:
            try:
                log = load_train_log(lp)
                row["episodes"] = int(len(log)) if row["episodes"] is None else row["episodes"]
                if "timestep" in log.columns and len(log):
                    row["timesteps"] = row["timesteps"] if row["timesteps"] is not None else int(log["timestep"].iloc[-1])
                tail = log.tail(ma_window)
                row["final_reward_ma"] = float(tail["reward"].mean())
                row["final_success_ma"] = float(tail["success"].astype(float).mean())
            except (ValueError, OSError, pd.errors.ParserError):
                pass
        if row["algo"] is None and row["log_path"] is None:
            continue
        rows.append(row)
    return pd.DataFrame(rows, columns=INDEX_COLUMNS)


def write_tensorboard(log_csv: str, logdir: str) -> str:
    """Convert a ``train_log.csv`` into TensorBoard scalars ``reward``, ``success``, ``epsilon``, ``loss``
    (x-axis = episode; empty epsilon/loss cells are skipped). Returns ``logdir``.
    Raises FileNotFoundError/ValueError if the log is missing or empty."""
    from torch.utils.tensorboard import SummaryWriter
    log = load_train_log(log_csv)
    if len(log) == 0:
        raise ValueError(f"{log_csv} has no rows")
    os.makedirs(logdir, exist_ok=True)
    w = SummaryWriter(log_dir=logdir)
    try:
        for _, r in log.iterrows():
            ep = int(r["episode"])
            for col, tag in (("reward", "reward"), ("success", "success"), ("epsilon", "epsilon"), ("loss", "loss"),
                             ("steps", "steps"), ("collisions", "collisions")):
                if col in log.columns and pd.notna(r[col]):
                    w.add_scalar(tag, float(r[col]), ep)
    finally:
        w.close()
    return logdir
