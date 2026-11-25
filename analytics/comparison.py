"""Compare agents on identical evaluation episodes (same seeds, same layouts for everyone)."""
from __future__ import annotations

import dataclasses
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from analytics.metrics import (RUN_METRIC_KEYS, _as_env_config, aggregate_over_seeds, evaluate_agent,
                               summarize)


def _summary_frame(rows: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "agent"
    return df


def compare_agents(agents: Mapping[str, object], env_config, seeds: Iterable[int],
                   layout_seeds: Optional[Sequence[int]] = None):
    """Evaluate every agent on the *same* seeds/layouts. Returns ``(long_df, summary_df)``.

    ``long_df`` has the episode columns of ``evaluate_agent`` (``label`` = dict key); ``summary_df`` is
    indexed by the dict key with the columns of ``summarize``.
    """
    seeds = list(seeds)
    frames, rows = [], {}
    for name, agent in agents.items():
        d = evaluate_agent(agent, env_config, seeds, layout_seeds, label=name)
        frames.append(d)
        rows[name] = summarize(d)
    long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return long_df, _summary_frame(rows)


def summarize_runs(long_df: pd.DataFrame, by: Sequence[str] = ("label",),
                   run_col: str = "train_seed") -> pd.DataFrame:
    """Per group: summarize each training run separately, then mean +- std across runs."""
    if long_df.empty:
        return pd.DataFrame()
    by = list(by)
    rows = {}
    for key, g in long_df.groupby(by, sort=False):
        runs = [summarize(r) for _, r in g.groupby(run_col, sort=True)]
        rows[key] = aggregate_over_seeds(runs)
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index = pd.MultiIndex.from_tuples(out.index, names=by) if len(by) > 1 else pd.Index(
        [k if not isinstance(k, tuple) else k[0] for k in out.index], name=by[0])
    return out


def compare_runs(runs: Mapping[str, Mapping[int, object]], env_config, seeds: Iterable[int],
                 layout_seeds: Optional[Sequence[int]] = None):
    """``runs``: ``{algo_label: {training_seed: agent}}``. Every run is evaluated on identical episodes.

    Returns ``(long_df, summary_df)``; ``long_df`` has an extra ``train_seed`` column and summary
    columns hold the mean over training runs plus ``*_std_runs`` (std across independent training
    runs, not across episodes) and ``n_runs``.
    """
    seeds = list(seeds)
    frames = []
    for name, by_seed in runs.items():
        for ts, agent in by_seed.items():
            d = evaluate_agent(agent, env_config, seeds, layout_seeds, label=name)
            d["train_seed"] = int(ts)
            frames.append(d)
    long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return long_df, summarize_runs(long_df, by=("label",))


def paired_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str = "reward",
                      n_boot: int = 5000, seed: int = 0) -> dict:
    """Paired comparison of two agents on the episodes (seeds) they share.

    Returns mean difference (a - b), a bootstrap 95% CI of that mean, a two-sided paired sign-flip
    permutation p-value (null: differences are symmetric around 0), and win/tie/loss counts.
    Pure numpy (scipy is not a dependency). Raises ValueError if no common seeds.
    """
    a = df_a.drop_duplicates("seed").set_index("seed")[metric].astype(float)
    b = df_b.drop_duplicates("seed").set_index("seed")[metric].astype(float)
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise ValueError("no common evaluation seeds between the two frames")
    d = (a.loc[common] - b.loc[common]).to_numpy()
    rng = np.random.RandomState(seed)
    obs = float(d.mean())
    idx = rng.randint(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    signs = rng.choice([-1.0, 1.0], size=(n_boot, d.size))
    perm = (signs * d).mean(axis=1)
    p = float((np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (n_boot + 1))
    return {"metric": metric, "n_pairs": int(d.size), "mean_diff": obs,
            "ci_low": float(np.quantile(boots, 0.025)), "ci_high": float(np.quantile(boots, 0.975)),
            "p_value": p, "wins_a": int((d > 0).sum()), "ties": int((d == 0).sum()),
            "wins_b": int((d < 0).sum())}


def reward_config_comparison(agents: Mapping[str, object], env_config, seeds: Iterable[int],
                             eval_reward: str = "default", layout_seeds: Optional[Sequence[int]] = None):
    """Experiment 2 helper. ``agents``: ``{reward_preset_trained_with: agent}``.

    Agents trained under different reward presets earn rewards on different scales, so all of them are
    evaluated under the *same* ``eval_reward`` preset (default: ``default``) to make reward comparable.
    Returns ``(long_df, summary_df)``.
    """
    cfg = dataclasses.replace(_as_env_config(env_config), reward=eval_reward)
    return compare_agents(agents, cfg, seeds, layout_seeds)


def battery_comparison(cases: Mapping[str, tuple], seeds: Iterable[int],
                       layout_seeds: Optional[Sequence[int]] = None):
    """Experiment 4 helper. ``cases``: ``{case_label: (agent, env_config)}`` (e.g. battery on / off).

    Each case is evaluated in its own env config on the same seeds. Energy per delivery is in
    ``summary_df['energy_per_delivery']``. Returns ``(long_df, summary_df)``.
    """
    seeds = list(seeds)
    frames, rows = [], {}
    for name, (agent, cfg) in cases.items():
        d = evaluate_agent(agent, cfg, seeds, layout_seeds, label=name)
        frames.append(d)
        rows[name] = summarize(d)
    long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return long_df, _summary_frame(rows)


def generalization_table(agents: Mapping[str, object], env_config, train_layout_seeds: Sequence[int],
                         test_layout_seeds: Sequence[int], seeds: Iterable[int]):
    """Per agent: metrics on training layouts vs unseen layouts and the gap (test - train).

    Returns ``(long_df, table)``; ``long_df`` has a ``split`` column ("train"/"test"); ``table`` has one row
    per ``(agent, split)`` using ``summarize`` columns, plus gap rows (``split == 'gap'``).
    """
    seeds = list(seeds)
    frames, rows = [], []
    for name, agent in agents.items():
        sums = {}
        for split, ls in (("train", train_layout_seeds), ("test", test_layout_seeds)):
            d = evaluate_agent(agent, env_config, seeds, ls, label=name)
            d["split"] = split
            frames.append(d)
            sums[split] = summarize(d)
            rows.append({"agent": name, "split": split, **sums[split]})
        rows.append({"agent": name, "split": "gap",
                     **{k: sums["test"][k] - sums["train"][k] for k in RUN_METRIC_KEYS}})
    long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return long_df, pd.DataFrame(rows)
