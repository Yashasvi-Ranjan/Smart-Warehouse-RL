"""Matplotlib figures (Agg-safe, no pyplot global state): every function returns a ``Figure``.

Figures are built with ``matplotlib.figure.Figure`` directly (no ``plt.figure``), so they can be embedded
in a Qt/pygame/tk canvas or saved with ``save_figure`` without leaking pyplot figures.
Style: one restrained palette (muted blues/greys + orange accent), light grid, no chart junk.
"""
from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from analytics.metrics import moving_average

PALETTE = ["#3B6EA5", "#9AAFC7", "#6B7280", "#D9822B", "#1F3F66", "#B8BEC7", "#2A9D8F"]
ALGO_COLORS = {"random": "#9CA3AF", "astar": "#6B7280", "qlearning": "#8FB0D6",
               "dqn": "#3B6EA5", "ppo": "#D9822B"}
TEXT = "#374151"
GRID = "#E5E7EB"

METRICS = {  # key -> (mean column, label, error kind)
    "reward": ("reward_mean", "Mean episode reward", "reward_std"),
    "success": ("success_rate", "Success rate", "ci"),
    "steps": ("steps_per_delivery", "Steps per delivery", "steps_per_delivery_std"),
    "collisions": ("collisions_mean", "Collisions per episode", "collisions_std"),
    "energy": ("energy_per_delivery", "Energy per delivery (moves)", "energy_per_delivery_std"),
    "recharges": ("recharges_mean", "Recharges per episode", "none"),
    "depleted": ("battery_depleted_rate", "Battery-depleted rate", "none"),
}


def color_for(name: str, i: int = 0) -> str:
    key = str(name).lower().replace("-", "").replace("_", "").replace(" ", "")
    for k, c in ALGO_COLORS.items():
        if key.startswith(k) or k in key:
            return c
    return PALETTE[i % len(PALETTE)]


def _style(ax, xlabel=None, ylabel=None, title=None):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9CA3AF")
    ax.tick_params(colors=TEXT, labelsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
    if title:
        ax.set_title(title, color=TEXT, fontsize=10, loc="left")


def _fig(w, h) -> Figure:
    return Figure(figsize=(w, h), facecolor="white", layout="constrained")


def save_figure(fig: Figure, path: str, dpi: int = 150) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    return path


# ----------------------------------------------------------------------------- training curves
LogsArg = Mapping[str, Union[pd.DataFrame, Sequence[pd.DataFrame]]]


def _curve_stack(logs: Sequence[pd.DataFrame], column: str, window: int):
    """Align several runs on episode index (truncate to shortest). Returns x, mean, std (std None for 1 run)."""
    curves = [moving_average(l[column].astype(float).to_numpy(), window) for l in logs if len(l)]
    if not curves:
        return None, None, None
    n = min(len(c) for c in curves)
    arr = np.vstack([c[:n] for c in curves])
    x = logs[0]["episode"].to_numpy()[:n]
    return x, arr.mean(0), (arr.std(0, ddof=1) if len(curves) > 1 else None)


def plot_training_curves(logs: LogsArg, window: int = 100, metrics: Sequence[str] = ("reward", "success")) -> Figure:
    """Moving-average training curves. ``logs``: ``{name: train_log_df | [train_log_df, ...]}``.

    With several logs per name the line is the mean across runs and the band is +-1 std across runs.
    Raises ValueError if there is no data.
    """
    norm = {k: ([v] if isinstance(v, pd.DataFrame) else list(v)) for k, v in logs.items()}
    norm = {k: [d for d in v if len(d)] for k, v in norm.items()}
    norm = {k: v for k, v in norm.items() if v}
    if not norm:
        raise ValueError("plot_training_curves: no training data")
    labels = {"reward": "Episode reward", "success": "Success rate", "steps": "Steps per episode",
              "collisions": "Collisions per episode"}
    cols = {"reward": "reward", "success": "success", "steps": "steps", "collisions": "collisions"}
    fig = _fig(5.2 * len(metrics), 3.6)
    axes = fig.subplots(1, len(metrics), squeeze=False)[0]
    for ax, m in zip(axes, metrics):
        for i, (name, runs) in enumerate(norm.items()):
            x, mu, sd = _curve_stack(runs, cols[m], window)
            if x is None:
                continue
            c = color_for(name, i)
            ax.plot(x, mu, color=c, lw=1.6, label=name)
            if sd is not None:
                ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.18, lw=0)
        _style(ax, "Training episode", f"{labels[m]} ({window}-episode moving avg)")
        if m == "success":
            ax.set_ylim(-0.02, 1.02)
    if len(norm) > 1:
        axes[0].legend(frameon=False, fontsize=8)
    return fig


# ----------------------------------------------------------------------------- bar charts
def _errors(summary: pd.DataFrame, col: str, kind: str):
    runs_col = col + "_std_runs"
    if runs_col in summary.columns:
        e = np.nan_to_num(summary[runs_col].to_numpy(dtype=float))
        return np.vstack([e, e]), "across training runs"
    if kind == "ci" and {"success_ci_low", "success_ci_high"} <= set(summary.columns):
        v = summary[col].to_numpy(dtype=float)
        # clip: float error can make the Wilson bound a hair inside the point estimate at 100% success
        return np.maximum(np.vstack([v - summary["success_ci_low"].to_numpy(float),
                                     summary["success_ci_high"].to_numpy(float) - v]), 0.0), "95% Wilson CI"
    if kind != "ci" and kind in summary.columns:
        e = summary[kind].to_numpy(dtype=float)
        return np.vstack([e, e]), "std across episodes"
    return None, ""


def plot_comparison_bars(summary: pd.DataFrame, metrics: Sequence[str] = ("reward", "success", "steps", "collisions", "energy")) -> Figure:
    """Bar chart per metric from a summary frame indexed by agent (``compare_agents``/``compare_runs`` output)."""
    if summary is None or len(summary) == 0:
        raise ValueError("plot_comparison_bars: empty summary")
    metrics = [m for m in metrics if METRICS[m][0] in summary.columns]
    ncol = min(3, len(metrics))
    nrow = int(np.ceil(len(metrics) / ncol))
    fig = _fig(3.6 * ncol, 3.0 * nrow)
    axes = np.atleast_2d(fig.subplots(nrow, ncol, squeeze=False)).ravel()
    names = [str(i) for i in summary.index]
    colors = [color_for(n, i) for i, n in enumerate(names)]
    x = np.arange(len(names))
    for ax, m in zip(axes, metrics):
        col, label, kind = METRICS[m]
        vals = summary[col].to_numpy(dtype=float)
        err, note = _errors(summary, col, kind)
        ax.bar(x, np.nan_to_num(vals), color=colors, width=0.65,
               yerr=None if err is None else np.nan_to_num(err), capsize=3,
               error_kw={"ecolor": "#4B5563", "lw": 1})
        ax.set_xticks(x, names, rotation=30, ha="right")
        _style(ax, None, label, None)
        if note:
            ax.set_title(f"error bars: {note}", fontsize=7, color="#6B7280", loc="left")
        if m == "success":
            ax.set_ylim(0, 1.05)
    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    return fig


def plot_episode_length_distribution(long_df: pd.DataFrame, column: str = "steps", group: str = "label") -> Figure:
    """Distribution of episode lengths per agent (box plot; every episode, incl. failed/timeouts)."""
    if long_df is None or len(long_df) == 0:
        raise ValueError("plot_episode_length_distribution: empty data")
    names = list(dict.fromkeys(long_df[group]))
    data = [long_df.loc[long_df[group] == n, column].to_numpy(float) for n in names]
    fig = _fig(max(4.5, 0.9 * len(names) + 2.5), 3.6)
    ax = fig.subplots()
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                    medianprops={"color": "#1F2937", "lw": 1.4}, flierprops={"markersize": 2.5, "alpha": 0.4})
    for patch, n, i in zip(bp["boxes"], names, range(len(names))):
        patch.set(facecolor=color_for(n, i), alpha=0.75, edgecolor="#4B5563")
    ax.set_xticks(range(1, len(names) + 1), names, rotation=30, ha="right")
    _style(ax, None, "Episode length (steps)")
    return fig


def plot_generalization(table: pd.DataFrame, metric: str = "success_rate", ylabel: Optional[str] = None) -> Figure:
    """Grouped bars: training layouts vs unseen layouts per agent. ``table``: ``generalization_table`` output
    (rows with ``split`` in train/test); error bars shown if ``<metric>_std_runs`` exists."""
    if table is None or len(table) == 0:
        raise ValueError("plot_generalization: empty table")
    t = table[table["split"].isin(["train", "test"])]
    agents = list(dict.fromkeys(t["agent"]))
    x = np.arange(len(agents))
    fig = _fig(max(4.5, 1.1 * len(agents) + 2.5), 3.6)
    ax = fig.subplots()
    w = 0.38
    for j, (split, color, lab) in enumerate((("train", "#3B6EA5", "Training layouts"), ("test", "#D9822B", "Unseen layouts"))):
        s = t[t["split"] == split].set_index("agent").reindex(agents)
        err = s[metric + "_std_runs"].to_numpy(float) if metric + "_std_runs" in s.columns else None
        ax.bar(x + (j - 0.5) * w, np.nan_to_num(s[metric].to_numpy(float)), w, color=color, label=lab,
               yerr=None if err is None else np.nan_to_num(err), capsize=3, error_kw={"ecolor": "#4B5563", "lw": 1})
    ax.set_xticks(x, agents, rotation=30, ha="right")
    _style(ax, None, ylabel or metric.replace("_", " "))
    if metric == "success_rate":
        ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    return fig


# ----------------------------------------------------------------------------- interactive (optional)
def plot_interactive_training(logs: LogsArg, window: int = 100, metric: str = "reward", path: Optional[str] = None):
    """Plotly training-curve figure (mean across runs). Writes standalone HTML when ``path`` is given.
    Returns the plotly Figure. Raises ImportError if plotly is missing, ValueError without data."""
    import plotly.graph_objects as go
    norm = {k: ([v] if isinstance(v, pd.DataFrame) else list(v)) for k, v in logs.items()}
    fig = go.Figure()
    n = 0
    for i, (name, runs) in enumerate(norm.items()):
        x, mu, sd = _curve_stack([r for r in runs if len(r)], metric, window)
        if x is None:
            continue
        n += 1
        c = color_for(name, i)
        fig.add_trace(go.Scatter(x=x, y=mu, mode="lines", name=name, line={"color": c, "width": 2}))
        if sd is not None:
            fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[mu + sd, (mu - sd)[::-1]], fill="toself",
                                     fillcolor="rgba(120,140,170,0.18)", line={"width": 0}, hoverinfo="skip",
                                     showlegend=False))
    if n == 0:
        raise ValueError("plot_interactive_training: no training data")
    fig.update_layout(template="simple_white", xaxis_title="Training episode",
                      yaxis_title=f"{metric} ({window}-episode moving avg)", margin={"l": 50, "r": 10, "t": 20, "b": 40})
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.write_html(path, include_plotlyjs="cdn")
    return fig
