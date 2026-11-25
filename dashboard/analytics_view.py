"""Analytics tab: matplotlib figures from analytics.plots rendered into the pygame window."""
from __future__ import annotations

import glob
import os
import time

import pandas as pd
import pygame

from dashboard import theme as T
from dashboard.base import BaseView, fig_to_surface, wrap_text
from dashboard.controller import MODELS_DIR, RESULTS_DIR
from dashboard.widgets import Button, Checkbox, Dropdown, FileList, TextField

LEFT_W = 300
ROW = 28
MODES = [("curves", "Training curves"), ("compare", "Algorithm comparison"), ("general", "Generalisation")]
CURVE_METRICS = [("reward", "Reward"), ("success", "Success"), ("steps", "Steps"), ("collisions", "Collisions")]
EXPORT_DIR = os.path.join(RESULTS_DIR, "ui_exports")


def load_experiments(results_dir=RESULTS_DIR) -> dict:
    """{exp_dir_name: summary DataFrame} for every experiments/results/exp*/summary.csv."""
    out = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "exp*", "summary.csv"))):
        try:
            out[os.path.basename(os.path.dirname(p))] = pd.read_csv(p)
        except Exception:
            continue
    return out


def comparison_frame(df: pd.DataFrame, condition=None):
    """Summary indexed by agent label (one condition) or None."""
    d = df.copy()
    if "condition" in d.columns:
        conds = list(dict.fromkeys(d["condition"]))
        condition = condition if condition in conds else conds[0]
        d = d[d["condition"] == condition]
    key = "label" if "label" in d.columns else "agent" if "agent" in d.columns else None
    if key is None or "reward_mean" not in d.columns:
        return None
    d = d.set_index(key)
    d.index.name = "agent"
    return d.drop(columns=[c for c in ("condition",) if c in d.columns])


def generalisation_table(df: pd.DataFrame):
    if not {"condition", "label"} <= set(df.columns):
        return None
    t = df.rename(columns={"label": "agent", "condition": "split"})
    t["split"] = t["split"].map({"train_layouts": "train", "unseen_layouts": "test"})
    t = t.dropna(subset=["split"])
    return t if len(t) else None


class AnalyticsView(BaseView):
    name = "Analytics"

    def __init__(self, app):
        super().__init__(app)
        g = self.group
        self.mode = "curves"
        self.dd_mode = g.add(Dropdown(MODES, "curves", self._mode_changed))
        self.runs = g.add(FileList(checks=True, on_change=lambda i: self._dirty(), row_h=32))
        self.metric_chk = {k: g.add(Checkbox(lbl, k in ("reward", "success"), lambda v: self._dirty()))
                           for k, lbl in CURVE_METRICS}
        self.tf_window = g.add(TextField("100", "int", minv=1, maxv=100000, on_commit=lambda v: self._dirty()))
        self.dd_exp = g.add(Dropdown([], None, lambda v: self._exp_changed()))
        self.dd_cond = g.add(Dropdown([], None, lambda v: self._dirty()))
        self.dd_gmetric = g.add(Dropdown([("success_rate", "Success rate"), ("reward_mean", "Mean reward")],
                                         "success_rate", lambda v: self._dirty()))
        self.btn_refresh = g.add(Button("Refresh data", self.refresh))
        self.btn_export = g.add(Button("Export CSV", self.export, primary=True))
        self.msg = ""
        self.surface = None
        self.surface_size = None
        self.empty = ""
        self._dirty_t = None
        self._data = None  # DataFrame currently plotted (for export)
        self._runs_df = pd.DataFrame()
        self.exps: dict = {}
        self.fig_rect = pygame.Rect(0, 0, 100, 100)
        self.refresh()

    # ------------------------------------------------------------ data
    def refresh(self):
        from analytics.export import discover_runs
        try:
            self._runs_df = discover_runs(RESULTS_DIR, MODELS_DIR)
        except Exception:
            self._runs_df = pd.DataFrame()
        df = self._runs_df
        items = []
        if len(df):
            for _, r in df[df["log_path"].notna()].iterrows():
                sub = f"{r['algo'] or '?'}  {int(r['episodes']) if pd.notna(r['episodes']) else '?'} ep"
                if pd.notna(r["final_success_ma"]):
                    sub += f"  final success {r['final_success_ma']:.2f}"
                items.append((r["run_name"], r["run_name"], sub))
        old = set(self.runs.checked)
        self.runs.set_items(items)
        self.runs.checked = old & {i[0] for i in items}
        if not self.runs.checked and items and self.mode == "curves":
            self.runs.checked = {items[-1][0]}
        self.exps = load_experiments()
        self.dd_exp.set_options([(k, k) for k in self.exps])
        if self.dd_exp.value not in self.exps:
            self.dd_exp.value = next(iter(self.exps), None)
        self._exp_changed(False)
        self._dirty()

    def on_show(self):
        self.refresh()

    def _exp_changed(self, dirty=True):
        d = self.exps.get(self.dd_exp.value)
        conds = list(dict.fromkeys(d["condition"])) if d is not None and "condition" in d.columns else []
        self.dd_cond.set_options([(c, c) for c in conds])
        if self.dd_cond.value not in conds:
            self.dd_cond.value = conds[0] if conds else None
        if dirty:
            self._dirty()

    def _mode_changed(self, m):
        self.mode = m
        self.msg = ""
        self._dirty()

    def _dirty(self):
        self._dirty_t = time.time()

    # ------------------------------------------------------------ rendering of figure
    def _build_figure(self):
        from analytics import plots
        from analytics.metrics import load_train_log
        self._data = None
        self.empty = ""
        if self.mode == "curves":
            names = [i[0] for i in self.runs.items if i[0] in self.runs.checked]
            if not self.runs.items:
                self.empty = ("No training logs found. Train a model in the Training tab, or run "
                              "`python -m experiments.run_experiments --exp 1 --quick`.")
                return None
            if not names:
                self.empty = "Select one or more runs on the left."
                return None
            metrics = [k for k, _ in CURVE_METRICS if self.metric_chk[k].value] or ["reward"]
            logs = {n: load_train_log(n, RESULTS_DIR) for n in names}
            self._data = pd.concat([d.assign(run=n) for n, d in logs.items()], ignore_index=True)
            win = int(self.tf_window.value or 100)
            return [plots.plot_training_curves(logs, window=win, metrics=[m]) for m in metrics]
        d = self.exps.get(self.dd_exp.value)
        if d is None:
            self.empty = ("No experiment summaries found (experiments/results/exp*/summary.csv). "
                          "Generate them with `python -m experiments.run_experiments --exp all --quick`.")
            return None
        if self.mode == "compare":
            s = comparison_frame(d, self.dd_cond.value)
            if s is None:
                self.empty = "This summary file has no per-agent metrics."
                return None
            self._data = s.reset_index()
            return plots.plot_comparison_bars(s)
        t = generalisation_table(d)
        if t is None:
            self.empty = ("The selected experiment has no train / unseen layout split. "
                          "Choose exp3 (generalisation).")
            return None
        self._data = t
        return plots.plot_generalization(t, self.dd_gmetric.value)

    def _render(self):
        try:
            fig = self._build_figure()
            if fig is None:
                self.surface = None
                return
            figs = fig if isinstance(fig, list) else [fig]
            w, h = self.fig_rect.size
            hh = h // len(figs)
            self.surface = pygame.Surface((w, h))
            self.surface.fill((255, 255, 255))
            for i, f in enumerate(figs):
                self.surface.blit(fig_to_surface(f, (w, hh)), (0, i * hh))
            self.surface_size = self.fig_rect.size
            self.msg = ""
        except Exception as ex:
            self.surface = None
            self.empty = f"Could not draw figure: {type(ex).__name__}: {ex}"

    # ------------------------------------------------------------ export
    def export(self):
        from analytics.export import export_results
        if self._data is None or len(self._data) == 0:
            self.msg = "Nothing to export for the current selection."
            return
        name = {"curves": "training_curves", "compare": "comparison", "general": "generalisation"}[self.mode]
        path = os.path.join(EXPORT_DIR, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        try:
            export_results(self._data, path)
            self.msg = f"Exported {os.path.relpath(path, os.path.dirname(RESULTS_DIR))}"
        except Exception as ex:
            self.msg = f"Export failed: {ex}"

    # ------------------------------------------------------------ layout / draw
    def layout(self, rect):
        super().layout(rect)
        x0, y = rect.x + 8, rect.y + 8
        self._h1 = y
        y += T.line_h("head") + 10
        self._lbl = [("View", x0, y + 5)]
        self.dd_mode.rect = pygame.Rect(x0 + 80, y, LEFT_W - 80, 24)
        y += ROW + 4
        self._ctl_y = y
        # curves controls
        self.runs.rect = pygame.Rect(x0, y, LEFT_W, max(80, rect.bottom - 8 - y - 36 - 28 - 70 - 40))
        yy = self.runs.rect.bottom + 8
        xs = x0
        for (k, _), cw in zip(CURVE_METRICS, (72, 74, 60, 94)):
            self.metric_chk[k].rect = pygame.Rect(xs, yy, cw, 24)
            xs += cw
        yy += ROW
        self._win_y = yy
        self.tf_window.rect = pygame.Rect(x0 + 80, yy, 70, 24)
        # experiment controls (same area as the runs list)
        self.dd_exp.rect = pygame.Rect(x0 + 80, y, LEFT_W - 80, 24)
        self.dd_cond.rect = pygame.Rect(x0 + 80, y + ROW, LEFT_W - 80, 24)
        self.dd_gmetric.rect = pygame.Rect(x0 + 80, y + ROW, LEFT_W - 80, 24)
        by = rect.bottom - 8 - 28
        self.btn_refresh.rect = pygame.Rect(x0, by, 120, 28)
        self.btn_export.rect = pygame.Rect(x0 + 128, by, 110, 28)
        self._msg_y = by - 20
        self.fig_rect = pygame.Rect(x0 + LEFT_W + 16, rect.y + 8, rect.right - 8 - (x0 + LEFT_W + 16), rect.h - 16)
        self._dirty()

    def update(self, dt):
        c = self.mode == "curves"
        for w in (self.runs, self.tf_window, *self.metric_chk.values()):
            w.visible = c
        self.dd_exp.visible = not c
        self.dd_cond.visible = self.mode == "compare" and bool(self.dd_cond.options) and len(self.dd_cond.options) > 1
        self.dd_gmetric.visible = self.mode == "general"
        if self._dirty_t is not None and time.time() - self._dirty_t > 0.15 and self.fig_rect.w > 100:
            self._dirty_t = None
            self._render()

    def draw(self, surf):
        x0 = self.rect.x + 8
        T.section_header(surf, "Data", x0, self._h1, LEFT_W)
        for t, x, y in self._lbl:
            T.draw_text(surf, t, (x, y), "body", T.MUTED)
        if self.mode == "curves":
            T.draw_text(surf, "Smoothing", (x0, self._win_y + 5), "body", T.MUTED)
            T.draw_text(surf, "episodes", (x0 + 156, self._win_y + 5), "body", T.MUTED)
        else:
            T.draw_text(surf, "Experiment", (x0, self._ctl_y + 5), "body", T.MUTED)
            if self.dd_cond.visible:
                T.draw_text(surf, "Condition", (x0, self._ctl_y + ROW + 5), "body", T.MUTED)
            if self.dd_gmetric.visible:
                T.draw_text(surf, "Metric", (x0, self._ctl_y + ROW + 5), "body", T.MUTED)
            if self.exps:
                T.draw_text(surf, f"{len(self.exps)} experiment summaries found", (x0, self._ctl_y + 2 * ROW + 8),
                            "label", T.MUTED)
        if self.msg:
            T.draw_text(surf, self.msg, (x0, self._msg_y), "label", T.MUTED, max_w=LEFT_W)
        self.group.draw(surf)
        fr = self.fig_rect
        T.panel(surf, fr, T.SURFACE, T.BORDER)
        if self.surface is not None:
            surf.blit(self.surface, fr.topleft, area=pygame.Rect(0, 0, fr.w, fr.h))
        else:
            for i, ln in enumerate(wrap_text(self.empty or "Rendering...", fr.w - 40, "body")):
                T.draw_text(surf, ln, (fr.x + 20, fr.y + 20 + i * 16), "body", T.MUTED)

    def hint(self):
        return "Tab next tab   Click runs to toggle them in the plot   Figures are matplotlib (Agg) output"
