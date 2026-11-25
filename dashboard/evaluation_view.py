"""Evaluation tab: run several agents on identical seeds and compare them."""
from __future__ import annotations

import os
import time

import numpy as np
import pygame

from dashboard import theme as T
from dashboard.base import BaseView, fig_to_surface, wrap_text
from dashboard.controller import (ALGO_LABELS, RESULTS_DIR, EvaluationJob, build_env_config, list_models,
                                  load_agent_from, load_app_config, model_summary, reward_presets)
from dashboard.widgets import Button, Checkbox, Dropdown, FileList, ProgressBar, Table, TextField

LEFT_W = 330
ROW = 28
EXPORT_DIR = os.path.join(RESULTS_DIR, "ui_exports")
LAYOUT_SETS = [("train", "Train layouts (config splits)"), ("test", "Unseen layouts (config splits)"),
               ("custom", "Specific layout seeds"), ("random", "Random layouts")]


def parse_seed_list(text: str):
    out = []
    for tok in text.replace(";", ",").replace(" ", ",").split(","):
        if not tok:
            continue
        if "-" in tok[1:]:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    if not out:
        raise ValueError("enter at least one layout seed (e.g. 0,1,2 or 1000-1009)")
    if any(s < 0 for s in out):
        raise ValueError("layout seeds must be >= 0")
    return out


class EvaluationView(BaseView):
    name = "Evaluation"

    def __init__(self, app):
        super().__init__(app)
        cfg = load_app_config()
        self.cfg = cfg
        g = self.group
        self.agents = g.add(FileList(checks=True, row_h=32))
        self.dd_layouts = g.add(Dropdown(LAYOUT_SETS, "train", lambda v: None))
        self.tf_layouts = g.add(TextField("0,1,2", None))
        self.tf_eps = g.add(TextField("30", "int", minv=1, maxv=10000))
        self.dd_reward = g.add(Dropdown([(p, p) for p in reward_presets()],
                                        (cfg.get("reward") or {}).get("preset", "default")))
        self.chk_batt = g.add(Checkbox("Battery enabled", (cfg.get("env") or {}).get("battery_enabled", True)))
        self.btn_run = g.add(Button("Run evaluation", self.run, primary=True))
        self.btn_stop = g.add(Button("Stop", self.stop))
        self.bar = ProgressBar((0, 0, 100, 14))
        self.table = g.add(Table([("agent", "Agent", 3, "left"), ("reward", "Reward mean +/- std", 3, "right"),
                                  ("success", "Success", 2, "right"), ("steps", "Steps / delivery", 2, "right"),
                                  ("coll", "Collisions", 2, "right"), ("energy", "Energy / delivery", 2, "right"),
                                  ("n", "Episodes", 1, "right")], row_h=22, selectable=True,
                                 on_select=lambda i: self._select_row()))
        self.btn_export = g.add(Button("Export CSV", self.export))
        self.tf_seed = g.add(TextField("", "int", minv=0, allow_empty=True))
        self.btn_replay = g.add(Button("Replay in simulation", self.replay, tooltip="Run the selected agent on this "
                                                                                      "evaluation seed in the Simulation tab"))
        self.job = None
        self.msg = ""
        self.sources: dict = {}  # label -> (algo, path)
        self.chart = None
        self.chart_size = None
        self._chart_dirty = False
        self._last_cfg = None
        self.fig_rect = pygame.Rect(0, 0, 10, 10)
        self.refresh_agents()

    def refresh_agents(self):
        items = [("random", "Random", "uniform random policy"),
                 ("astar", "A* (privileged)", "classical upper bound; reads the map")]
        self.sources = {"random": ("random", None), "astar": ("astar", None)}
        for m in list_models():
            lab = m["run_name"]
            items.append((lab, lab, f"{ALGO_LABELS.get(m['algo'], m['algo'])}   {model_summary(m)}"))
            self.sources[lab] = (m["algo"], m["path"])
        keep = set(self.agents.checked) if self.agents.items else {"random", "astar"}
        self.agents.set_items(items)
        self.agents.checked = keep & {i[0] for i in items}

    def on_show(self):
        self.refresh_agents()

    # ------------------------------------------------------------ run
    def _settings(self):
        splits = self.cfg.get("splits") or {}
        mode = self.dd_layouts.value
        if mode == "train":
            ls = list(splits.get("train_layout_seeds", [0]))
        elif mode == "test":
            ls = list(splits.get("test_layout_seeds", [1000]))
        elif mode == "custom":
            ls = parse_seed_list(self.tf_layouts.text)
        else:
            ls = None
        if not self.tf_eps.valid:
            raise ValueError("Episodes must be a positive integer")
        n = int(self.tf_eps.value)
        base = int(splits.get("eval_seed_base", 100000))
        env_cfg = build_env_config(battery=self.chk_batt.value, reward=self.dd_reward.value, layout_seeds=ls)
        return env_cfg, list(range(base, base + n)), ls

    @property
    def running(self):
        return self.job is not None and self.job.running

    def run(self):
        if self.running:
            return
        self.msg = ""
        try:
            env_cfg, seeds, ls = self._settings()
            chosen = [i[0] for i in self.agents.items if i[0] in self.agents.checked]
            if not chosen:
                raise ValueError("select at least one agent")
            agents, errs = {}, []
            for lab in chosen:
                algo, path = self.sources[lab]
                try:
                    agents[lab] = load_agent_from(algo, path)
                except Exception as ex:
                    errs.append(f"{lab}: {ex}")
            if not agents:
                raise ValueError("no agent could be loaded: " + "; ".join(errs))
            if errs:
                self.msg = "Skipped " + "; ".join(errs)
        except Exception as ex:
            self.msg = str(ex)
            return
        self._last_cfg = (env_cfg, seeds, ls)
        self.job = EvaluationJob(agents, env_cfg, seeds, ls)
        self.job.start()
        self.chart = None
        self.table.rows = []
        self._done_handled = False

    def stop(self):
        if self.job:
            self.job.stop()

    def _fill_results(self):
        s = self.job.summary
        rows = []
        for lab, r in s.iterrows():
            rows.append({"agent": lab, "reward": f"{r['reward_mean']:.1f} +/- {r['reward_std']:.1f}",
                         "success": f"{r['success_rate'] * 100:.0f}%",
                         "steps": "-" if np.isnan(r["steps_per_delivery"]) else f"{r['steps_per_delivery']:.1f}",
                         "coll": f"{r['collisions_mean']:.2f}",
                         "energy": "-" if np.isnan(r["energy_per_delivery"]) else f"{r['energy_per_delivery']:.1f}",
                         "n": int(r["n"])})
        self.table.rows = rows
        self.table.selected = 0 if rows else -1
        self._select_row()
        self._chart_dirty = True

    def _select_row(self):
        if self.job and self._last_cfg and self.tf_seed.text == "":
            self.tf_seed.set_text(str(self._last_cfg[1][0]))

    def _render_chart(self):
        from analytics import plots
        try:
            fig = plots.plot_comparison_bars(self.job.summary)
            self.chart = fig_to_surface(fig, self.fig_rect.size)
        except Exception as ex:
            self.chart = None
            self.msg = f"Chart failed: {ex}"
        self._chart_dirty = False

    def export(self):
        from analytics.export import export_results, export_summary
        if not self.job or self.job.long_df is None:
            self.msg = "Run an evaluation first."
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        try:
            p1 = export_results(self.job.long_df, os.path.join(EXPORT_DIR, f"evaluation_episodes_{stamp}.csv"))
            export_summary(self.job.summary, os.path.join(EXPORT_DIR, f"evaluation_summary_{stamp}.csv"))
            self.msg = "Exported episodes + summary to experiments/results/ui_exports/"
        except Exception as ex:
            self.msg = f"Export failed: {ex}"

    def replay(self):
        j = self.job
        i = self.table.selected
        if not (j and j.summary is not None and 0 <= i < len(self.table.rows)) or self._last_cfg is None:
            self.msg = "Select a result row first."
            return
        lab = self.table.rows[i]["agent"]
        if not self.tf_seed.valid or self.tf_seed.value is None:
            self.msg = "Enter an evaluation seed."
            return
        env_cfg, _, ls = self._last_cfg
        algo, path = self.sources[lab]
        self.app.replay(algo, path, env_cfg.battery_enabled, self.dd_reward.value, ls, int(self.tf_seed.value))

    # ------------------------------------------------------------ layout / draw
    def layout(self, rect):
        super().layout(rect)
        x0, y = rect.x + 8, rect.y + 8
        self._h1 = y
        y += T.line_h("head") + 10
        self.agents.rect = pygame.Rect(x0, y, LEFT_W, max(100, min(260, rect.h - 340)))
        y = self.agents.rect.bottom + 12
        self._h2 = y
        y += T.line_h("head") + 10
        self._lbl = []
        lw = 100

        def row(label, w, ww=None):
            nonlocal y
            self._lbl.append((label, x0, y + 5))
            w.rect = pygame.Rect(x0 + lw, y, ww or LEFT_W - lw, 24)
            y += ROW

        row("Layout set", self.dd_layouts)
        self._seeds_row_y = y
        row("Episodes", self.tf_eps, 80)
        row("Reward preset", self.dd_reward)
        self.chk_batt.rect = pygame.Rect(x0 + lw, y, 160, 24)
        y += ROW
        row("Layout seeds", self.tf_layouts)
        y += 4
        self.btn_run.rect = pygame.Rect(x0, y, 140, 28)
        self.btn_stop.rect = pygame.Rect(x0 + 148, y, 70, 28)
        y += 36
        self.bar.rect = pygame.Rect(x0, y, LEFT_W, 14)
        self._bar_y = y
        self._msg_y = y + 22
        rx = x0 + LEFT_W + 24
        rw = rect.right - 8 - rx
        self.rx, self.rw = rx, rw
        self._hr = rect.y + 8
        ty = self._hr + T.line_h("head") + 10
        self.table.rect = pygame.Rect(rx, ty, rw, 22 * 7 + 2)
        by = self.table.rect.bottom + 8
        self.btn_export.rect = pygame.Rect(rx, by, 100, 28)
        self._rep_lbl = (rx + 120, by + 8)
        self.tf_seed.rect = pygame.Rect(rx + 220, by + 2, 90, 24)
        self.btn_replay.rect = pygame.Rect(rx + 318, by, 160, 28)
        cy = by + 36
        self.fig_rect = pygame.Rect(rx, cy, rw, max(100, rect.bottom - 8 - cy))
        self._chart_dirty = self.job is not None and self.job.summary is not None

    def update(self, dt):
        j = self.job
        self.tf_layouts.visible = self.dd_layouts.value == "custom"
        self.btn_run.enabled = not self.running
        self.btn_stop.enabled = self.running
        have = bool(j and j.summary is not None)
        self.btn_export.enabled = self.btn_replay.enabled = have
        if j:
            self.bar.value = j.progress
            self.bar.text = f"{j.progress * 100:.0f}%  {j.current}" if j.running else ""
            if j.status == "error":
                self.msg = f"Evaluation failed: {j.error}"
            if j.status == "done" and not getattr(self, "_done_handled", True):
                self._done_handled = True
                if j.summary is not None:
                    self._fill_results()
        if self._chart_dirty and have and self.fig_rect.h > 80:
            self._render_chart()

    def draw(self, surf):
        x0 = self.rect.x + 8
        T.section_header(surf, "Agents (same seeds for all)", x0, self._h1, LEFT_W)
        T.section_header(surf, "Settings", x0, self._h2, LEFT_W)
        for t, x, y in self._lbl:
            if t == "Layout seeds" and not self.tf_layouts.visible:
                continue
            T.draw_text(surf, t, (x, y), "body", T.MUTED)
        self.bar.draw(surf)
        if self.msg:
            for i, ln in enumerate(wrap_text(self.msg, LEFT_W, "label")[:4]):
                T.draw_text(surf, ln, (x0, self._msg_y + i * 12), "label", T.ERR if "fail" in self.msg.lower() else T.MUTED)
        T.section_header(surf, "Results", self.rx, self._hr, self.rw)
        self.group.draw(surf)
        if not self.table.rows:
            T.draw_text(surf, "Run an evaluation to fill this table.", (self.rx + 8, self.table.rect.y + 30),
                        "label", T.MUTED)
        T.draw_text(surf, "Seed", (self._rep_lbl[0] + 70, self._rep_lbl[1]), "body", T.MUTED)
        T.panel(surf, self.fig_rect, T.SURFACE, T.BORDER)
        if self.chart is not None:
            surf.blit(self.chart, self.fig_rect.topleft, area=pygame.Rect(0, 0, *self.fig_rect.size))
        else:
            T.draw_text(surf, "Bar charts appear after a run.", (self.fig_rect.x + 16, self.fig_rect.y + 16),
                        "label", T.MUTED)

    def hint(self):
        return "Tab next tab   Click an agent to include it   Evaluation uses deterministic policies and identical seeds"
