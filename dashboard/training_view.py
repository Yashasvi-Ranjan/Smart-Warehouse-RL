"""Training tab: configure and run the real trainers in a background thread with live curves."""
from __future__ import annotations

import time

import numpy as np
import pygame

from dashboard import theme as T
from dashboard.base import BaseView, label_value, wrap_text
from dashboard.controller import (ALGO_LABELS, MODELS_DIR, RESULTS_DIR, TRAINER_MODULES, TrainingJob,
                                  build_env_config, load_app_config, module_available, reward_presets)
from dashboard.widgets import Button, Checkbox, Dropdown, ProgressBar, TextField

ALGOS = ("qlearning", "dqn", "ppo")
LEFT_W = 450
BUDGET_KEY = {"qlearning": "episodes", "dqn": "episodes", "ppo": "total_timesteps"}
ROW = 28


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def parse_list(text: str):
    t = text.strip().strip("[]")
    if not t:
        return None
    return [int(x) for x in t.replace(" ", "").split(",") if x]


class Field:
    """One hyperparameter: label + widget built from the config.yaml default."""

    def __init__(self, key, default):
        self.key, self.default = key, default
        if isinstance(default, bool):
            self.kind, self.widget = "bool", Checkbox("", default)
        elif isinstance(default, int):
            self.kind, self.widget = "int", TextField(str(default), "int")
        elif isinstance(default, float):
            self.kind, self.widget = "float", TextField(repr(default), "float")
        elif isinstance(default, (list, tuple)):
            self.kind, self.widget = "list", TextField(",".join(str(x) for x in default))
        else:  # None default -> optional float
            self.kind, self.widget = "opt", TextField("", "float", allow_empty=True)

    def value(self):
        """(ok, value-or-None-to-omit)."""
        w = self.widget
        if self.kind == "bool":
            return True, w.value
        if self.kind == "list":
            try:
                v = parse_list(w.text)
            except ValueError:
                return False, None
            return v is not None, v
        ok, v = w.parse()
        return ok, v


class TrainingView(BaseView):
    name = "Training"

    def __init__(self, app):
        super().__init__(app)
        cfg = load_app_config()
        self.cfg = cfg
        self.job: TrainingJob | None = None
        self.algo = "qlearning"
        self.msg = ""
        self.budget = None
        g = self.group
        self.dd_algo = g.add(Dropdown([(a, ALGO_LABELS[a]) for a in ALGOS], self.algo, self._algo_changed))
        self.tf_name = g.add(TextField(self._default_name(), None))
        self.tf_seed = g.add(TextField("0", "int", minv=0))
        self.dd_layout = g.add(Dropdown([("fixed", "Fixed layout seed"), ("train", "Train set (config splits)"),
                                         ("random", "Random each episode")], "fixed"))
        self.tf_layout = g.add(TextField("0", "int", minv=0))
        self.dd_reward = g.add(Dropdown([(p, p) for p in reward_presets()],
                                        (cfg.get("reward") or {}).get("preset", "default")))
        self.chk_batt = g.add(Checkbox("Battery enabled", (cfg.get("env") or {}).get("battery_enabled", True)))
        self.fields: dict = {}
        for a in ALGOS:
            sec = dict(cfg.get(a) or {})
            fl = [Field(k, v) for k, v in sec.items()]
            if a == "qlearning":
                fl.append(Field("use_action_mask", False))
            self.fields[a] = fl
            for f in fl:
                g.add(f.widget)
        self.btn_start = g.add(Button("Start training", self.start, primary=True))
        self.btn_stop = g.add(Button("Stop", self.stop))
        self.btn_use = g.add(Button("Use in Simulation", self.use_in_sim, primary=True))
        self.bar = ProgressBar((0, 0, 100, 14))
        self._apply_algo_visibility()

    def _default_name(self):
        return f"{self.algo}_ui_{time.strftime('%m%d_%H%M%S')}"

    def _algo_changed(self, a):
        self.algo = a
        self.tf_name.set_text(self._default_name())
        self._apply_algo_visibility()

    def _apply_algo_visibility(self):
        for a, fl in self.fields.items():
            for f in fl:
                f.widget.visible = a == self.algo

    def on_show(self):
        pass

    @property
    def running(self):
        return self.job is not None and self.job.running

    # ------------------------------------------------------------ actions
    def collect(self):
        """Return (env_config, hparams, seed, name) or raise ValueError with a readable message."""
        for w, n in ((self.tf_seed, "Seed"), (self.tf_layout, "Layout seed")):
            if not w.valid:
                raise ValueError(f"{n} is not a valid integer")
        name = self.tf_name.text.strip()
        if not name or any(ch in name for ch in "/\\ "):
            raise ValueError("Run name must be non-empty and contain no spaces or slashes")
        hp = {}
        for f in self.fields[self.algo]:
            ok, v = f.value()
            if not ok:
                if f.kind == "list" and not f.widget.text.strip():
                    continue
                raise ValueError(f"Invalid value for {f.key}")
            if v is not None:
                hp[f.key] = v
        mode = self.dd_layout.value
        splits = self.cfg.get("splits") or {}
        seeds = {"fixed": [self.tf_layout.value], "train": list(splits.get("train_layout_seeds", [0])),
                 "random": None}[mode]
        env_cfg = build_env_config(battery=self.chk_batt.value, reward=self.dd_reward.value, layout_seeds=seeds)
        return env_cfg, hp, int(self.tf_seed.value), name

    def start(self):
        if self.running:
            return
        self.msg = ""
        try:
            env_cfg, hp, seed, name = self.collect()
        except Exception as ex:
            self.msg = str(ex)
            self.job = None
            return
        self.job = TrainingJob(self.algo, env_cfg, hp, seed, name, MODELS_DIR, RESULTS_DIR)
        self.budget = hp.get(BUDGET_KEY[self.algo])
        self.job.start()

    def stop(self):
        if self.job:
            self.job.stop()

    def use_in_sim(self):
        j = self.job
        if j and j.result and j.result.get("model_path"):
            self.app.use_model_in_sim(j.algo, j.result["model_path"])

    # ------------------------------------------------------------ layout
    def layout(self, rect):
        super().layout(rect)
        x0 = rect.x + 8
        y = rect.y + 8
        self._hdr1 = y
        y += T.line_h("head") + 10
        lw, fw = 110, LEFT_W - 110
        self._lbl = []

        def row(label, w, ww=None, dx=0):
            nonlocal y
            self._lbl.append((label, x0, y + 5))
            w.rect = pygame.Rect(x0 + lw + dx, y, ww or fw, 24)
            y += ROW

        row("Algorithm", self.dd_algo)
        row("Run name", self.tf_name)
        row("Training seed", self.tf_seed, 80)
        row("Layouts", self.dd_layout)
        row("Layout seed", self.tf_layout, 80)
        row("Reward preset", self.dd_reward)
        self.chk_batt.rect = pygame.Rect(x0 + lw, y, 160, 24)
        y += ROW + 8
        self._hdr2 = y
        y += T.line_h("head") + 10
        colw = LEFT_W // 2
        self._field_pos = {}
        for a, fl in self.fields.items():
            for i, f in enumerate(fl):
                col, r = divmod(i, 8)
                fx = x0 + col * colw
                fy = y + r * ROW
                lwid = 134
                f.widget.rect = pygame.Rect(fx + lwid, fy, colw - lwid - 8, 24)
                self._field_pos[(a, f.key)] = (fx, fy + 5)
        y += 8 * ROW + 6
        self.btn_start.rect = pygame.Rect(x0, y, 130, 28)
        self.btn_stop.rect = pygame.Rect(x0 + 138, y, 70, 28)
        self._status_y = y + 36
        # right area
        self.rx = x0 + LEFT_W + 24
        self.rw = rect.right - 8 - self.rx

    # ------------------------------------------------------------ draw
    def update(self, dt):
        j = self.job
        self.btn_start.enabled = not self.running
        self.btn_stop.enabled = self.running
        for w in (self.dd_algo, self.tf_name, self.tf_seed, self.dd_layout, self.tf_layout, self.dd_reward,
                  self.chk_batt):
            w.enabled = not self.running
        for f in self.fields[self.algo]:
            f.widget.enabled = not self.running
        self.tf_layout.enabled = (not self.running) and self.dd_layout.value == "fixed"
        self.btn_use.visible = bool(j and j.status == "done" and j.result and j.result.get("model_path"))
        if j and j.status in ("done",) and not getattr(j, "_refreshed", False):
            j._refreshed = True
            self.app.models_changed()

    def draw(self, surf):
        x0 = self.rect.x + 8
        T.section_header(surf, "Experiment", x0, self._hdr1, LEFT_W)
        for text, x, y in self._lbl:
            T.draw_text(surf, text, (x, y), "body", T.MUTED)
        T.section_header(surf, f"Hyperparameters ({ALGO_LABELS[self.algo]})", x0, self._hdr2, LEFT_W)
        for f in self.fields[self.algo]:
            fx, fy = self._field_pos[(self.algo, f.key)]
            T.draw_text(surf, f.key, (fx, fy), "body", T.MUTED, max_w=130)
        ok, why = module_available(TRAINER_MODULES[self.algo])
        y = self._status_y
        if not ok:
            T.draw_text(surf, f"Trainer unavailable: {why}", (x0, y), "body", T.ERR, max_w=LEFT_W)
        elif self.msg:
            for i, ln in enumerate(wrap_text(self.msg, LEFT_W, "body")[:3]):
                T.draw_text(surf, ln, (x0, y + i * 16), "body", T.ERR)
        self.group.draw(surf)
        self._draw_right(surf)

    def _draw_right(self, surf):
        x, w = self.rx, self.rw
        j = self.job
        T.section_header(surf, "Progress", x, self.rect.y + 8, w)
        if j:
            T.draw_text(surf, f"{ALGO_LABELS[j.algo]}  /  {j.run_name}", (x + w, self.rect.y + 11), "label", T.MUTED,
                        "topright", w - 100)
        y = self.rect.y + 8 + T.line_h("head") + 10
        hist, latest = ([], {}) if not j else j.snapshot()
        status = "idle" if not j else j.status
        cw = w // 6
        for k in ("epsilon", "loss"):  # last reported value (final callback may omit them)
            if latest.get(k) is None:
                prev = [h[k] for h in hist if h.get(k) is not None]
                latest[k] = prev[-1] if prev else None
        cells = [("Status", status), ("Episode", _fmt(latest.get("episode"))), ("Timestep", _fmt(latest.get("timestep"))),
                 ("Epsilon", _fmt(latest.get("epsilon"), 3)), ("Loss", _fmt(latest.get("loss"), 4)),
                 ("Elapsed", f"{j.elapsed:.1f} s" if j else "-")]
        for i, (a, b) in enumerate(cells):
            label_value(surf, x + i * cw, y, cw - 8, a, b,
                        T.ERR if (a == "Status" and status == "error") else T.TEXT)
        y += 38
        # progress bar
        frac = 0.0
        if j and self.budget and latest:
            cur = latest.get("timestep") if j.algo == "ppo" else latest.get("episode")
            frac = (cur or 0) / float(self.budget)
        if j and j.status == "done":
            frac = 1.0
        self.bar.rect = pygame.Rect(x, y, w, 14)
        self.bar.value = frac
        self.bar.text = f"{min(1.0, frac) * 100:.0f}%" if j else ""
        self.bar.draw(surf)
        y += 26
        # plots
        use_ts = bool(j and j.algo == "ppo")
        xs = [h.get("timestep" if use_ts else "episode", 0) for h in hist]
        ph = max(90, (self.rect.bottom - y - 8 - 110) // 2)
        self._plot(surf, pygame.Rect(x, y, w, ph), xs, [h.get("reward_ma") for h in hist],
                   "Reward, moving average", "timestep" if use_ts else "episode")
        y += ph + 8
        self._plot(surf, pygame.Rect(x, y, w, ph), xs, [h.get("success_ma") for h in hist],
                   "Success rate, moving average", "timestep" if use_ts else "episode", ylim=(0, 1))
        y += ph + 10
        self._draw_result(surf, x, y, w, j)

    def _draw_result(self, surf, x, y, w, j):
        if not j:
            T.draw_text(surf, "Configure an experiment and press Start. Curves show real moving averages "
                              "reported by the trainer.", (x, y), "label", T.MUTED, max_w=w)
            return
        if j.status == "error":
            for i, ln in enumerate(wrap_text("Training failed: " + j.error, w, "body")[:3]):
                T.draw_text(surf, ln, (x, y + i * 16), "body", T.ERR)
        elif j.status in ("running", "stopping"):
            T.draw_text(surf, "Stopping; the trainer saves the model and log before it exits..."
                        if j.status == "stopping" else f"Training {j.run_name} ...", (x, y), "body", T.MUTED)
        elif j.status == "done" and j.result:
            T.draw_text(surf, "Saved", (x, y), "label", T.MUTED)
            T.draw_text(surf, j.result.get("model_path", ""), (x, y + 13), "body", T.TEXT, max_w=w - 150)
            T.draw_text(surf, j.result.get("log_path", ""), (x, y + 31), "label", T.MUTED, max_w=w - 150)
            self.btn_use.rect = pygame.Rect(x + w - 140, y + 4, 140, 28)

    def _plot(self, surf, rect, xs, ys, title, xlabel, ylim=None):
        T.panel(surf, rect, T.SURFACE, T.BORDER)
        T.draw_text(surf, title, (rect.x + 8, rect.y + 6), "label", T.MUTED)
        pr = pygame.Rect(rect.x + 52, rect.y + 24, rect.w - 52 - 14, rect.h - 24 - 24)
        pts = [(a, b) for a, b in zip(xs, ys) if b is not None]
        if len(pts) < 1:
            T.draw_text(surf, "Waiting for the first progress callback", pr.center, "label", T.MUTED, "center")
            return
        xa = np.array([p[0] for p in pts], float)
        ya = np.array([p[1] for p in pts], float)
        x0, x1 = xa.min(), max(xa.max(), xa.min() + 1)
        if ylim:
            y0, y1 = ylim
        else:
            y0, y1 = ya.min(), ya.max()
            pad = max(1e-6, (y1 - y0) * 0.1)
            y0, y1 = y0 - pad, y1 + pad
        for k in range(5):
            v = y0 + (y1 - y0) * k / 4
            yy = pr.bottom - int(pr.h * k / 4)
            pygame.draw.line(surf, T.BORDER, (pr.x, yy), (pr.right, yy))
            T.draw_text(surf, f"{v:.2f}" if abs(y1 - y0) < 20 else f"{v:.0f}", (pr.x - 6, yy), "label", T.MUTED, "midright")
        for k in range(5):
            v = x0 + (x1 - x0) * k / 4
            xx = pr.x + int(pr.w * k / 4)
            T.draw_text(surf, f"{v:.0f}", (xx, pr.bottom + 4), "label", T.MUTED, "midtop")
        T.draw_text(surf, xlabel, (pr.right, rect.y + 6), "label", T.MUTED, "topright")
        P = [(pr.x + (a - x0) / (x1 - x0) * pr.w, pr.bottom - (b - y0) / (y1 - y0) * pr.h) for a, b in zip(xa, ya)]
        if len(P) > 1:
            pygame.draw.lines(surf, T.ACCENT, False, P, 2)
        else:
            pygame.draw.circle(surf, T.ACCENT, P[0], 3)
        pygame.draw.rect(surf, T.BORDER_STRONG, pr, 1)

    def hint(self):
        return "Tab next tab   Enter commits a field   Esc reverts a field"
