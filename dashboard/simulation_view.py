"""Simulation tab: warehouse grid (dominant) + configuration / status / episode history panel."""
from __future__ import annotations

import time

import pygame

from dashboard import theme as T
from dashboard.base import BaseView, label_value, wrap_text
from dashboard.controller import (ACTION_NAMES, ALGO_LABELS, SimulationController, algo_availability,
                                  list_models, model_summary, reward_presets)
from dashboard.widgets import (Button, Checkbox, Dropdown, FileList, Group, Label, ProgressBar, Slider,
                               Table, TextField)
from environment import renderer

PANEL_W = 340
MAX_SPEED = 61  # slider value meaning "max"
ROW = 28


class SimulationView(BaseView):
    name = "Simulation"

    def __init__(self, app):
        super().__init__(app)
        self.ctrl: SimulationController = app.sim
        self.running = False
        self.acc = 0.0
        self.speed = 8
        self.auto_next = False
        self._next_timer = None
        self.cell = 40
        self.grid_rect = pygame.Rect(0, 0, 10, 10)
        self._anim = None  # (from_pos, start_time, dur)
        self._last_pos = tuple(self.ctrl.env.robot_pos)
        self.show_trail = True
        self.modal = None

        g = self.group
        self.btn_run = g.add(Button("Start", self.toggle_run, primary=True, tooltip="Start / pause  [Space]"))
        self.btn_step = g.add(Button("Step", self.step_once, tooltip="Single step  [S]"))
        self.btn_reset = g.add(Button("Reset", self.reset, tooltip="Restart this episode (same task)  [R]"))
        self.btn_new = g.add(Button("New task", self.new_task, tooltip="Next task seed on the same layout  [N]"))
        self.sl_speed = g.add(Slider(1, MAX_SPEED, self.speed, self._speed_changed,
                                     tooltip="Steps per second; right end = maximum  [+/-]"))
        self.chk_auto = g.add(Checkbox("Auto-next", False, lambda v: setattr(self, "auto_next", v),
                                       tooltip="Start the next task automatically"))
        self.chk_trail = g.add(Checkbox("Trail", True, lambda v: setattr(self, "show_trail", v)))
        # configuration
        self.dd_agent = g.add(Dropdown([], "random", self._agent_changed))
        self.btn_load = g.add(Button("Load model...", self.open_browser))
        self.dd_layout = g.add(Dropdown([("fixed", "Fixed seed"), ("random", "Random per task")], "fixed",
                                        self._layout_mode))
        self.btn_prev = g.add(Button("<", lambda: self._bump_layout(-1), tooltip="Previous layout seed"))
        self.tf_seed = g.add(TextField("0", "int", on_commit=self._seed_commit, minv=0, maxv=2 ** 31 - 2))
        self.btn_next = g.add(Button(">", lambda: self._bump_layout(1), tooltip="Next layout seed"))
        self.dd_reward = g.add(Dropdown([(p, p) for p in reward_presets()], self.ctrl.reward, self._reward_changed))
        self.chk_batt = g.add(Checkbox("Battery", self.ctrl.battery, self._battery_changed))
        self.chk_mask = g.add(Checkbox("Action mask", False, self._mask_changed,
                                       tooltip="Restrict the agent to valid actions (if it supports masking)"))
        self.bar = ProgressBar((0, 0, 100, 12))
        self.table = g.add(Table([("n", "#", 1, "right"), ("result", "Result", 3, "left"),
                                  ("reward", "Reward", 2, "right"), ("steps", "Steps", 2, "right")],
                                 row_h=18))
        # model browser (modal)
        self.mg = Group()
        self.fl = self.mg.add(FileList(row_h=34))
        self.btn_m_load = self.mg.add(Button("Load", self._browser_load, primary=True))
        self.btn_m_cancel = self.mg.add(Button("Cancel", self.close_browser))
        self.m_models: list = []
        self.refresh_algos()

    # ------------------------------------------------------------ helpers
    def refresh_algos(self):
        av = algo_availability()
        self.availability = av
        opts = []
        for k in ("random", "astar", "qlearning", "dqn", "ppo"):
            ok, why = av[k]
            opts.append((k, ALGO_LABELS[k] + ("" if ok else "  (unavailable)"), ok))
        self.dd_agent.set_options(opts)
        self.dd_agent.value = self.ctrl.agent_key

    def sync_from_controller(self):
        c = self.ctrl
        self.refresh_algos()
        self.dd_reward.value = c.reward
        self.chk_batt.value = c.battery
        random_mode = c.layout_seed is None and c.layout_seeds_set is None
        self.dd_layout.value = "random" if random_mode else "fixed"
        if c.layout_seed is not None:
            self.tf_seed.set_text(str(c.layout_seed))
        self.chk_mask.value = bool(getattr(c.agent, "use_action_mask", False))
        self.running = False
        self._last_pos = tuple(c.env.robot_pos)

    def on_show(self):
        self.refresh_algos()

    # ------------------------------------------------------------ actions
    def toggle_run(self):
        if self.ctrl.done:
            self.ctrl.reset(new_task=True)
            self._last_pos = tuple(self.ctrl.env.robot_pos)
        self.running = not self.running
        self._next_timer = None

    def step_once(self):
        self.running = False
        if self.ctrl.done:
            self.ctrl.reset(new_task=True)
            self._last_pos = tuple(self.ctrl.env.robot_pos)
            return
        self._do_step(animate=False)

    def reset(self):
        self.running = False
        self.ctrl.reset()
        self._anim = None
        self._last_pos = tuple(self.ctrl.env.robot_pos)

    def new_task(self):
        self.ctrl.reset(new_task=True)
        self._anim = None
        self._last_pos = tuple(self.ctrl.env.robot_pos)

    def _do_step(self, animate=True):
        c = self.ctrl
        if c.done:
            return
        c.step()
        pos = tuple(c.env.robot_pos)
        if animate and pos != self._last_pos and self.speed < MAX_SPEED and self.speed <= 15:
            self._anim = (self._last_pos, time.perf_counter(), min(0.1, 0.7 / self.speed))
        else:
            self._anim = None
        self._last_pos = pos
        if c.done:
            self.running = False
            if self.auto_next:
                self._next_timer = 0.8

    def _speed_changed(self, v):
        self.speed = v

    def _agent_changed(self, key):
        ok, why = self.availability[key]
        if not self.ctrl.set_agent(key):
            self.dd_agent.value = self.ctrl.agent_key
        self.chk_mask.value = bool(getattr(self.ctrl.agent, "use_action_mask", False))
        self.reset()

    def _layout_mode(self, v):
        if v == "random":
            self.ctrl.configure(layout_seed=None)
        else:
            self.ctrl.configure(layout_seed=self.tf_seed.value or 0)
        self.running = False

    def _seed_commit(self, v):
        self.dd_layout.value = "fixed"
        self.ctrl.configure(layout_seed=int(v))
        self.running = False

    def _bump_layout(self, d):
        v = max(0, (self.tf_seed.value or 0) + d)
        self.tf_seed.set_text(v)
        self._seed_commit(v)

    def _reward_changed(self, v):
        self.ctrl.configure(reward=v)
        self.running = False

    def _battery_changed(self, v):
        self.ctrl.configure(battery=v)
        self.running = False

    def _mask_changed(self, v):
        self.ctrl.set_action_mask(v)

    # ------------------------------------------------------------ model browser
    def open_browser(self):
        self.m_models = list_models()
        self.fl.set_items([(m["path"], f"{m['run_name']}", f"{ALGO_LABELS.get(m['algo'], m['algo'])}   {model_summary(m)}")
                           for m in self.m_models])
        cur = self.ctrl.model_path
        for i, m in enumerate(self.m_models):
            if m["path"] == cur:
                self.fl.selected = i
        self.modal = True

    def close_browser(self):
        self.modal = None

    def _browser_load(self):
        i = self.fl.selected
        if 0 <= i < len(self.m_models):
            m = self.m_models[i]
            self.load_model(m["algo"], m["path"])
        self.modal = None

    def load_model(self, algo, path):
        self.ctrl.set_agent(algo, path)
        self.sync_from_controller()
        self.reset()

    # ------------------------------------------------------------ layout
    def layout(self, rect):
        super().layout(rect)
        px = rect.right - PANEL_W - 8
        left = pygame.Rect(rect.x + 8, rect.y + 8, px - rect.x - 16, rect.h - 16)
        # toolbar
        x, y = left.x, left.y
        for b, w in ((self.btn_run, 76), (self.btn_step, 56), (self.btn_reset, 56), (self.btn_new, 76)):
            b.rect = pygame.Rect(x, y, w, ROW)
            x += w + 8
        x += 8
        self._speed_lbl_x = x
        self.sl_speed.rect = pygame.Rect(x + 44, y, 130, ROW)
        x += 44 + 130 + 52
        self.chk_auto.rect = pygame.Rect(x, y, 84, ROW)
        self.chk_trail.rect = pygame.Rect(x + 90, y, 60, ROW)
        # grid
        env = self.ctrl.env
        H, W = env.config.grid_h, env.config.grid_w
        area = pygame.Rect(left.x, y + ROW + 8, left.w, left.h - ROW - 8 - 28)
        self.cell = max(8, min(area.w // W, area.h // H))
        gw, gh = W * self.cell, H * self.cell
        self.grid_rect = pygame.Rect(area.x, area.y, gw, gh)
        self.legend_y = self.grid_rect.bottom + 10
        # right panel
        x0, w = px, PANEL_W
        self.px, self.pw = x0, w
        y = rect.y + 8
        y = y + T.line_h("head") + 10
        lw = 78  # label column
        cx, cw = x0 + lw, w - lw
        self._lbl = []

        def row(label, widgets):
            nonlocal y
            self._lbl.append((label, x0, y + 5))
            for wd, ox, ww in widgets:
                wd.rect = pygame.Rect(cx + ox, y, ww, 24)
            y += ROW

        row("Agent", [(self.dd_agent, 0, cw)])
        self._model_y = y
        self.btn_load.rect = pygame.Rect(cx, y, 110, 24)
        y += ROW
        self._model_text_y = y - 2
        y += 22 if self.ctrl.agent_error else 4
        self._err_y = self._model_text_y
        row("Layout", [(self.dd_layout, 0, cw)])
        row("Layout seed", [(self.btn_prev, 0, 26), (self.tf_seed, 30, 72), (self.btn_next, 106, 26)])
        row("Reward", [(self.dd_reward, 0, cw)])
        self._chk_y = y
        self.chk_batt.rect = pygame.Rect(cx, y, 84, 24)
        self.chk_mask.rect = pygame.Rect(cx + 90, y, 110, 24)
        y += ROW + 6
        self._status_y = y
        # status block height: 5 rows of 32 + battery 28 + last-action 20 + header
        status_h = T.line_h("head") + 10 + 32 * 5 + 30 + 4
        self._status_h = status_h
        ty = y + status_h + 4
        self._hist_y = ty
        hy = ty + T.line_h("head") + 10
        self.table.rect = pygame.Rect(x0, hy, w, max(60, rect.bottom - 8 - hy))

    # ------------------------------------------------------------ update / draw
    def update(self, dt):
        c = self.ctrl
        self.btn_run.label = "Pause" if self.running else "Start"
        ok = self.availability.get(c.agent_key, (True, ""))[0]
        for b in (self.btn_run, self.btn_step):
            b.enabled = ok
        self.chk_mask.visible = hasattr(c.agent, "use_action_mask")
        self.tf_seed.enabled = self.btn_prev.enabled = self.btn_next.enabled = self.dd_layout.value == "fixed"
        if self._next_timer is not None:
            self._next_timer -= dt
            if self._next_timer <= 0:
                self._next_timer = None
                c.reset(new_task=True)
                self._last_pos = tuple(c.env.robot_pos)
                self.running = True
        if self.running and not c.done:
            if self.speed >= MAX_SPEED:
                t0 = time.perf_counter()
                while not c.done and time.perf_counter() - t0 < 0.012:
                    self._do_step(animate=False)
            else:
                self.acc += dt
                n = int(self.acc * self.speed)
                if n:
                    self.acc -= n / self.speed
                    for _ in range(min(n, 30)):
                        self._do_step()
        else:
            self.acc = 0.0
        self._fill_table()

    def _fill_table(self):
        hist = list(self.ctrl.history)[::-1]
        rows, colors = [], {}
        for i, h in enumerate(hist):
            rows.append({"n": h["n"], "result": h["result"], "reward": f"{h['reward']:.1f}", "steps": h["steps"]})
            colors[(i, "result")] = T.OK if h["result"] == "delivered" else T.ERR if "battery" in h["result"] else T.WARN
        self.table.rows, self.table.colors = rows, colors

    def _visual_pos(self):
        if self._anim:
            f, t0, dur = self._anim
            k = (time.perf_counter() - t0) / dur
            if k >= 1:
                self._anim = None
            else:
                to = self._last_pos
                return (f[0] + (to[0] - f[0]) * k, f[1] + (to[1] - f[1]) * k)
        return None

    def draw(self, surf):
        c = self.ctrl
        # grid
        state = c.render_state()
        state["show_trail"] = self.show_trail
        vp = self._visual_pos()
        if vp:
            state["robot_visual_pos"] = vp
        gsurf = pygame.Surface(self.grid_rect.size)
        renderer.draw_warehouse(gsurf, state, self.cell)
        surf.blit(gsurf, self.grid_rect)
        pygame.draw.rect(surf, T.BORDER_STRONG, self.grid_rect.inflate(2, 2), 1)
        if c.done:
            banner = pygame.Rect(self.grid_rect.x, self.grid_rect.y, self.grid_rect.w, 24)
            pygame.draw.rect(surf, (36, 38, 44), banner)
            col = (150, 210, 170) if c.result == "delivered" else (230, 150, 140)
            T.draw_text(surf, f"Episode ended: {c.result}  |  reward {c.ep_reward:.1f}  |  {c.steps} steps",
                        (banner.x + 8, banner.centery), "body", col, "midleft", banner.w - 16)
        self._draw_legend(surf)
        # toolbar labels
        T.draw_text(surf, "Speed", (self._speed_lbl_x, self.sl_speed.rect.centery), "body", T.MUTED, "midleft")
        sp = "max" if self.speed >= MAX_SPEED else f"{self.speed}/s"
        T.draw_text(surf, sp, (self.sl_speed.rect.right + 6, self.sl_speed.rect.centery), "body", T.TEXT, "midleft")
        # right panel
        x0, w = self.px, self.pw
        T.section_header(surf, "Configuration", x0, self.rect.y + 8, w)
        for text, x, y in self._lbl:
            T.draw_text(surf, text, (x, y), "body", T.MUTED)
        mt = c.model_path and f"Model: {c.model_meta.get('run_name', '')}  [{c.model_meta.get('algo', '')}]" or \
            ("Model: not required" if c.agent_key in ("random", "astar") else "Model: none loaded")
        T.draw_text(surf, mt, (x0 + 78 + 118, self._model_y + 12), "label", T.MUTED, "midleft", w - 78 - 118)
        if c.agent_error:
            lines = wrap_text(c.agent_error, w, "label")[:2]
            for i, ln in enumerate(lines):
                T.draw_text(surf, ln, (x0, self._model_text_y + i * 12), "label", T.ERR)
        self.group.draw(surf)
        self._draw_status(surf)
        T.section_header(surf, "Episodes", x0, self._hist_y, w)
        if not self.table.rows:
            T.draw_text(surf, "No finished episodes yet", (x0 + 6, self.table.rect.y + 26), "label", T.MUTED)
        if self.modal:
            self._draw_modal(surf)

    def _draw_legend(self, surf):
        x, y = self.grid_rect.x, self.legend_y
        items = [("Wall", renderer.WALL_C), ("Shelf", renderer.SHELF_C), ("Pickup", renderer.PICKUP_C),
                 ("Delivery", renderer.DELIVERY_C), ("Charger", renderer.CHARGER_C), ("Robot", renderer.ROBOT_C),
                 ("Package", renderer.PACKAGE_C), ("Trail", renderer.TRAIL_C)]
        for name, col in items:
            pygame.draw.rect(surf, col, (x, y + 1, 10, 10))
            pygame.draw.rect(surf, T.BORDER_STRONG, (x, y + 1, 10, 10), 1)
            r = T.draw_text(surf, name, (x + 14, y), "label", T.MUTED)
            x = r.right + 14
        T.draw_text(surf, "Outlined P / D = active task", (x, y), "label", T.MUTED)

    def _draw_status(self, surf):
        c = self.ctrl
        x0, w, y = self.px, self.pw, self._status_y
        y = T.section_header(surf, "Status", x0, y, w)
        cw = (w - 8) // 2
        env = c.env
        cap = env.config.battery_capacity
        res_col = T.TEXT if not c.result else T.OK if c.result == "delivered" else T.ERR if "battery" in c.result else T.WARN
        cells = [
            ("Layout seed", str(c.env.layout_seed)), ("Task seed", str(c.task_seed)),
            ("Step", f"{c.steps} / {env.config.max_steps}"), ("Reward", f"{c.ep_reward:.1f}  ({c.last_reward:+.1f})"),
            ("Collisions", str(c.collisions)), ("Invalid actions", str(c.invalid)),
            ("Energy used (moves)", str(c.energy)), ("Carrying", "yes" if env.carrying else "no"),
            ("Package", c.package_state), ("Result", c.result or "running"),
        ]
        for i, (a, b) in enumerate(cells):
            label_value(surf, x0 + (i % 2) * (cw + 8), y + (i // 2) * 32, cw, a, b,
                        res_col if a == "Result" else T.TEXT)
        y += 32 * 5
        # battery bar
        T.draw_text(surf, "Battery", (x0, y), "label", T.MUTED)
        bar = pygame.Rect(x0, y + 14, cw - 70, 12)
        if env.config.battery_enabled:
            frac = env.battery / max(1, cap)
            self.bar.rect, self.bar.value = bar, frac
            self.bar.color = T.ACCENT if frac > 0.25 else T.ERR
            self.bar.text = ""
            self.bar.draw(surf)
            T.draw_text(surf, f"{env.battery} / {cap}", (bar.right + 8, bar.centery), "body", T.TEXT, "midleft")
        else:
            T.draw_text(surf, "disabled", (bar.x, bar.centery), "body", T.MUTED, "midleft")
        la = "-" if c.last_action is None else f"{ACTION_NAMES[c.last_action]}  ->  {c.last_event}"
        T.draw_text(surf, "Last action", (x0 + cw + 8, y), "label", T.MUTED)
        T.draw_text(surf, la, (x0 + cw + 8, y + 13), "body", T.TEXT, max_w=cw)

    def _modal_rect(self):
        w, h = 520, 420
        return pygame.Rect(self.rect.centerx - w // 2, self.rect.centery - h // 2, w, h)

    def _draw_modal(self, surf):
        veil = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        veil.fill((30, 34, 40, 90))
        surf.blit(veil, (0, 0))
        mr = self._modal_rect()
        T.panel(surf, mr, T.SURFACE, T.BORDER_STRONG)
        T.section_header(surf, "Load model", mr.x + 12, mr.y + 10, mr.w - 24)
        self.fl.rect = pygame.Rect(mr.x + 12, mr.y + 44, mr.w - 24, mr.h - 44 - 52)
        self.btn_m_load.rect = pygame.Rect(mr.right - 12 - 80 - 8 - 80, mr.bottom - 40, 80, 28)
        self.btn_m_cancel.rect = pygame.Rect(mr.right - 12 - 80, mr.bottom - 40, 80, 28)
        T.draw_text(surf, f"{len(self.m_models)} saved models in models/", (mr.x + 12, mr.bottom - 32), "label", T.MUTED)
        self.mg.draw(surf)

    # ------------------------------------------------------------ events
    def handle_event(self, e):
        if self.modal:
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                self.close_browser()
                return True
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._browser_load()
                return True
            self.mg.dispatch(e)
            return True
        if self.group.dispatch(e):
            return True
        if e.type == pygame.KEYDOWN and not self.editing():
            k = e.key
            if k == pygame.K_SPACE:
                if self.btn_run.enabled:
                    self.toggle_run()
            elif k == pygame.K_r:
                self.reset()
            elif k == pygame.K_s:
                if self.btn_step.enabled:
                    self.step_once()
            elif k == pygame.K_n:
                self.new_task()
            elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.sl_speed.value = min(MAX_SPEED, self.sl_speed.value + (1 if self.speed < 10 else 5))
                self.speed = self.sl_speed.value
            elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.sl_speed.value = max(1, self.sl_speed.value - (1 if self.speed <= 10 else 5))
                self.speed = self.sl_speed.value
            else:
                return False
            return True
        return False

    def hint(self):
        return "Space start/pause   S step   R reset   N new task   +/- speed   Tab next tab"

    def cell_at(self, pos):
        if not self.grid_rect.collidepoint(pos):
            return None
        return ((pos[1] - self.grid_rect.y) // self.cell, (pos[0] - self.grid_rect.x) // self.cell)

    def tooltip(self, pos):
        if self.modal:
            return ""
        cell = self.cell_at(pos)
        if cell is None:
            return self.group.tip_at(pos)
        r, c = cell
        s = self.ctrl.env.get_render_state()
        H, W = s["grid"].shape
        if not (0 <= r < H and 0 <= c < W):
            return ""
        v = int(s["grid"][r, c])
        if v == 1:
            kind = "Wall"
        elif v == 2:
            kind = "Shelf"
        elif (r, c) in map(tuple, s["pickup_stations"]):
            kind = "Pickup station" + (" (active)" if (r, c) == tuple(s["active_pickup"]) else "")
        elif (r, c) in map(tuple, s["delivery_stations"]):
            kind = "Delivery station" + (" (active)" if (r, c) == tuple(s["active_delivery"]) else "")
        elif (r, c) in map(tuple, s["chargers"]):
            kind = "Charger"
        else:
            kind = "Floor"
        return f"{kind}  (row {r}, col {c})"
