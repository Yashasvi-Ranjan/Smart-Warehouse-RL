"""Application shell: window, tab strip, event loop, shared state."""
from __future__ import annotations

import time

import pygame

from dashboard import theme as T
from dashboard.analytics_view import AnalyticsView
from dashboard.controller import SimulationController
from dashboard.evaluation_view import EvaluationView
from dashboard.simulation_view import SimulationView
from dashboard.training_view import TrainingView
from dashboard.widgets import Tabs

TAB_H = 32
HINT_H = 22
TITLES = ["Simulation", "Training", "Analytics", "Evaluation"]


class App:
    def __init__(self, size=(1280, 800), headless=False):
        pygame.init()
        pygame.font.init()
        flags = pygame.RESIZABLE
        self.screen = pygame.display.set_mode(size, flags)
        pygame.display.set_caption("Smart Warehouse RL")
        pygame.display.set_allow_screensaver(True) if hasattr(pygame.display, "set_allow_screensaver") else None
        self.clock = pygame.time.Clock()
        self.sim = SimulationController()
        self.tabs = Tabs(TITLES, 0, self._tab_changed, (8, 0, 4 * 112, TAB_H))
        self.views = [SimulationView(self), TrainingView(self), AnalyticsView(self), EvaluationView(self)]
        self.current = 0
        self.running = True
        self.tip, self._tip_t, self._tip_pos = "", 0.0, (0, 0)
        self.min_size = (1100, 700)
        self.relayout()

    @property
    def view(self):
        return self.views[self.current]

    def content_rect(self):
        w, h = self.screen.get_size()
        return pygame.Rect(0, TAB_H, w, h - TAB_H - HINT_H)

    def relayout(self):
        w, h = self.screen.get_size()
        if w < self.min_size[0] or h < self.min_size[1]:
            w, h = max(w, self.min_size[0]), max(h, self.min_size[1])
            self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        for v in self.views:
            v.layout(self.content_rect())

    def _tab_changed(self, i):
        self.current = i
        self.view.on_show()
        self.view.layout(self.content_rect())
        for w in self.view.group.widgets:
            if hasattr(w, "blur"):
                w.blur()

    def switch_to(self, i):
        self.tabs.active = i
        self._tab_changed(i)

    # ---------------------------------------------------- cross-tab actions
    def use_model_in_sim(self, algo, path):
        sim = self.views[0]
        sim.load_model(algo, path)
        self.switch_to(0)

    def models_changed(self):
        self.views[0].refresh_algos()
        self.views[3].refresh_agents()

    def replay(self, algo, path, battery, reward, layout_seeds, seed):
        sim = self.views[0]
        self.sim.set_agent(algo, path)
        self.sim.configure(battery=battery, reward=reward, layout_seed=None, layout_seeds_set=layout_seeds)
        self.sim.reset(task_seed=seed)
        sim.sync_from_controller()
        sim.layout(self.content_rect())
        self.switch_to(0)

    # ---------------------------------------------------- loop
    def handle_event(self, e):
        if e.type == pygame.QUIT:
            self.running = False
            return
        if e.type == pygame.VIDEORESIZE or e.type == getattr(pygame, "WINDOWRESIZED", -1):
            self.relayout()
            return
        v = self.view
        if e.type == pygame.KEYDOWN and e.key == pygame.K_TAB and not v.editing():
            shift = e.mod & pygame.KMOD_SHIFT
            self.tabs.set_active(self.current + (-1 if shift else 1))
            return
        if self.tabs.handle_event(e):
            return
        v.handle_event(e)

    def tick(self, dt):
        if self.view is self.views[0] or True:
            self.view.update(dt)
        mouse = pygame.mouse.get_pos()
        if mouse != self._tip_pos:
            self._tip_pos, self._tip_t, self.tip = mouse, 0.0, ""
        else:
            self._tip_t += dt
            if self._tip_t > 0.5 and not self.tip:
                self.tip = self.view.tooltip(mouse)

    def draw(self):
        s = self.screen
        s.fill(T.BG)
        self.view.draw(s)
        # tab strip + hint line drawn on top
        w, h = s.get_size()
        pygame.draw.rect(s, T.PANEL, (0, 0, w, TAB_H))
        self.tabs.draw(s)
        T.draw_text(s, "Smart Warehouse RL", (w - 12, TAB_H // 2), "label", T.MUTED, "midright")
        pygame.draw.rect(s, T.PANEL, (0, h - HINT_H, w, HINT_H))
        pygame.draw.line(s, T.BORDER, (0, h - HINT_H), (w, h - HINT_H))
        T.draw_text(s, self.view.hint(), (12, h - HINT_H // 2), "label", T.MUTED, "midleft", w - 24)
        if self.tip:
            tw = T.text_w(self.tip, "label") + 12
            r = pygame.Rect(self._tip_pos[0] + 12, self._tip_pos[1] + 16, tw, 20)
            r.clamp_ip(s.get_rect())
            pygame.draw.rect(s, (40, 44, 50), r, border_radius=2)
            T.draw_text(s, self.tip, r.center, "label", (240, 242, 245), "center")

    def run(self):
        last = time.perf_counter()
        while self.running:
            for e in pygame.event.get():
                self.handle_event(e)
            now = time.perf_counter()
            self.tick(min(0.1, now - last))
            last = now
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

    def render_all_tabs(self):
        """Render every tab once (smoke test / screenshots)."""
        for i in range(len(self.views)):
            self.switch_to(i)
            self.view.update(0.016)
            self.view.update(0.5)
            self.draw()
