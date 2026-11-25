"""Small immediate-style widget set. Every widget owns a ``rect`` and implements
``handle_event(event) -> bool`` (True = consumed) and ``draw(surface)``."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import pygame

from dashboard import theme as T

ROW_H = 24


class Widget:
    tooltip: str = ""

    def __init__(self, rect=(0, 0, 10, ROW_H)):
        self.rect = pygame.Rect(rect)
        self.visible = True
        self.enabled = True
        self.hover = False

    def handle_event(self, e) -> bool:
        return False

    def draw(self, surf) -> None:
        pass

    def draw_popup(self, surf) -> None:
        pass

    @property
    def captures(self) -> bool:  # open popups capture all events
        return False

    def blur(self) -> None:
        pass


class Group:
    """Ordered set of widgets: dispatches events (popups first), draws, finds tooltips."""

    def __init__(self):
        self.widgets: list[Widget] = []
        self.focus: Optional[Widget] = None

    def add(self, *ws):
        for w in ws:
            self.widgets.append(w)
        return ws[0] if len(ws) == 1 else ws

    def dispatch(self, e) -> bool:
        live = [w for w in self.widgets if w.visible]
        for w in live:
            if w.captures:
                return w.handle_event(e) or e.type in (pygame.MOUSEBUTTONDOWN,)
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            for w in live:
                if isinstance(w, TextField) and w.focused and not w.rect.collidepoint(e.pos):
                    w.blur()
        if e.type == pygame.MOUSEMOTION:
            for w in live:
                w.hover = w.rect.collidepoint(e.pos) and w.enabled
        for w in reversed(live):
            if w.handle_event(e):
                return True
        return False

    def draw(self, surf):
        for w in self.widgets:
            if w.visible:
                w.draw(surf)
        for w in self.widgets:
            if w.visible:
                w.draw_popup(surf)

    def tip_at(self, pos) -> str:
        for w in self.widgets:
            if w.visible and w.tooltip and w.rect.collidepoint(pos):
                return w.tooltip
        return ""

    def editing(self) -> bool:
        return any(isinstance(w, TextField) and w.visible and w.focused for w in self.widgets)


class Label(Widget):
    def __init__(self, text="", rect=(0, 0, 10, 16), kind="label", color=T.MUTED):
        super().__init__(rect)
        self.text, self.kind, self.color = text, kind, color

    def draw(self, surf):
        T.draw_text(surf, self.text, self.rect.topleft, self.kind, self.color, max_w=self.rect.w)


class Button(Widget):
    def __init__(self, label, on_click: Callable[[], None] = None, rect=(0, 0, 80, ROW_H), primary=False,
                 tooltip=""):
        super().__init__(rect)
        self.label, self.on_click, self.primary, self.tooltip = label, on_click, primary, tooltip
        self.pressed = False

    def handle_event(self, e):
        if not self.enabled:
            return False
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.rect.collidepoint(e.pos):
            self.pressed = True
            return True
        if e.type == pygame.MOUSEBUTTONUP and e.button == 1 and self.pressed:
            self.pressed = False
            if self.rect.collidepoint(e.pos) and self.on_click:
                self.on_click()
            return True
        return False

    def draw(self, surf):
        if self.primary:
            fill = T.ACCENT_DARK if (self.pressed or self.hover) else T.ACCENT
            fg, border = (255, 255, 255), T.ACCENT_DARK
        else:
            fill = T.HOVER if (self.hover or self.pressed) else T.SURFACE
            fg, border = T.TEXT, T.BORDER_STRONG
        if not self.enabled:
            fill, fg, border = T.PANEL, T.DISABLED, T.BORDER
        pygame.draw.rect(surf, fill, self.rect, border_radius=2)
        pygame.draw.rect(surf, border, self.rect, 1, border_radius=2)
        T.draw_text(surf, self.label, self.rect.center, "body", fg, "center", self.rect.w - 8)


class Checkbox(Widget):
    def __init__(self, label, value=False, on_change=None, rect=(0, 0, 120, ROW_H), tooltip=""):
        super().__init__(rect)
        self.label, self.value, self.on_change, self.tooltip = label, bool(value), on_change, tooltip

    def handle_event(self, e):
        if self.enabled and e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.rect.collidepoint(e.pos):
            self.value = not self.value
            if self.on_change:
                self.on_change(self.value)
            return True
        return False

    def draw(self, surf):
        box = pygame.Rect(self.rect.x, self.rect.centery - 6, 12, 12)
        col = T.TEXT if self.enabled else T.DISABLED
        pygame.draw.rect(surf, T.SURFACE, box, border_radius=2)
        pygame.draw.rect(surf, T.ACCENT if self.value and self.enabled else T.BORDER_STRONG, box, 1, border_radius=2)
        if self.value:
            pygame.draw.rect(surf, T.ACCENT if self.enabled else T.DISABLED, box.inflate(-6, -6))
        T.draw_text(surf, self.label, (box.right + 6, self.rect.centery), "body", col, "midleft",
                    self.rect.w - 20)


class Dropdown(Widget):
    """options: list of (value, label[, enabled])."""

    def __init__(self, options: Sequence, value=None, on_change=None, rect=(0, 0, 140, ROW_H), tooltip=""):
        super().__init__(rect)
        self.options = list(options)
        self.value = value if value is not None else (self.options[0][0] if self.options else None)
        self.on_change, self.open, self.tooltip = on_change, False, tooltip
        self._hover_i = -1

    @property
    def captures(self):
        return self.open

    def set_options(self, options):
        self.options = list(options)

    def label_of(self, v):
        for o in self.options:
            if o[0] == v:
                return o[1]
        return ""

    def _list_rect(self):
        h = len(self.options) * ROW_H
        r = pygame.Rect(self.rect.x, self.rect.bottom, self.rect.w, h)
        sh = pygame.display.get_surface().get_height() if pygame.display.get_surface() else 800
        if r.bottom > sh - 4:
            r.bottom = self.rect.top
        return r

    def blur(self):
        self.open = False

    def handle_event(self, e):
        if not self.enabled and not self.open:
            return False
        if self.open:
            lr = self._list_rect()
            if e.type == pygame.MOUSEMOTION:
                self._hover_i = (e.pos[1] - lr.y) // ROW_H if lr.collidepoint(e.pos) else -1
                return True
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if lr.collidepoint(e.pos):
                    i = (e.pos[1] - lr.y) // ROW_H
                    if 0 <= i < len(self.options):
                        self.select(i)
                self.open = False
                return True
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                self.open = False
                return True
            return e.type != pygame.MOUSEBUTTONUP
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.rect.collidepoint(e.pos):
            self.open = True
            self._hover_i = -1
            return True
        return False

    def select(self, i: int):
        v = self.options[i][0]
        changed = v != self.value
        self.value = v
        if changed and self.on_change:
            self.on_change(v)

    def draw(self, surf):
        fill = T.SURFACE if self.enabled else T.PANEL
        pygame.draw.rect(surf, fill, self.rect, border_radius=2)
        pygame.draw.rect(surf, T.ACCENT if self.open else T.BORDER_STRONG, self.rect, 1, border_radius=2)
        T.draw_text(surf, self.label_of(self.value), (self.rect.x + 8, self.rect.centery), "body",
                    T.TEXT if self.enabled else T.DISABLED, "midleft", self.rect.w - 26)
        cx, cy = self.rect.right - 12, self.rect.centery
        pygame.draw.polygon(surf, T.MUTED, [(cx - 4, cy - 2), (cx + 4, cy - 2), (cx, cy + 3)])

    def draw_popup(self, surf):
        if not self.open:
            return
        lr = self._list_rect()
        pygame.draw.rect(surf, T.SURFACE, lr)
        pygame.draw.rect(surf, T.BORDER_STRONG, lr, 1)
        for i, o in enumerate(self.options):
            r = pygame.Rect(lr.x + 1, lr.y + i * ROW_H, lr.w - 2, ROW_H)
            ok = o[2] if len(o) > 2 else True
            if i == self._hover_i:
                pygame.draw.rect(surf, T.ACCENT_SOFT, r)
            if o[0] == self.value:
                pygame.draw.rect(surf, T.ACCENT, (r.x, r.y + 3, 2, r.h - 6))
            T.draw_text(surf, o[1], (r.x + 8, r.centery), "body", T.TEXT if ok else T.DISABLED, "midleft", r.w - 12)


class TextField(Widget):
    """Single-line input. numeric: None | 'int' | 'float' (validated; invalid -> red border)."""

    def __init__(self, text="", numeric=None, rect=(0, 0, 80, ROW_H), on_commit=None, minv=None, maxv=None,
                 allow_empty=False, tooltip=""):
        super().__init__(rect)
        self.text, self.numeric, self.on_commit = str(text), numeric, on_commit
        self.minv, self.maxv, self.allow_empty, self.tooltip = minv, maxv, allow_empty, tooltip
        self.focused = False
        self.cursor = len(self.text)
        self._orig = self.text

    def set_text(self, t):
        self.text = str(t)
        self.cursor = len(self.text)

    def parse(self):
        """Return (ok, value). Empty with allow_empty -> (True, None)."""
        s = self.text.strip()
        if s == "":
            return (True, None) if self.allow_empty or self.numeric is None else (False, None)
        if self.numeric is None:
            return True, s
        try:
            v = int(s) if self.numeric == "int" else float(s)
        except ValueError:
            return False, None
        if (self.minv is not None and v < self.minv) or (self.maxv is not None and v > self.maxv):
            return False, None
        return True, v

    @property
    def valid(self):
        return self.parse()[0]

    @property
    def value(self):
        return self.parse()[1]

    def blur(self):
        if self.focused:
            self.focused = False
            if self.on_commit and self.valid:
                self.on_commit(self.value)

    def handle_event(self, e):
        if not self.enabled:
            return False
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.rect.collidepoint(e.pos):
                self.focused = True
                self._orig = self.text
                self.cursor = len(self.text)
                return True
            return False
        if not self.focused:
            return False
        if e.type == pygame.KEYDOWN:
            k = e.key
            if k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.blur()
            elif k == pygame.K_ESCAPE:
                self.text = self._orig
                self.focused = False
            elif k == pygame.K_BACKSPACE:
                if self.cursor > 0:
                    self.text = self.text[:self.cursor - 1] + self.text[self.cursor:]
                    self.cursor -= 1
            elif k == pygame.K_DELETE:
                self.text = self.text[:self.cursor] + self.text[self.cursor + 1:]
            elif k == pygame.K_LEFT:
                self.cursor = max(0, self.cursor - 1)
            elif k == pygame.K_RIGHT:
                self.cursor = min(len(self.text), self.cursor + 1)
            elif k == pygame.K_HOME:
                self.cursor = 0
            elif k == pygame.K_END:
                self.cursor = len(self.text)
            elif k == pygame.K_TAB:
                return False
            else:
                ch = e.unicode
                if ch and ch.isprintable():
                    if self.numeric and ch not in "0123456789.-+eE":
                        return True
                    if self.numeric == "int" and ch in ".eE":
                        return True
                    self.text = self.text[:self.cursor] + ch + self.text[self.cursor:]
                    self.cursor += 1
            return True
        return False

    def draw(self, surf):
        ok = self.valid
        border = T.ERR if not ok else (T.ACCENT if self.focused else T.BORDER_STRONG)
        pygame.draw.rect(surf, T.SURFACE if self.enabled else T.PANEL, self.rect, border_radius=2)
        pygame.draw.rect(surf, border, self.rect, 1, border_radius=2)
        f = T.font("body")
        inner = self.rect.w - 12
        shown = self.text
        off = 0
        while f.size(shown[off:self.cursor if self.focused else len(shown)])[0] > inner and off < len(shown):
            off += 1
        shown = shown[off:]
        clip = surf.get_clip()
        surf.set_clip(self.rect.inflate(-4, -2))
        T.draw_text(surf, shown, (self.rect.x + 6, self.rect.centery), "body",
                    T.TEXT if self.enabled else T.DISABLED, "midleft")
        if self.focused:
            cx = self.rect.x + 6 + f.size(shown[:self.cursor - off])[0]
            pygame.draw.line(surf, T.TEXT, (cx, self.rect.y + 5), (cx, self.rect.bottom - 5))
        surf.set_clip(clip)


class Slider(Widget):
    def __init__(self, vmin, vmax, value, on_change=None, rect=(0, 0, 140, ROW_H), fmt=str, tooltip=""):
        super().__init__(rect)
        self.vmin, self.vmax, self.value, self.on_change, self.fmt, self.tooltip = vmin, vmax, int(value), on_change, fmt, tooltip
        self.drag = False

    def _set_from(self, x):
        track = self.rect.inflate(-12, 0)
        f = min(1.0, max(0.0, (x - track.x) / max(1, track.w)))
        v = int(round(self.vmin + f * (self.vmax - self.vmin)))
        if v != self.value:
            self.value = v
            if self.on_change:
                self.on_change(v)

    def handle_event(self, e):
        if not self.enabled:
            return False
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.rect.collidepoint(e.pos):
            self.drag = True
            self._set_from(e.pos[0])
            return True
        if e.type == pygame.MOUSEMOTION and self.drag:
            self._set_from(e.pos[0])
            return True
        if e.type == pygame.MOUSEBUTTONUP and e.button == 1 and self.drag:
            self.drag = False
            return True
        return False

    def draw(self, surf):
        track = self.rect.inflate(-12, 0)
        f = (self.value - self.vmin) / max(1, self.vmax - self.vmin)
        y = self.rect.centery
        pygame.draw.line(surf, T.BORDER_STRONG, (track.x, y), (track.right, y), 2)
        pygame.draw.line(surf, T.ACCENT, (track.x, y), (track.x + int(track.w * f), y), 2)
        kx = track.x + int(track.w * f)
        pygame.draw.rect(surf, T.SURFACE, (kx - 4, y - 7, 8, 14), border_radius=2)
        pygame.draw.rect(surf, T.ACCENT, (kx - 4, y - 7, 8, 14), 1, border_radius=2)


class ProgressBar(Widget):
    def __init__(self, rect=(0, 0, 200, 14)):
        super().__init__(rect)
        self.value = 0.0
        self.text = ""
        self.color = T.ACCENT

    def draw(self, surf):
        pygame.draw.rect(surf, T.SURFACE, self.rect)
        w = int((self.rect.w - 2) * min(1.0, max(0.0, self.value)))
        if w > 0:
            pygame.draw.rect(surf, self.color, (self.rect.x + 1, self.rect.y + 1, w, self.rect.h - 2))
        pygame.draw.rect(surf, T.BORDER_STRONG, self.rect, 1)
        if self.text:
            T.draw_text(surf, self.text, self.rect.center, "label",
                        (255, 255, 255) if self.value > 0.55 else T.TEXT, "center", self.rect.w - 6)


class Tabs(Widget):
    def __init__(self, labels: Sequence[str], active=0, on_change=None, rect=(0, 0, 400, 32)):
        super().__init__(rect)
        self.labels, self.active, self.on_change = list(labels), active, on_change
        self.tab_w = 112

    def _rects(self):
        return [pygame.Rect(self.rect.x + i * self.tab_w, self.rect.y, self.tab_w, self.rect.h)
                for i in range(len(self.labels))]

    def set_active(self, i):
        i %= len(self.labels)
        if i != self.active:
            self.active = i
            if self.on_change:
                self.on_change(i)

    def handle_event(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            for i, r in enumerate(self._rects()):
                if r.collidepoint(e.pos):
                    self.set_active(i)
                    return True
        return False

    def draw(self, surf):
        pygame.draw.line(surf, T.BORDER, (self.rect.x, self.rect.bottom - 1), (surf.get_width(), self.rect.bottom - 1))
        for i, r in enumerate(self._rects()):
            on = i == self.active
            T.draw_text(surf, self.labels[i], r.center, "body", T.TEXT if on else T.MUTED, "center")
            if on:
                pygame.draw.rect(surf, T.ACCENT, (r.x + 8, r.bottom - 3, r.w - 16, 3))


class Table(Widget):
    """columns: (key, header, width_weight, align). rows: list of dicts or sequences keyed by column."""

    def __init__(self, columns, rect=(0, 0, 300, 160), row_h=18, selectable=False, on_select=None):
        super().__init__(rect)
        self.columns, self.row_h, self.selectable, self.on_select = list(columns), row_h, selectable, on_select
        self.rows: list = []
        self.colors: dict = {}  # (row_index, key) -> colour
        self.selected = -1
        self.scroll = 0

    def visible_rows(self):
        return max(1, (self.rect.h - self.row_h) // self.row_h)

    def handle_event(self, e):
        if e.type == pygame.MOUSEWHEEL and self.rect.collidepoint(pygame.mouse.get_pos()):
            self.scroll = max(0, min(max(0, len(self.rows) - self.visible_rows()), self.scroll - e.y))
            return True
        if self.selectable and e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.rect.collidepoint(e.pos):
            i = (e.pos[1] - self.rect.y - self.row_h) // self.row_h + self.scroll
            if 0 <= i < len(self.rows) and e.pos[1] > self.rect.y + self.row_h:
                self.selected = i
                if self.on_select:
                    self.on_select(i)
            return True
        return False

    def draw(self, surf):
        pygame.draw.rect(surf, T.SURFACE, self.rect)
        pygame.draw.rect(surf, T.BORDER, self.rect, 1)
        total = sum(c[2] for c in self.columns)
        xs, x = [], self.rect.x
        for c in self.columns:
            w = int(self.rect.w * c[2] / total)
            xs.append((x, w))
            x += w
        hr = pygame.Rect(self.rect.x, self.rect.y, self.rect.w, self.row_h)
        pygame.draw.rect(surf, T.PANEL, hr)
        pygame.draw.line(surf, T.BORDER, hr.bottomleft, hr.bottomright)
        for (cx, cw), c in zip(xs, self.columns):
            self._cell(surf, c[1], cx, cw, hr.centery, c[3], "label", T.MUTED)
        clip = surf.get_clip()
        surf.set_clip(self.rect.inflate(-2, -2))
        n = self.visible_rows()
        for k in range(n):
            i = self.scroll + k
            if i >= len(self.rows):
                break
            y = self.rect.y + self.row_h * (k + 1)
            if i == self.selected:
                pygame.draw.rect(surf, T.ACCENT_SOFT, (self.rect.x + 1, y, self.rect.w - 2, self.row_h))
            row = self.rows[i]
            for (cx, cw), c in zip(xs, self.columns):
                v = row.get(c[0], "") if isinstance(row, dict) else row[self.columns.index(c)]
                self._cell(surf, v, cx, cw, y + self.row_h // 2, c[3], "body", self.colors.get((i, c[0]), T.TEXT))
        surf.set_clip(clip)

    def _cell(self, surf, text, cx, cw, cy, align, kind, color):
        if align == "right":
            T.draw_text(surf, text, (cx + cw - 6, cy), kind, color, "midright", cw - 8)
        else:
            T.draw_text(surf, text, (cx + 6, cy), kind, color, "midleft", cw - 8)


class FileList(Widget):
    """Scrollable list of items (id, primary, secondary). single-select, or ``checks`` for multi-select."""

    def __init__(self, rect=(0, 0, 300, 200), checks=False, on_change=None, row_h=32):
        super().__init__(rect)
        self.items: list = []
        self.checks, self.on_change, self.row_h = checks, on_change, row_h
        self.selected = -1
        self.checked: set = set()
        self.scroll = 0

    def set_items(self, items):
        self.items = list(items)
        ids = {i[0] for i in self.items}
        self.checked &= ids
        self.selected = min(self.selected, len(self.items) - 1)
        self.scroll = 0

    def selected_id(self):
        return self.items[self.selected][0] if 0 <= self.selected < len(self.items) else None

    def _n(self):
        return max(1, self.rect.h // self.row_h)

    def handle_event(self, e):
        if e.type == pygame.MOUSEWHEEL and self.rect.collidepoint(pygame.mouse.get_pos()):
            self.scroll = max(0, min(max(0, len(self.items) - self._n()), self.scroll - e.y))
            return True
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.rect.collidepoint(e.pos):
            i = (e.pos[1] - self.rect.y) // self.row_h + self.scroll
            if 0 <= i < len(self.items):
                if self.checks:
                    iid = self.items[i][0]
                    self.checked ^= {iid}
                self.selected = i
                if self.on_change:
                    self.on_change(i)
            return True
        return False

    def draw(self, surf):
        pygame.draw.rect(surf, T.SURFACE, self.rect)
        clip = surf.get_clip()
        surf.set_clip(self.rect.inflate(-2, -2))
        for k in range(self._n() + 1):
            i = self.scroll + k
            if i >= len(self.items):
                break
            r = pygame.Rect(self.rect.x, self.rect.y + k * self.row_h, self.rect.w, self.row_h)
            iid, a, b = self.items[i][:3]
            if i == self.selected:
                pygame.draw.rect(surf, T.ACCENT_SOFT, r.inflate(-2, 0))
            x = r.x + 8
            if self.checks:
                box = pygame.Rect(x, r.centery - 6, 12, 12)
                pygame.draw.rect(surf, T.SURFACE, box, border_radius=2)
                pygame.draw.rect(surf, T.BORDER_STRONG, box, 1, border_radius=2)
                if iid in self.checked:
                    pygame.draw.rect(surf, T.ACCENT, box.inflate(-6, -6))
                x += 20
            T.draw_text(surf, a, (x, r.y + 4), "body", T.TEXT, max_w=r.right - x - 6)
            T.draw_text(surf, b, (x, r.y + 18), "label", T.MUTED, max_w=r.right - x - 6)
            pygame.draw.line(surf, T.BORDER, (r.x, r.bottom - 1), (r.right, r.bottom - 1))
        surf.set_clip(clip)
        pygame.draw.rect(surf, T.BORDER_STRONG, self.rect, 1)
        if not self.items:
            T.draw_text(surf, "No items", self.rect.center, "body", T.MUTED, "center")
