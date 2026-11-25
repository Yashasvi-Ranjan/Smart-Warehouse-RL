"""Shared view base class and small layout helpers."""
from __future__ import annotations

import io

import pygame

from dashboard import theme as T
from dashboard.widgets import Group


class BaseView:
    name = "view"

    def __init__(self, app):
        self.app = app
        self.group = Group()
        self.rect = pygame.Rect(0, 0, 100, 100)

    def layout(self, rect: pygame.Rect) -> None:
        self.rect = pygame.Rect(rect)

    def on_show(self) -> None:
        pass

    def update(self, dt: float) -> None:
        pass

    def draw(self, surf) -> None:
        pass

    def handle_event(self, e) -> bool:
        return self.group.dispatch(e)

    def editing(self) -> bool:
        return self.group.editing()

    def hint(self) -> str:
        return ""

    def tooltip(self, pos) -> str:
        return self.group.tip_at(pos)


def label_value(surf, x, y, w, label, value, color=T.TEXT):
    """Small muted label above a value (status grid cell). Returns the cell height."""
    T.draw_text(surf, label, (x, y), "label", T.MUTED, max_w=w)
    T.draw_text(surf, value, (x, y + 13), "body", color, max_w=w)
    return 32


def fig_to_surface(fig, size) -> pygame.Surface:
    """Render a matplotlib Figure through the Agg canvas into a pygame surface of ``size`` pixels."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    w, h = max(50, int(size[0])), max(50, int(size[1]))
    dpi = 100
    fig.set_dpi(dpi)
    fig.set_size_inches(w / dpi, h / dpi)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    cw, ch = canvas.get_width_height()
    return pygame.image.frombuffer(bytes(buf), (cw, ch), "RGBA").convert()


def wrap_text(text: str, width: int, kind="label") -> list:
    f = T.font(kind)
    lines, cur = [], ""
    for word in text.split():
        t = (cur + " " + word).strip()
        if f.size(t)[0] <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines
