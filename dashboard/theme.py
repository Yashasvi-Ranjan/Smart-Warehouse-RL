"""Visual constants and text helpers. One font family, three sizes, one blue accent."""
from __future__ import annotations

import os

import pygame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BG = (238, 239, 241)
PANEL = (248, 248, 249)
SURFACE = (255, 255, 255)
BORDER = (203, 206, 211)
BORDER_STRONG = (160, 165, 173)
TEXT = (30, 34, 40)
MUTED = (98, 106, 117)
DISABLED = (170, 175, 182)
ACCENT = (36, 94, 168)
ACCENT_DARK = (26, 70, 128)
ACCENT_SOFT = (222, 232, 245)
HOVER = (230, 232, 236)
OK = (58, 126, 84)
WARN = (176, 112, 36)
ERR = (170, 52, 48)

SP = 4  # spacing unit
SIZES = {"body": 13, "label": 11, "head": 15}
FAMILY = "helveticaneue,helvetica,arial,dejavusans,liberationsans"

_fonts: dict = {}


def font(kind: str = "body") -> pygame.font.Font:
    f = _fonts.get(kind)
    if f is None:
        if not pygame.font.get_init():
            pygame.font.init()
        try:
            f = pygame.font.SysFont(FAMILY, SIZES[kind], bold=(kind == "head"))
        except Exception:
            f = pygame.font.Font(None, SIZES[kind] + 4)
        _fonts[kind] = f
    return f


def reset_fonts() -> None:
    _fonts.clear()


def fit(text: str, width: int, kind: str = "body") -> str:
    """Truncate with an ellipsis so the rendered text is at most ``width`` px."""
    f = font(kind)
    if width <= 0:
        return ""
    if f.size(text)[0] <= width:
        return text
    ell = "..."
    while text and f.size(text + ell)[0] > width:
        text = text[:-1]
    return text + ell if text else ""


def draw_text(surf, text, pos, kind="body", color=TEXT, anchor="topleft", max_w=None):
    text = str(text)
    if max_w is not None:
        text = fit(text, max_w, kind)
    img = font(kind).render(text, True, color)
    r = img.get_rect(**{anchor: pos})
    surf.blit(img, r)
    return r


def text_w(text: str, kind: str = "body") -> int:
    return font(kind).size(str(text))[0]


def line_h(kind: str = "body") -> int:
    return font(kind).get_linesize()


def panel(surf, rect, fill=PANEL, border=BORDER):
    pygame.draw.rect(surf, fill, rect, border_radius=2)
    pygame.draw.rect(surf, border, rect, 1, border_radius=2)


def section_header(surf, text, x, y, w):
    draw_text(surf, text, (x, y), "head", TEXT)
    yy = y + line_h("head") + 2
    pygame.draw.line(surf, BORDER, (x, yy), (x + w, yy))
    return yy + SP * 2
