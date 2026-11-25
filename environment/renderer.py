"""Pygame drawing of a warehouse state (neutral, restrained palette, no gradients).

``draw_warehouse(surface, state, cell_size)`` draws onto any pygame surface of size
``(W * cell_size, H * cell_size)``; ``state`` is ``WarehouseEnv.get_render_state()``.

Optional extra keys understood in ``state`` (all ignored when absent, so the env's rgb_array render is
unchanged): ``trail`` (list of visited (r, c) cells), ``robot_visual_pos`` (float (r, c) used instead of
``robot_pos`` for smooth interpolation), ``show_trail`` (bool, default True when ``trail`` is present).
``LEGEND`` lists (label, colour) pairs for UI legends.
"""
from __future__ import annotations

FLOOR_C = (236, 236, 234)
GRID_C = (214, 214, 212)
WALL_C = (78, 82, 90)
SHELF_C = (184, 158, 124)
SHELF_EDGE_C = (150, 124, 94)
PICKUP_C = (218, 172, 84)
DELIVERY_C = (110, 160, 124)
CHARGER_C = (92, 154, 156)
PICKUP_DIM_C = (234, 220, 190)
DELIVERY_DIM_C = (206, 224, 210)
ROBOT_C = (44, 96, 168)
ROBOT_EDGE_C = (26, 60, 108)
PACKAGE_C = (214, 170, 90)
PACKAGE_EDGE_C = (120, 84, 40)
TRAIL_C = (176, 196, 224)
BAR_BG_C = (200, 200, 198)
BAR_OK_C = (96, 150, 110)
BAR_LOW_C = (196, 108, 88)
ACTIVE_C = (36, 38, 44)
TEXT_C = (50, 52, 58)

LEGEND = [("Wall", WALL_C), ("Shelf", SHELF_C), ("Pickup", PICKUP_C), ("Delivery", DELIVERY_C),
          ("Charger", CHARGER_C), ("Robot", ROBOT_C)]

_font_cache: dict = {}


def _font(size: int):
    import pygame
    f = _font_cache.get(size)
    if f is None:
        try:
            if not pygame.font.get_init():
                pygame.font.init()
            try:
                f = pygame.font.SysFont("helveticaneue,helvetica,arial,dejavusans", size, bold=True)
            except Exception:
                f = pygame.font.Font(None, size)
        except Exception:  # fonts unavailable -> draw without letters
            f = False
        _font_cache[size] = f
    return f or None


def _letter(surface, ch: str, rect, size: int) -> None:
    f = _font(size)
    if f is None:
        return
    img = f.render(ch, True, TEXT_C)
    surface.blit(img, img.get_rect(center=rect.center))


def draw_warehouse(surface, state: dict, cell_size: int) -> None:
    import pygame
    cs = int(cell_size)
    grid = state["grid"]
    H, W = grid.shape
    surface.fill(FLOOR_C)
    for r in range(H):
        for c in range(W):
            v = int(grid[r, c])
            rect = pygame.Rect(c * cs, r * cs, cs, cs)
            if v == 1:
                pygame.draw.rect(surface, WALL_C, rect)
            elif v == 2:
                pygame.draw.rect(surface, SHELF_C, rect)
                inner = rect.inflate(-max(4, cs // 4), -max(4, cs // 4))
                pygame.draw.rect(surface, SHELF_EDGE_C, inner, 1)
                for k in (1, 2):  # two shelf boards
                    y = inner.y + k * inner.h // 3
                    pygame.draw.line(surface, SHELF_EDGE_C, (inner.x, y), (inner.right - 1, y), 1)
    trail = state.get("trail")
    if trail and state.get("show_trail", True) and len(trail) > 1:
        for (r, c) in trail[:-1]:
            tr = pygame.Rect(c * cs, r * cs, cs, cs).inflate(-cs * 2 // 3, -cs * 2 // 3)
            pygame.draw.rect(surface, TRAIL_C, tr)
    pad = max(2, cs // 10)
    ap, ad = tuple(state["active_pickup"]), tuple(state["active_delivery"])
    carrying = bool(state["carrying"])
    for kind, cells, color, dim, ch in (("p", state["pickup_stations"], PICKUP_C, PICKUP_DIM_C, "P"),
                                        ("d", state["delivery_stations"], DELIVERY_C, DELIVERY_DIM_C, "D"),
                                        ("c", state["chargers"], CHARGER_C, CHARGER_C, "C")):
        for (r, c) in cells:
            rect = pygame.Rect(c * cs, r * cs, cs, cs).inflate(-2 * pad, -2 * pad)
            active = (kind == "p" and (r, c) == ap) or (kind == "d" and (r, c) == ad)
            pygame.draw.rect(surface, color if (active or kind == "c") else dim, rect)
            _letter(surface, ch, rect, max(11, cs // 3))
            if active:
                pygame.draw.rect(surface, ACTIVE_C, rect.inflate(pad, pad), max(1, cs // 24) + 1)
    for r in range(H + 1):
        pygame.draw.line(surface, GRID_C, (0, r * cs), (W * cs, r * cs), 1)
    for c in range(W + 1):
        pygame.draw.line(surface, GRID_C, (c * cs, 0), (c * cs, H * cs), 1)
    # robot (blue square with a white centre dot; no heading)
    rr, rc = state.get("robot_visual_pos") or state["robot_pos"]
    cell = pygame.Rect(int(round(rc * cs)), int(round(rr * cs)), cs, cs)
    rect = cell.inflate(-cs // 3, -cs // 3)
    pygame.draw.rect(surface, ROBOT_C, rect, border_radius=max(2, cs // 12))
    pygame.draw.rect(surface, ROBOT_EDGE_C, rect, 1, border_radius=max(2, cs // 12))
    pygame.draw.circle(surface, (236, 240, 247), rect.center, max(2, cs // 14))
    if carrying:
        s = max(6, cs // 3)
        pk = pygame.Rect(0, 0, s, s)
        pk.center = (rect.right - s // 4, rect.top + s // 4)
        pygame.draw.rect(surface, PACKAGE_C, pk)
        pygame.draw.rect(surface, PACKAGE_EDGE_C, pk, 1)
    if state["battery_enabled"]:
        frac = max(0.0, min(1.0, state["battery"] / max(1, state["battery_capacity"])))
        bw, bh = cs - 2 * pad - 2, max(3, cs // 12)
        bx, by = cell.x + pad + 1, cell.y + cs - pad - bh - 1
        pygame.draw.rect(surface, BAR_BG_C, (bx, by, bw, bh))
        pygame.draw.rect(surface, BAR_OK_C if frac > 0.25 else BAR_LOW_C, (bx, by, int(bw * frac), bh))
