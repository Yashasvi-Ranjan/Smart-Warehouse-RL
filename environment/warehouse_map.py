"""Warehouse layout (grid), generator, BFS utilities and task solvability search.

Cell index convention: flat index ``i = r * W + c``. A layout is a pure function of
``(cfg.layout_key(), layout_seed)`` (see :func:`generate_layout`).
"""
from __future__ import annotations

from collections import OrderedDict, deque
from typing import Optional

import numpy as np

FLOOR, WALL, SHELF = 0, 1, 2
UNREACHABLE = 10 ** 6
# up, down, left, right
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


class WarehouseMap:
    """Immutable layout. ``grid`` is an int8 array (FLOOR/WALL/SHELF); stations are (row, col) tuples."""

    def __init__(self, grid: np.ndarray, pickup_stations, delivery_stations, chargers, layout_seed: int = -1):
        self.grid = np.asarray(grid, dtype=np.int8)
        self.grid.setflags(write=False)
        self.height, self.width = self.grid.shape
        self.layout_seed = int(layout_seed)
        self.pickup_stations = [tuple(map(int, p)) for p in pickup_stations]
        self.delivery_stations = [tuple(map(int, p)) for p in delivery_stations]
        self.chargers = [tuple(map(int, p)) for p in chargers]
        H, W = self.height, self.width
        self.blocked = self.grid != FLOOR
        self.n_cells = H * W
        flat_blocked = self.blocked.reshape(-1).tolist()
        # neighbour table: nbr[i][a] = flat index of target for action a, or -1 if blocked/out of bounds
        nbr = []
        for r in range(H):
            for c in range(W):
                row = []
                for dr, dc in MOVES:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and not flat_blocked[rr * W + cc]:
                        row.append(rr * W + cc)
                    else:
                        row.append(-1)
                nbr.append(row)
        self.nbr = nbr
        self.floor_cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(~self.blocked))]
        special = set(self.pickup_stations) | set(self.delivery_stations) | set(self.chargers)
        self.start_cells = [p for p in self.floor_cells if p not in special]
        self.is_charger = [False] * self.n_cells
        for r, c in self.chargers:
            self.is_charger[r * W + c] = True
        self._dist_cache: dict[int, list] = {}
        self._table_cache: dict[int, dict] = {}
        for p in self.pickup_stations + self.delivery_stations + self.chargers:
            self.dist_map(p)

    # ---------------------------------------------------------------- aliases / helpers
    @property
    def pickups(self):
        return self.pickup_stations

    @property
    def deliveries(self):
        return self.delivery_stations

    def idx(self, pos) -> int:
        return int(pos[0]) * self.width + int(pos[1])

    def pos(self, idx: int) -> tuple:
        return divmod(int(idx), self.width)

    def is_blocked(self, r: int, c: int) -> bool:
        return not (0 <= r < self.height and 0 <= c < self.width) or bool(self.blocked[r, c])

    def signature(self) -> bytes:
        return self.grid.tobytes() + repr((self.pickup_stations, self.delivery_stations, self.chargers)).encode()

    # ---------------------------------------------------------------- BFS
    def bfs(self, src_idx: int) -> list:
        """Flat list of shortest path lengths (in moves) from ``src_idx``; UNREACHABLE for blocked/unreached."""
        n = self.n_cells
        dist = [UNREACHABLE] * n
        dist[src_idx] = 0
        q = deque([src_idx])
        nbr = self.nbr
        while q:
            u = q.popleft()
            du = dist[u] + 1
            for v in nbr[u]:
                if v >= 0 and dist[v] == UNREACHABLE:
                    dist[v] = du
                    q.append(v)
        return dist

    def dist_map(self, pos) -> list:
        """Cached flat BFS distance list *to/from* ``pos`` (grid is undirected)."""
        i = self.idx(pos)
        d = self._dist_cache.get(i)
        if d is None:
            d = self._dist_cache[i] = self.bfs(i)
        return d

    def distance(self, a, b) -> int:
        return self.dist_map(a)[self.idx(b)]

    def shortest_path(self, a, b) -> Optional[list]:
        """List of (r, c) cells from ``a`` to ``b`` inclusive, or None."""
        dist = self.dist_map(b)
        cur = self.idx(a)
        if dist[cur] >= UNREACHABLE:
            return None
        path = [self.pos(cur)]
        while dist[cur] > 0:
            for v in self.nbr[cur]:
                if v >= 0 and dist[v] == dist[cur] - 1:
                    cur = v
                    break
            path.append(self.pos(cur))
        return path

    def is_connected(self) -> bool:
        if not self.floor_cells:
            return False
        d = self.bfs(self.idx(self.floor_cells[0]))
        return all(d[self.idx(p)] < UNREACHABLE for p in self.floor_cells)

    # ---------------------------------------------------------------- solvability
    def min_task_steps(self, start, pickup, delivery, battery: int, capacity: int,
                       recharge_amount: int, battery_enabled: bool = True) -> Optional[int]:
        """Minimum number of actions to pick up and deliver under the battery rules, or None if impossible.

        Exact search over (cell, battery, carrying) with recharging allowed. Battery rules: a move
        costs 1; a move that leaves battery 0 on a non-charger cell is a depletion (terminal, never
        part of a solution); moves at battery 0 are depletion. Includes the pickup and deliver actions.
        """
        s, p, dl = self.idx(start), self.idx(pickup), self.idx(delivery)
        if not battery_enabled:
            d1 = self.dist_map(pickup)[s]
            d2 = self.dist_map(delivery)[p]
            if d1 >= UNREACHABLE or d2 >= UNREACHABLE:
                return None
            return d1 + 1 + d2 + 1
        nbr, is_chg = self.nbr, self.is_charger
        B = capacity + 1
        seen = bytearray(self.n_cells * B * 2)
        b0 = min(int(battery), capacity)
        start_state = (s * B + b0) * 2
        seen[start_state] = 1
        frontier = [(s, b0, 0)]
        steps = 0
        while frontier:
            nxt = []
            steps += 1
            for cell, b, ph in frontier:
                if ph == 0 and cell == p:  # pickup action
                    k = (cell * B + b) * 2 + 1
                    if not seen[k]:
                        seen[k] = 1
                        nxt.append((cell, b, 1))
                if ph == 1 and cell == dl:
                    return steps  # deliver action
                if b >= 1:
                    nb = b - 1
                    for t in nbr[cell]:
                        if t < 0 or (nb == 0 and not is_chg[t]):
                            continue
                        k = (t * B + nb) * 2 + ph
                        if not seen[k]:
                            seen[k] = 1
                            nxt.append((t, nb, ph))
                if is_chg[cell] and b < capacity:
                    nb = min(capacity, b + recharge_amount)
                    k = (cell * B + nb) * 2 + ph
                    if not seen[k]:
                        seen[k] = 1
                        nxt.append((cell, nb, ph))
            frontier = nxt
        return None

    # ---------------------------------------------------------------- observation tables
    def tables(self, view_radius: int) -> dict:
        """Precomputed per-cell tables used by the env hot path (cached per view radius)."""
        t = self._table_cache.get(view_radius)
        if t is not None:
            return t
        H, W, r = self.height, self.width, int(view_radius)
        pad = np.pad(self.blocked, r, constant_values=True)
        offs = [(dr, dc) for dr in range(-r, r + 1) for dc in range(-r, r + 1) if (dr, dc) != (0, 0)]
        win = np.empty((H * W, len(offs)), dtype=np.float32)
        for k, (dr, dc) in enumerate(offs):
            win[:, k] = pad[r + dr: r + dr + H, r + dc: r + dc + W].reshape(-1)
        blk4 = np.empty((H * W, 4), dtype=np.float32)
        for k, (dr, dc) in enumerate(MOVES):
            pad1 = np.pad(self.blocked, 1, constant_values=True)
            blk4[:, k] = pad1[1 + dr: 1 + dr + H, 1 + dc: 1 + dc + W].reshape(-1)
        open4 = blk4 < 0.5
        H1, W1 = H - 1, W - 1
        near = np.zeros((H * W, 2), dtype=np.float32)
        if self.chargers:
            for rr in range(H):
                for cc in range(W):
                    best = min(range(len(self.chargers)),
                               key=lambda k: (abs(self.chargers[k][0] - rr) + abs(self.chargers[k][1] - cc), k))
                    near[rr * W + cc] = (self.chargers[best][0] / H1, self.chargers[best][1] / W1)
        t = {"win": win, "blk4": blk4, "open4": open4, "near_charger": near,
             "row_norm": [i / H1 for i in range(H)], "col_norm": [i / W1 for i in range(W)]}
        self._table_cache[view_radius] = t
        return t

    # ---------------------------------------------------------------- text
    def ascii(self) -> str:
        ch = {FLOOR: ".", WALL: "#", SHELF: "S"}
        rows = [[ch[int(v)] for v in row] for row in self.grid]
        for r, c in self.pickup_stations:
            rows[r][c] = "P"
        for r, c in self.delivery_stations:
            rows[r][c] = "D"
        for r, c in self.chargers:
            rows[r][c] = "C"
        return "\n".join("".join(r) for r in rows)


# ======================================================================================
# generator
# ======================================================================================
def _spread(rng, cands: list, n: int, min_dist: int) -> Optional[list]:
    """Pick n distinct cells preferring pairwise Manhattan distance >= min_dist; relax if impossible."""
    if len(cands) < n:
        return None
    order = [cands[i] for i in rng.permutation(len(cands))]
    chosen: list = []
    for md in range(min_dist, -1, -1):
        for p in order:
            if len(chosen) == n:
                break
            if p in chosen:
                continue
            if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) >= md for q in chosen):
                chosen.append(p)
        if len(chosen) == n:
            break
    return chosen if len(chosen) == n else None


def _try_generate(cfg, rng, layout_seed: int) -> Optional[WarehouseMap]:
    H, W = cfg.grid_h, cfg.grid_w
    grid = np.full((H, W), WALL, dtype=np.int8)
    grid[1:H - 1, 1:W - 1] = FLOOR
    # Shelf bands (rows of shelf blocks) separated by one-cell aisles. Row 1 = top aisle, row H-2 = dock row.
    r = 2
    while r <= H - 4:
        hmax = min(2, H - 4 - r + 1)
        h = 2 if (hmax == 2 and rng.random() < 0.75) else 1
        if rng.random() >= 0.08:
            c = 2 + int(rng.integers(0, 2))
            while c <= W - 3:
                length = int(rng.integers(2, 5))
                end = min(c + length - 1, W - 3)
                if end - c + 1 >= 2 and rng.random() > 0.08:
                    grid[r:r + h, c:end + 1] = SHELF
                c = end + 1 + (1 if rng.random() < 0.85 else 2)
        r += h + 1
    if int((grid == SHELF).sum()) < max(6, int(0.22 * (H - 2) * (W - 2))):
        return None
    # pickup stations: floor cells adjacent to a shelf (not on the dock row)
    cands = []
    for rr in range(1, H - 2):
        for cc in range(1, W - 1):
            if grid[rr, cc] == FLOOR and (grid[rr - 1, cc] == SHELF or grid[rr + 1, cc] == SHELF
                                          or grid[rr, cc - 1] == SHELF or grid[rr, cc + 1] == SHELF):
                cands.append((rr, cc))
    pickups = _spread(rng, cands, cfg.n_pickup_stations, 3)
    if pickups is None:
        return None
    # delivery stations: dock row H-2, one random column in each of n equal column segments
    cols = np.array_split(np.arange(1, W - 1), cfg.n_delivery_stations)
    deliveries = [(H - 2, int(rng.choice(seg))) for seg in cols]
    # chargers: along the outer aisles (next to the walls), excluding dock row and stations
    taken = set(pickups) | set(deliveries)
    ccands = []
    for rr in range(1, H - 2):
        for cc in range(1, W - 1):
            if (rr == 1 or cc == 1 or cc == W - 2) and grid[rr, cc] == FLOOR and (rr, cc) not in taken:
                ccands.append((rr, cc))
    chargers = _spread(rng, ccands, cfg.n_chargers, max(2, (H + W) // 4)) if cfg.n_chargers else []
    if chargers is None:
        return None
    m = WarehouseMap(grid, pickups, deliveries, chargers, layout_seed)
    return m if m.is_connected() else None


def generate_layout(cfg, layout_seed: int) -> WarehouseMap:
    """Deterministic layout from ``(cfg, layout_seed)``: outer walls, rows of shelf blocks with aisles,
    pickup stations beside shelves, delivery docks on the bottom row, chargers along the outer aisles.
    Floor connectivity is verified (retry with a derived sub-seed if not, still deterministic)."""
    layout_seed = int(layout_seed)
    if layout_seed < 0:
        raise ValueError("layout_seed must be >= 0")
    for attempt in range(500):
        rng = np.random.default_rng([layout_seed, attempt])
        m = _try_generate(cfg, rng, layout_seed)
        if m is not None:
            return m
    raise RuntimeError(f"Could not generate a valid layout for seed {layout_seed} with {cfg.to_dict()}")


_CACHE: "OrderedDict[tuple, WarehouseMap]" = OrderedDict()
_CACHE_MAX = 512


def get_layout(cfg, layout_seed: int) -> WarehouseMap:
    """Cached :func:`generate_layout` (WarehouseMap is immutable)."""
    key = (cfg.layout_key(), int(layout_seed))
    m = _CACHE.get(key)
    if m is not None:
        _CACHE.move_to_end(key)
        return m
    m = generate_layout(cfg, layout_seed)
    _CACHE[key] = m
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return m
