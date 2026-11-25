"""A* planning agent (classical baseline)."""
from __future__ import annotations

import heapq
from typing import Any, Optional

import numpy as np

from agents.base import Agent

UP, DOWN, LEFT, RIGHT, PICKUP, DELIVER, RECHARGE = range(7)
_MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


class AStarAgent(Agent):
    """PRIVILEGED classical baseline: it reads the full map, the active pickup/delivery cells, the
    chargers and the exact battery level from ``env.unwrapped``. It is an *upper-bound reference*
    for the learning agents, not an RL competitor.

    Planning is genuine A* (own binary-heap implementation, Manhattan-distance heuristic) over the
    state ``(cell, battery, carrying)`` on the grid from ``env.unwrapped.warehouse_map`` (walls and
    shelves are blocked). Actions are: move (cost 1, battery -1), pickup, deliver (cost 1) and
    recharge at a charger (cost 1, battery += recharge_amount). Because battery is part of the search
    state, the optimal plan automatically contains a charger detour (the charger and number of
    recharges minimising the total step count among feasible plans) when the battery cannot cover
    pickup -> delivery directly. With the battery disabled (or ample) the plan is the plain
    robot -> pickup -> pickup action -> delivery -> deliver action shortest path, whose length equals
    ``env.unwrapped.optimal_steps``. The plan is recomputed whenever the real state deviates from the
    state the plan predicted. If no plan exists the agent emits a harmless no-cost action.
    """

    name = "astar"

    def __init__(self, **_ignored: Any):
        self._plan: list = []       # list of (action, expected_state_after)
        self._i = 0
        self._expect = None         # state expected before next action
        self.replans = 0
        self.last_plan_failed = False

    def reset(self) -> None:
        self._plan, self._i, self._expect = [], 0, None

    # ------------------------------------------------------------------ search
    @staticmethod
    def plan(m, start: int, pick: int, deliv: int, carrying: bool, battery: int, capacity: int,
             recharge: int, battery_enabled: bool) -> Optional[list]:
        """A* from the given state. Returns [(action, state_after), ...] or None if unsolvable.
        State = (cell, battery, carrying); battery is fixed at 0 when the battery is disabled."""
        W = m.width
        nbr, is_chg = m.nbr, m.is_charger
        pr, pc = divmod(pick, W)
        dr_, dc_ = divmod(deliv, W)
        leg = abs(pr - dr_) + abs(pc - dc_)

        def h(cell: int, carry: int) -> int:
            r, c = divmod(cell, W)
            if carry:
                return abs(r - dr_) + abs(c - dc_) + 1
            return abs(r - pr) + abs(c - pc) + 1 + leg + 1

        b0 = min(battery, capacity) if battery_enabled else 0
        s0 = (start, b0, int(carrying))
        g = {s0: 0}
        parent: dict = {}
        heap = [(h(start, int(carrying)), 0, 0, s0)]
        counter = 1
        closed = set()
        while heap:
            f, negg, _, s = heapq.heappop(heap)
            if s in closed:
                continue
            closed.add(s)
            cell, b, carry = s
            gs = g[s]
            if carry and cell == deliv:
                # goal: perform deliver action
                path = [(DELIVER, s)]
                cur = s
                while cur in parent:
                    cur, act = parent[cur][0], parent[cur][1]
                    path.append((act, cur))
                path.reverse()
                # path entries hold (action, state_BEFORE_action); convert to state AFTER
                out = []
                for k, (act, before) in enumerate(path):
                    after = path[k + 1][1] if k + 1 < len(path) else None
                    out.append((act, after))
                return out
            succ = []
            if battery_enabled:
                if b >= 1:
                    nb = b - 1
                    for a in range(4):
                        t = nbr[cell][a]
                        if t < 0 or (nb == 0 and not is_chg[t]):
                            continue
                        succ.append((a, (t, nb, carry)))
                if is_chg[cell] and b < capacity:
                    succ.append((RECHARGE, (cell, min(capacity, b + recharge), carry)))
            else:
                for a in range(4):
                    t = nbr[cell][a]
                    if t >= 0:
                        succ.append((a, (t, 0, carry)))
            if not carry and cell == pick:
                succ.append((PICKUP, (cell, b, 1)))
            for a, ns in succ:
                ng = gs + 1
                if ns in closed or ng >= g.get(ns, 1 << 30):
                    continue
                g[ns] = ng
                parent[ns] = (s, a)
                heapq.heappush(heap, (ng + h(ns[0], ns[2]), -ng, counter, ns))
                counter += 1
        return None

    # ------------------------------------------------------------------ acting
    def _state(self, e) -> tuple:
        W = e.warehouse_map.width
        r, c = e.robot_pos
        batt = e.battery if e.config.battery_enabled else 0
        return (r * W + c, batt, int(e.carrying))

    def act(self, obs: np.ndarray, env: Optional[Any] = None, deterministic: bool = True) -> int:
        if env is None:
            raise ValueError("AStarAgent is privileged and needs env")
        e = env.unwrapped
        cur = self._state(e)
        if self._expect != cur or self._i >= len(self._plan):
            self._replan(e, cur)
        if not self._plan:
            return self._fallback(e)
        act, after = self._plan[self._i]
        self._i += 1
        self._expect = after
        return int(act)

    def _replan(self, e, cur) -> None:
        m = e.warehouse_map
        W = m.width
        cfg = e.config
        pr, pc = e.pickup_pos
        dr, dc = e.delivery_pos
        plan = self.plan(m, cur[0], pr * W + pc, dr * W + dc, bool(cur[2]), cur[1], cfg.battery_capacity,
                         cfg.recharge_amount, cfg.battery_enabled)
        self.replans += 1
        self.last_plan_failed = plan is None
        self._plan = plan or []
        self._i = 0
        self._expect = cur

    @staticmethod
    def _fallback(e) -> int:
        """No feasible plan: pick a non-colliding move if any, else a no-cost invalid action."""
        mask = e.action_masks()
        for a in range(4):
            if mask[a]:
                return a
        return PICKUP
