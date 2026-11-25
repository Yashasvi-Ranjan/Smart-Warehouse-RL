"""WarehouseEnv: Gymnasium environment implementing docs/INTERFACES.md.

Extensions beyond the contract (all optional / additive):
* ``reset(options=...)`` also accepts ``start_pos``, ``pickup_pos``, ``delivery_pos`` (``(r, c)``) and
  ``battery`` (int) to force parts of the task (used by tests/UI; forced tasks are not checked for
  solvability).  ``layout_seed`` is the contract option.
* ``info["action_event"]`` is the event of the action itself; on a truncating step ``info["event"]`` is
  ``"timeout"`` (as listed in the contract) while ``action_event`` keeps the real one.
* ``info["pickup_pos"]`` / ``info["delivery_pos"]`` are not provided (use ``env.unwrapped`` helpers).

Notes / documented choices:
* Start cell is a random floor cell that is not a pickup/delivery/charger cell.
* Depletion: with battery enabled, a move that leaves battery 0 on a non-charger cell terminates
  (reward = step + depleted, event ``depleted``, the move HAS happened); a move attempted at battery 0
  terminates with ``depleted`` without moving (checked before collision).
* ``action_mask`` of a terminal state is computed ignoring the battery rule for moves so that it is never
  all-False (avoids NaNs when agents mask next-state values).
* Stepping after the episode ended (or before ``reset``) raises ``RuntimeError``.
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvConfig
from .warehouse_map import UNREACHABLE, WarehouseMap, get_layout

UP, DOWN, LEFT, RIGHT, PICKUP, DELIVER, RECHARGE = range(7)
ACTION_NAMES = ("up", "down", "left", "right", "pickup", "deliver", "recharge")


def obs_fields(view_radius: int) -> dict:
    n_win = (2 * view_radius + 1) ** 2 - 1
    return {
        "robot": slice(0, 2), "pickup": slice(2, 4), "delivery": slice(4, 6), "carrying": slice(6, 7),
        "battery": slice(7, 8), "nearest_charger": slice(8, 10), "blocked_udlr": slice(10, 14),
        "can_pickup": slice(14, 15), "can_deliver": slice(15, 16), "on_charger": slice(16, 17),
        "goal_delta": slice(17, 19), "local_window": slice(19, 19 + n_win),
    }


class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(self, config: "EnvConfig | dict | None" = None, render_mode: Optional[str] = None,
                 cell_size: int = 48):
        super().__init__()
        if config is None:
            config = EnvConfig()
        elif isinstance(config, dict):
            config = EnvConfig.from_dict(config)
        elif not isinstance(config, EnvConfig):
            raise TypeError("config must be EnvConfig, dict or None")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")
        self.config = config
        self.render_mode = render_mode
        self.cell_size = int(cell_size)
        r = config.view_radius
        self.n_obs = 19 + (2 * r + 1) ** 2 - 1
        self.OBS_FIELDS = obs_fields(r)
        self.observation_space = spaces.Box(-1.0, 1.0, (self.n_obs,), np.float32)
        self.action_space = spaces.Discrete(7)

        self._H, self._W = config.grid_h, config.grid_w
        self._H1, self._W1 = config.grid_h - 1, config.grid_w - 1
        self._batt_en = config.battery_enabled
        self._cap = config.battery_capacity
        self._recharge = config.recharge_amount
        self._rcfg = config.reward
        self._rt = self._rcfg.base_table()
        self._bonus = self._rcfg.battery_bonus
        self._shaping = self._rcfg.shaping
        self._gamma = self._rcfg.gamma

        self._map: Optional[WarehouseMap] = None
        self._done = True
        self._needs_reset = True
        self._pos = 0
        self._carry = False
        self._batt = self._cap
        self._steps = 0
        self._window = None
        self._clock = None
        self._pg_screen = None
        self._last_event = "reset"

    # ================================================================== public helpers
    @property
    def warehouse_map(self) -> WarehouseMap:
        return self._map

    @property
    def robot_pos(self) -> tuple:
        return divmod(self._pos, self._W)

    @property
    def pickup_pos(self) -> tuple:
        return divmod(self._pick, self._W)

    @property
    def delivery_pos(self) -> tuple:
        return divmod(self._deliv, self._W)

    @property
    def carrying(self) -> bool:
        return self._carry

    @property
    def battery(self) -> int:
        return self._batt

    @property
    def chargers(self) -> list:
        return list(self._map.chargers)

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def layout_seed(self) -> int:
        return self._map.layout_seed

    @property
    def optimal_steps(self) -> int:
        return self._optimal

    def potential(self) -> float:
        """Shaping potential Phi(s) of the current state (uses true BFS distances)."""
        d = self._d_deliv if self._carry else self._d_pick
        return -(d[self._pos] + (0 if self._carry else self._leg))

    def action_masks(self) -> np.ndarray:
        return self._mask(self._done)

    def discrete_state(self, B: int = 6) -> tuple:
        W = self._W
        pos = self._pos
        r, c = divmod(pos, W)
        pr, pc = divmod(self._pick, W)
        dr, dc = divmod(self._deliv, W)
        if self._batt_en:
            bucket = min(B - 1, int(self._batt / self._cap * B))
        else:
            bucket = 0
        b = self._blk4[pos]
        return (r, c, pr, pc, dr, dc, int(self._carry), bucket,
                int(b[0]), int(b[1]), int(b[2]), int(b[3]))

    def get_render_state(self) -> dict:
        m = self._map
        return {
            "grid": m.grid, "height": m.height, "width": m.width,
            "robot_pos": self.robot_pos,
            "pickup_stations": list(m.pickup_stations), "delivery_stations": list(m.delivery_stations),
            "chargers": list(m.chargers),
            "active_pickup": self.pickup_pos, "active_delivery": self.delivery_pos,
            "carrying": self._carry, "battery": self._batt, "battery_capacity": self._cap,
            "battery_enabled": self._batt_en, "steps": self._steps, "max_steps": self.config.max_steps,
            "layout_seed": m.layout_seed, "event": self._last_event, "total_reward": self._ep_reward,
        }

    # ================================================================== reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}
        cfg = self.config
        rng = self.np_random
        forced = options.get("layout_seed")
        if cfg.layout_seeds is not None:
            ls = int(cfg.layout_seeds[int(rng.integers(len(cfg.layout_seeds)))])
        else:
            ls = int(rng.integers(0, 2 ** 31 - 1))
        if forced is not None:
            ls = int(forced)
        explicit = any(k in options for k in ("start_pos", "pickup_pos", "delivery_pos", "battery"))
        task = None
        for _ in range(30):
            m = get_layout(cfg, ls)
            task = self._sample_task(m, rng, options, explicit)
            if task is not None:
                break
            if forced is not None or cfg.layout_seeds is not None:
                break
            ls = int(rng.integers(0, 2 ** 31 - 1))
        if task is None:
            raise RuntimeError(f"No solvable task found on layout {ls}; check battery/max_steps settings")
        start, pick, deliv, batt = task
        W = self._W
        self._map = m
        self._nbr = m.nbr
        self._is_chg = m.is_charger
        t = m.tables(cfg.view_radius)
        self._win, self._blk4, self._open4, self._near = t["win"], t["blk4"], t["open4"], t["near_charger"]
        self._rn, self._cn = t["row_norm"], t["col_norm"]
        self._pos = start[0] * W + start[1]
        self._pick = pick[0] * W + pick[1]
        self._deliv = deliv[0] * W + deliv[1]
        self._d_pick = m.dist_map(pick)
        self._d_deliv = m.dist_map(deliv)
        self._leg = self._d_pick[self._deliv]
        self._optimal = self._d_pick[self._pos] + 1 + self._leg + 1
        self._carry = False
        self._batt = batt
        self._steps = 0
        self._done = False
        self._needs_reset = False
        self._ep_reward = 0.0
        self._collisions = self._invalid = self._energy = self._recharges = 0
        self._picked = False
        self._depleted = False
        self._timeout = False
        self._success = False
        self._last_event = "reset"
        base = np.zeros(self.n_obs, dtype=np.float32)
        base[2] = self._rn[pick[0]]
        base[3] = self._cn[pick[1]]
        base[4] = self._rn[deliv[0]]
        base[5] = self._cn[deliv[1]]
        self._base = base
        self._phi = self.potential() if self._shaping else 0.0
        obs, mask = self._obs_mask(False)
        info = self._info(mask, "reset")
        if self.render_mode == "human":
            self.render()
        return obs, info

    def _sample_task(self, m: WarehouseMap, rng, options: dict, explicit: bool):
        cfg = self.config
        lo, hi = cfg.battery_init_range
        for _ in range(100):
            pick = m.pickup_stations[int(rng.integers(len(m.pickup_stations)))]
            deliv = m.delivery_stations[int(rng.integers(len(m.delivery_stations)))]
            start = m.start_cells[int(rng.integers(len(m.start_cells)))]
            batt = int(round(float(rng.uniform(lo, hi)) * cfg.battery_capacity)) if self._batt_en else cfg.battery_capacity
            if explicit:
                start = tuple(options.get("start_pos", start))
                pick = tuple(options.get("pickup_pos", pick))
                deliv = tuple(options.get("delivery_pos", deliv))
                batt = int(options.get("battery", batt))
                for p in (start, pick, deliv):
                    if m.is_blocked(*p):
                        raise ValueError(f"forced cell {p} is blocked")
                if not 0 <= batt <= cfg.battery_capacity:
                    raise ValueError("forced battery out of range")
                return start, pick, deliv, batt
            if self._task_ok(m, start, pick, deliv, batt):
                return start, pick, deliv, batt
        return None

    def _task_ok(self, m, start, pick, deliv, batt) -> bool:
        cfg = self.config
        d1 = m.dist_map(pick)[m.idx(start)]
        d2 = m.dist_map(deliv)[m.idx(pick)]
        if d1 >= UNREACHABLE or d2 >= UNREACHABLE:
            return False
        opt = d1 + 1 + d2 + 1
        if not self._batt_en or batt > d1 + d2:
            return opt <= cfg.max_steps  # fast path: enough battery without recharging
        s = m.min_task_steps(start, pick, deliv, batt, cfg.battery_capacity, cfg.recharge_amount, True)
        return s is not None and s <= cfg.max_steps

    # ================================================================== step
    def step(self, action):
        if self._done:
            raise RuntimeError("step() called on a finished episode (or before reset()); call reset() first")
        a = int(action)
        if not 0 <= a < 7:
            raise ValueError(f"invalid action {action!r}; expected 0..6")
        self._steps += 1
        terminated = False
        batt_en = self._batt_en
        pos = self._pos
        if a < 4:
            if batt_en and self._batt <= 0:
                event = "depleted"
                terminated = True
                self._depleted = True
            else:
                t = self._nbr[pos][a]
                if t < 0:
                    event = "collision"
                    self._collisions += 1
                else:
                    self._pos = t
                    self._energy += 1
                    event = "move"
                    if batt_en:
                        self._batt -= 1
                        if self._batt <= 0 and not self._is_chg[t]:
                            event = "depleted"
                            terminated = True
                            self._depleted = True
        elif a == 4:
            if pos == self._pick and not self._carry:
                self._carry = True
                self._picked = True
                event = "pickup"
            else:
                event = "invalid_pickup"
                self._invalid += 1
        elif a == 5:
            if pos == self._deliv and self._carry:
                event = "deliver"
                terminated = True
                self._success = True
            else:
                event = "invalid_deliver"
                self._invalid += 1
        else:
            if batt_en and self._is_chg[pos] and self._batt < self._cap:
                self._batt = min(self._cap, self._batt + self._recharge)
                self._recharges += 1
                event = "recharge"
            else:
                event = "invalid_recharge"
                self._invalid += 1

        reward = self._rt[event]
        if event == "deliver" and batt_en and self._batt > 0:
            reward += self._bonus
        if self._shaping:
            phi_new = 0.0 if terminated else self.potential()
            reward += self._gamma * phi_new - self._phi
            self._phi = phi_new
        self._ep_reward += reward

        truncated = (not terminated) and self._steps >= self.config.max_steps
        info_event = event
        if truncated:
            self._timeout = True
            info_event = "timeout"
        done = terminated or truncated
        self._done = done
        self._last_event = info_event
        obs, mask = self._obs_mask(done)
        info = self._info(mask, info_event)
        info["action_event"] = event
        if done:
            info["episode_stats"] = self._episode_stats()
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    # ================================================================== obs / info
    def _mask(self, final: bool = False) -> np.ndarray:
        pos = self._pos
        m = np.zeros(7, dtype=np.bool_)
        if final or not self._batt_en or self._batt >= 1:
            m[:4] = self._open4[pos]
        m[4] = pos == self._pick and not self._carry
        m[5] = pos == self._deliv and self._carry
        m[6] = self._batt_en and self._is_chg[pos] and self._batt < self._cap
        return m

    def _obs_mask(self, final: bool):
        pos = self._pos
        mask = self._mask(final)
        o = self._base.copy()
        r, c = divmod(pos, self._W)
        rn, cn = self._rn[r], self._cn[c]
        o[0] = rn
        o[1] = cn
        o[6] = self._carry
        o[7] = self._batt / self._cap if self._batt_en else 1.0
        o[8:10] = self._near[pos]
        o[10:14] = self._blk4[pos]
        o[14] = mask[4]
        o[15] = mask[5]
        o[16] = self._is_chg[pos]
        if self._carry:
            o[17] = o[4] - rn
            o[18] = o[5] - cn
        else:
            o[17] = o[2] - rn
            o[18] = o[3] - cn
        o[19:] = self._win[pos]
        return o, mask

    def _info(self, mask, event: str) -> dict:
        return {"action_mask": mask, "event": event, "layout_seed": self._map.layout_seed,
                "robot_pos": divmod(self._pos, self._W), "carrying": self._carry,
                "battery": self._batt, "steps": self._steps}

    def _episode_stats(self) -> dict:
        return {
            "reward": float(self._ep_reward), "steps": self._steps, "success": self._success,
            "picked_up": self._picked, "collisions": self._collisions, "invalid_actions": self._invalid,
            "energy_used": self._energy, "recharges": self._recharges,
            "battery_depleted": self._depleted, "timeout": self._timeout,
            "optimal_steps": int(self._optimal), "layout_seed": self._map.layout_seed,
            "battery_enabled": self._batt_en,
        }

    # ================================================================== render
    def render(self):
        if self.render_mode is None:
            return None
        if self._map is None:
            raise RuntimeError("render() called before reset()")
        import pygame
        from .renderer import draw_warehouse
        cs = self.cell_size
        size = (self._W * cs, self._H * cs)
        if self.render_mode == "rgb_array":
            surf = pygame.Surface(size)
            draw_warehouse(surf, self.get_render_state(), cs)
            return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2)).copy()
        # human
        if self._pg_screen is None:
            pygame.init()
            pygame.display.init()
            self._pg_screen = pygame.display.set_mode(size)
            pygame.display.set_caption("Smart Warehouse")
            self._clock = pygame.time.Clock()
        draw_warehouse(self._pg_screen, self.get_render_state(), cs)
        pygame.event.pump()
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
        return None

    def close(self):
        if self._pg_screen is not None:
            import pygame
            pygame.display.quit()
            self._pg_screen = None
