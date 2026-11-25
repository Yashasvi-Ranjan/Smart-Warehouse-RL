"""Reward configuration and per-step reward computation.

All reward values are *signed rewards* (so the step penalty is ``-0.2``, not ``0.2``).

Per-step reward = ``event_reward`` (+ battery bonus on a delivery with battery > 0)
(+ potential-based shaping term if ``shaping`` is on).  The step penalty is applied to
EVERY action.

Shaping (preset ``shaped``): F = gamma * Phi(s') - Phi(s) with
``Phi(s) = -(d(robot, goal) + (d(pickup, delivery) if not carrying else 0))`` where ``d`` is the
true BFS shortest-path distance, goal = active pickup cell if not carrying else the active
delivery cell.  Phi of a *terminal* state is defined as 0 (it is 0 anyway on delivery; on battery
depletion we set it to 0 explicitly) which keeps the shaping policy-invariant.  Truncation is not
terminal for shaping purposes.  Note with gamma < 1 a non-moving step yields (gamma-1)*Phi >= 0,
a tiny constant-order bonus that is part of standard PBRS (it cancels in the optimal policy).
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

EVENTS = ("move", "collision", "pickup", "invalid_pickup", "deliver", "invalid_deliver",
          "recharge", "invalid_recharge", "depleted", "timeout", "reset")


@dataclass
class RewardConfig:
    deliver: float = 100.0
    pickup: float = 20.0
    battery_bonus: float = 10.0   # extra on delivery when battery enabled and battery > 0
    step: float = -0.2            # applied to every action
    collision: float = -5.0
    invalid: float = -2.0         # invalid pickup / deliver / recharge
    depleted: float = -50.0
    shaping: bool = False
    gamma: float = 0.99           # discount used by the shaping term

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "shaping":
                self.shaping = bool(v)
            else:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(f"RewardConfig.{f.name} must be a number, got {v!r}")
                setattr(self, f.name, float(v))
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("RewardConfig.gamma must be in [0, 1]")

    # ---- construction -------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "RewardConfig":
        """``{"preset": name, **overrides}`` or plain field dict (base = ``default``). Unknown keys raise."""
        d = dict(d)
        base_name = d.pop("preset", "default")
        base = get_preset(base_name)
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"Unknown RewardConfig keys: {sorted(unknown)}")
        return replace(base, **d)

    @classmethod
    def from_any(cls, spec: Any) -> "RewardConfig":
        if spec is None:
            return get_preset("default")
        if isinstance(spec, RewardConfig):
            return replace(spec)
        if isinstance(spec, str):
            return get_preset(spec)
        if isinstance(spec, dict):
            return cls.from_dict(spec)
        raise ValueError(f"Cannot build RewardConfig from {type(spec).__name__}")

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def preset_name(self) -> str:
        for k, v in REWARD_PRESETS.items():
            if v == self:
                return k
        return "custom"

    # ---- reward tables -----------------------------------------------------------------
    def base_table(self) -> dict[str, float]:
        """Reward (excluding battery bonus and shaping) for each event, step penalty included."""
        s = self.step
        return {
            "move": s, "collision": s + self.collision,
            "pickup": s + self.pickup, "invalid_pickup": s + self.invalid,
            "deliver": s + self.deliver, "invalid_deliver": s + self.invalid,
            "recharge": s, "invalid_recharge": s + self.invalid,
            "depleted": s + self.depleted, "timeout": s, "reset": 0.0,
        }


def get_preset(name: str) -> RewardConfig:
    if name not in REWARD_PRESETS:
        raise ValueError(f"Unknown reward preset {name!r}; choose from {sorted(REWARD_PRESETS)}")
    return replace(REWARD_PRESETS[name])


_D = RewardConfig()
REWARD_PRESETS: dict[str, RewardConfig] = {
    "default": _D,
    "sparse": RewardConfig(deliver=100.0, pickup=0.0, battery_bonus=0.0, step=-0.2,
                           collision=0.0, invalid=0.0, depleted=-50.0),
    "shaped": RewardConfig(shaping=True),
    "high_move_penalty": RewardConfig(step=-1.0),
    "low_collision_penalty": RewardConfig(collision=-0.5),
    "no_pickup_bonus": RewardConfig(pickup=0.0),
}


def event_reward(cfg: RewardConfig, event: str, battery_enabled: bool = True,
                 battery_after: int = 1) -> float:
    """Reference (slow) reward for one action, WITHOUT shaping. The env uses an equivalent fast path."""
    r = cfg.base_table()[event]
    if event == "deliver" and battery_enabled and battery_after > 0:
        r += cfg.battery_bonus
    return r


def potential(dist_to_goal: float, leg: float, carrying: bool) -> float:
    """Phi(s) = -(d(robot, goal) + (d(pickup, delivery) if not carrying))."""
    return -(dist_to_goal + (0.0 if carrying else leg))


def shaping_term(gamma: float, phi_old: float, phi_new: float, terminal: bool = False) -> float:
    return gamma * (0.0 if terminal else phi_new) - phi_old
