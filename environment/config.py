"""EnvConfig: environment configuration mirroring ``config.yaml: env`` plus the reward spec."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional

from .rewards import RewardConfig


def load_config(path: str = "config.yaml") -> dict:
    """Load the whole yaml file and return it as a dict."""
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if data is not None else {}


@dataclass
class EnvConfig:
    grid_h: int = 10
    grid_w: int = 10
    max_steps: int = 200
    battery_enabled: bool = True
    battery_capacity: int = 60
    battery_init_range: tuple = (0.5, 1.0)
    recharge_amount: int = 20
    n_chargers: int = 2
    n_pickup_stations: int = 3
    n_delivery_stations: int = 3
    view_radius: int = 2
    layout_seeds: Optional[list] = None
    reward: Any = "default"  # preset name | dict | RewardConfig ; normalised to RewardConfig

    def __post_init__(self) -> None:
        self.reward = RewardConfig.from_any(self.reward)
        r = self.battery_init_range
        if not isinstance(r, (list, tuple)) or len(r) != 2:
            raise ValueError("battery_init_range must be [lo, hi]")
        self.battery_init_range = (float(r[0]), float(r[1]))
        if self.layout_seeds is not None:
            self.layout_seeds = [int(s) for s in self.layout_seeds]
        for name in ("grid_h", "grid_w", "max_steps", "battery_capacity", "recharge_amount",
                     "n_chargers", "n_pickup_stations", "n_delivery_stations", "view_radius"):
            v = getattr(self, name)
            if isinstance(v, bool) or int(v) != v:
                raise ValueError(f"EnvConfig.{name} must be an integer, got {v!r}")
            setattr(self, name, int(v))
        self.battery_enabled = bool(self.battery_enabled)
        self.validate()

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        if self.grid_h < 6 or self.grid_w < 6:
            raise ValueError("grid_h and grid_w must be >= 6 (room for walls, shelves and aisles)")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.view_radius < 1:
            raise ValueError("view_radius must be >= 1")
        if self.n_pickup_stations < 1 or self.n_delivery_stations < 1:
            raise ValueError("need at least one pickup and one delivery station")
        if self.n_delivery_stations > self.grid_w - 2:
            raise ValueError("n_delivery_stations must be <= grid_w - 2 (they sit on the bottom dock row)")
        if self.n_chargers < 0:
            raise ValueError("n_chargers must be >= 0")
        lo, hi = self.battery_init_range
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError("battery_init_range must satisfy 0 <= lo <= hi <= 1")
        if self.battery_enabled:
            if self.battery_capacity < self.grid_h + self.grid_w:
                raise ValueError(
                    "battery_capacity must be >= grid_h + grid_w so that tasks are solvable "
                    f"(got {self.battery_capacity} < {self.grid_h + self.grid_w})")
            if self.n_chargers >= 1 and self.recharge_amount < 1:
                raise ValueError("recharge_amount must be >= 1 when chargers exist")
        if self.layout_seeds is not None:
            if len(self.layout_seeds) == 0:
                raise ValueError("layout_seeds must be null or a non-empty list")
            if any(s < 0 for s in self.layout_seeds):
                raise ValueError("layout seeds must be >= 0")

    # ------------------------------------------------------------------ (de)serialisation
    @classmethod
    def from_dict(cls, d: dict) -> "EnvConfig":
        """Build from a dict (e.g. ``load_config(p)["env"]``, optionally with a ``reward`` entry such as
        ``"shaped"``, ``{"preset": "shaped"}`` or a field dict). Unknown keys raise ``ValueError``."""
        if not isinstance(d, dict):
            raise ValueError("EnvConfig.from_dict expects a dict")
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"Unknown EnvConfig keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        """JSON-safe dict (``from_dict(to_dict())`` round-trips)."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "reward":
                v = v.to_dict()
            elif f.name == "battery_init_range":
                v = [float(x) for x in v]
            elif f.name == "layout_seeds":
                v = None if v is None else [int(x) for x in v]
            out[f.name] = v
        return out

    def layout_key(self) -> tuple:
        """The fields a layout depends on (layout = pure function of this and the layout seed)."""
        return (self.grid_h, self.grid_w, self.n_chargers, self.n_pickup_stations,
                self.n_delivery_stations)
