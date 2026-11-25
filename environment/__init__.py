"""Smart Warehouse environment package."""
import gymnasium as _gym

from .config import EnvConfig, load_config
from .rewards import REWARD_PRESETS, RewardConfig
from .warehouse_env import WarehouseEnv
from .warehouse_map import WarehouseMap, generate_layout

if "SmartWarehouse-v0" not in _gym.registry:
    _gym.register(id="SmartWarehouse-v0", entry_point="environment.warehouse_env:WarehouseEnv")

__all__ = ["WarehouseEnv", "EnvConfig", "RewardConfig", "REWARD_PRESETS", "WarehouseMap",
           "generate_layout", "load_config"]
