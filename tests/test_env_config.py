import json

import pytest

from environment import EnvConfig, REWARD_PRESETS, RewardConfig
from environment.config import load_config
from conftest import ROOT


def test_defaults_match_yaml():
    y = load_config(str(ROOT / "config.yaml"))
    cfg = EnvConfig.from_dict({**y["env"], "reward": y["reward"]})
    assert cfg.to_dict() == EnvConfig().to_dict()
    assert cfg.reward == REWARD_PRESETS["default"]


def test_roundtrip_json_safe():
    cfg = EnvConfig(layout_seeds=[1, 2], reward="shaped", battery_init_range=(0.4, 0.9))
    d = json.loads(json.dumps(cfg.to_dict()))
    assert EnvConfig.from_dict(d).to_dict() == cfg.to_dict()
    assert EnvConfig.from_dict(d).reward.shaping is True


def test_unknown_keys_raise():
    with pytest.raises(ValueError):
        EnvConfig.from_dict({"grid_h": 10, "bogus": 1})
    with pytest.raises(ValueError):
        RewardConfig.from_dict({"deliver": 1, "bogus": 2})
    with pytest.raises(ValueError):
        EnvConfig(reward="nope")


@pytest.mark.parametrize("bad", [
    {"grid_h": 4}, {"grid_w": 5}, {"battery_capacity": 10}, {"battery_init_range": (0.9, 0.5)},
    {"n_delivery_stations": 20}, {"max_steps": 0}, {"layout_seeds": []}, {"view_radius": 0},
    {"n_pickup_stations": 0}, {"recharge_amount": 0}])
def test_validation(bad):
    with pytest.raises(ValueError):
        EnvConfig(**bad)


def test_reward_specs():
    assert EnvConfig(reward="sparse").reward == REWARD_PRESETS["sparse"]
    assert EnvConfig(reward={"preset": "shaped", "gamma": 0.9}).reward.gamma == 0.9
    assert EnvConfig(reward=REWARD_PRESETS["high_move_penalty"]).reward.step == -1.0
    assert RewardConfig.from_any({"step": -0.5}).step == -0.5
    assert set(REWARD_PRESETS) == {"default", "sparse", "shaped", "high_move_penalty",
                                   "low_collision_penalty", "no_pickup_bonus"}
