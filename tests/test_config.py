import numpy as np
import pytest

from floodsim.config import Config, GliderParams, TeamInputs, load_config
from floodsim.rng import Streams


def test_load_default_config():
    cfg = load_config("configs/default.toml")
    assert cfg.glider.mass == cfg.team.mass_kg
    assert cfg.sensor.camera_range_m == cfg.team.camera_range_m
    assert cfg.ceiling_agl == max(400.0, 2 * cfg.team.camera_range_m)


def test_battery_capacity_placeholder():
    g = GliderParams()
    assert 170 < g.capacity_wh < 190  # plan: ~180 Wh for 150 min at 5 kg


def test_team_validation():
    with pytest.raises(ValueError):
        TeamInputs(mass_kg=30).validate()


def test_streams_independent():
    a, b = Streams(3), Streams(3)
    a["terrain"].random(1000)  # extra draws on one stream
    assert a["people"].random() == b["people"].random()
    assert Streams(3)["terrain"].random() != Streams(4)["terrain"].random()
