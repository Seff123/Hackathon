import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from floodsim.config import (
    Bounds, Config, ScenarioBounds, ScenarioSpec, TeamInputs, load_config, sample_raw_spec,
)
from floodsim.rng import stream

ROOT = Path(__file__).resolve().parents[1]


def test_bounds_sampling():
    rng = np.random.default_rng(0)
    for dist in ("uniform", "loguniform", "normal_clipped"):
        b = Bounds(2.0, 5.0, dist)
        xs = [b.sample(rng) for _ in range(500)]
        assert min(xs) >= 2.0 and max(xs) <= 5.0
    b = Bounds(3, 25, "int")
    xs = {b.sample(rng) for _ in range(2000)}
    assert xs == set(range(3, 26))


def test_raw_spec_roundtrip_and_hash():
    s = sample_raw_spec(7, ScenarioBounds())
    d = json.loads(s.to_json())
    s2 = ScenarioSpec.from_dict(d)
    assert s2 == s and s2.spec_hash == s.spec_hash
    assert sample_raw_spec(7, ScenarioBounds()).spec_hash == s.spec_hash
    assert sample_raw_spec(8, ScenarioBounds()).spec_hash != s.spec_hash


def test_streams_independent():
    a = stream(1, "terrain").random(5)
    b = stream(1, "terrain").random(5)
    c = stream(1, "weather").random(5)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_team_validation():
    with pytest.raises(ValueError):
        TeamInputs(mass_kg=30).validate()
    with pytest.raises(ValueError):
        TeamInputs(battery_life_min=5).validate()
    assert TeamInputs(camera_range_m=250).ceiling_agl_m == 500


def test_battery_from_life():
    cfg = Config()
    gp = cfg.glider
    p_ref = float(gp.drag(gp.v_ref)) * gp.v_ref / gp.prop_eff + gp.payload_w
    assert gp.battery_capacity_wh == pytest.approx(cfg.team.battery_life_min / 60 * p_ref)
    assert 85 < gp.battery_capacity_wh < 100  # ~93 Wh for the 1.5 kg / 1 m team airframe


def test_load_default_toml():
    cfg = load_config(ROOT / "configs" / "default.toml", team={"mass_kg": 2.0})
    assert cfg.team.mass_kg == 2.0 and cfg.glider.mass_kg == 2.0
    assert cfg.glider.span_m == 1.0 and cfg.team.signal_radius_km == 8.0
    cfg = load_config(None, bounds={"n_drones": 4})
    assert cfg.bounds.n_drones.lo == cfg.bounds.n_drones.hi == 4


def test_cli_help():
    r = subprocess.run([sys.executable, "-m", "floodsim", "--help"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0 and "evaluate" in r.stdout
