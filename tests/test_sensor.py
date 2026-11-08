import math
from types import SimpleNamespace

import numpy as np
import pytest

from floodsim.config import SensorParams
from floodsim.people import IN_WATER, OUTDOOR, PeopleState
from floodsim.sensor import Sensor, pd_range
from floodsim.terrain import OPEN


def _flat_terrain(n=200, z=0.0):
    return SimpleNamespace(cell_size=50.0, n=n, z=np.full((n, n), z, np.float32), landcover=np.full((n, n), OPEN, np.int8))


def _one_person(x, y, state=OUTDOOR):
    from floodsim.config import PeopleParams
    p = PeopleState.create({"x": np.array([x]), "y": np.array([y]), "cat": np.array([2]), "mobile": np.array([False])},
                           SimpleNamespace(walk_speed_ms=1.0, drift_fraction=0.5), PeopleParams())
    p.state[:] = state
    return p


def test_pd_table():
    R = 140.0
    for z, land, water in [(70, 0.90, 0.62), (105, 0.70, 0.31), (140, 0.48, 0.16), (210, 0.21, 0.06)]:
        assert pd_range(z, 0.5, R) == pytest.approx(land, abs=0.02)
        assert pd_range(z, 0.3, R) == pytest.approx(water, abs=0.02)


@pytest.mark.parametrize("R", [70.0, 140.0, 200.0])
def test_camera_range(R):
    assert float(pd_range(R, 0.5, R)) == pytest.approx(0.475, abs=0.01)
    ter = _flat_terrain()
    s = Sensor(SensorParams(), ter, R, 0.0, np.random.default_rng(0))
    p = _one_person(5025.0, 5025.0)
    hits = 0
    N = 4000
    for _ in range(N):
        fr, truth = s.frame(0, 5025.0, 5025.0, R, 0.0, p)
        hits += len(fr.detections) > 0
    assert hits / N == pytest.approx(0.475, abs=0.03)


def test_empirical_detection_rate():
    ter = _flat_terrain()
    s = Sensor(SensorParams(), ter, 140.0, 0.0, np.random.default_rng(1))
    p = _one_person(5030.0, 5010.0, IN_WATER)
    pd = float(pd_range(100.0, 0.3, 140.0) * 0.8)
    hits = sum(len(s.frame(0, 5000.0, 5000.0, 100.0, 0.0, p)[0].detections) > 0 for _ in range(10000))
    assert hits / 10000 == pytest.approx(pd, abs=0.02)


def test_small_footprint_and_area():
    ter = _flat_terrain()
    s = Sensor(SensorParams(), ter, 140.0, 0.0, np.random.default_rng(2))
    rng = np.random.default_rng(3)
    p = _one_person(10.0, 10.0)
    p.found[:] = True
    for _ in range(200):
        x, y = rng.uniform(1000, 9000, 2)
        fr, _ = s.frame(0, x, y, 60.0, 0.0, p)
        assert int(y // 50) * 200 + int(x // 50) in set(fr.cell_idx.tolist())
    for h in (100.0, 140.0, 250.0):
        areas = []
        for _ in range(1000):
            x, y = rng.uniform(1000, 9000, 2)
            fr, _ = s.frame(0, x, y, h, 0.0, p)
            areas.append(len(fr.cell_idx) * 2500.0)
        fp = math.pi * (h * math.tan(math.radians(30))) ** 2
        assert np.mean(areas) == pytest.approx(fp, rel=0.10)


def test_detections_in_observed_cells_and_frames_skip_high():
    ter = _flat_terrain()
    s = Sensor(SensorParams(), ter, 140.0, 0.05, np.random.default_rng(4))
    p = _one_person(5040.0, 5040.0)
    for _ in range(300):
        fr, _ = s.frame(0, 5000.0, 5000.0, 120.0, 0.0, p)
        assert set(fr.detections) <= set(fr.cell_idx.tolist())
        assert not hasattr(fr, "pd")
    fr, _ = s.frame(0, 5000.0, 5000.0, 2 * 140.0 + 1, 0.0, p)
    assert fr is None
