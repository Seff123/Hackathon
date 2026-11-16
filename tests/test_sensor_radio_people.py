import math

import numpy as np
import pytest

from floodsim.config import Config, SensorParams
from floodsim.people import (HOME_INSIDE, IN_WATER, OUTDOOR_S, ROOF, SAFE_HIGH, People, Placement,
                             place_people)
from floodsim.radio import BASE_ID, Radio
from floodsim.scenario import sample_spec, terrain_for
from floodsim.sensor import Sensor, p_detect

CFG = Config()


@pytest.fixture(scope="module")
def terr_spec():
    spec = sample_spec(1, cfg=CFG)
    terr, _ = terrain_for(spec, CFG)
    return terr, spec


def _one_person(terr, spec, x, y, cat=2, mobile=False):
    pl = Placement(x=np.array([x]), y=np.array([y]), category=np.array([cat], np.int8),
                   mobile=np.array([mobile]))
    return People(pl, spec, terr, CFG.people)


def _open_cell(terr):
    from floodsim.terrain import OPEN

    iy, ix = np.argwhere((terr.landcover == OPEN) & ~terr.road_mask)[len(terr.z) * 3]
    return (ix + 0.5) * terr.cell_size, (iy + 0.5) * terr.cell_size


@pytest.mark.parametrize("R", [70.0, 140.0, 250.0])
def test_camera_range_gives_half(R):
    assert float(p_detect(R, R)) == pytest.approx(0.475, abs=0.01)


def test_detection_table_values():
    R = 140.0
    for frac, land, water in ((0.5, 0.90, 0.62), (0.75, 0.70, 0.31), (1.0, 0.48, 0.16), (1.5, 0.21, 0.06)):
        assert float(p_detect(frac * R, R, 0.5)) == pytest.approx(land, abs=0.02)
        assert float(p_detect(frac * R, R, 0.3)) == pytest.approx(water, abs=0.02)


def test_empirical_detection_rate(terr_spec):
    terr, spec = terr_spec
    x, y = _open_cell(terr)
    ppl = _one_person(terr, spec, x, y)
    s = Sensor(CFG.sensor, terr, CFG.team.camera_range_m, 1, np.random.default_rng(0))
    gz = float(terr.z[int(y // 50), int(x // 50)])
    z = gz + 120.0
    hits = 0
    for _ in range(10000):
        _, truth = s.frame(0.0, 0, x, y, z, ppl, gz)
        hits += len(truth)
    p = float(p_detect(z - ppl.elevation()[0], CFG.team.camera_range_m, 0.5, s.cover.ravel()[ppl.cells()[0]], 1.0))
    assert hits / 10000 == pytest.approx(p, abs=0.02)


def test_small_footprint_and_area(terr_spec):
    terr, _ = terr_spec
    s = Sensor(CFG.sensor, terr, 140.0, 1, np.random.default_rng(1))
    rng = np.random.default_rng(2)
    for _ in range(200):
        x, y = rng.uniform(500, 9500, 2)
        cells, _ = s.footprint_cells(x, y, 60.0)
        assert int(y // 50) * terr.n + int(x // 50) in cells
    for h in (100.0, 160.0, 250.0):
        areas = []
        for _ in range(1000):
            x, y = rng.uniform(500, 9500, 2)
            cells, r = s.footprint_cells(x, y, h)
            areas.append(len(cells) * 2500.0)
        assert np.mean(areas) == pytest.approx(math.pi * (h * s.tan) ** 2, rel=0.10)


def test_detections_in_observed_cells(terr_spec):
    terr, spec = terr_spec
    sp = SensorParams(false_alarm_rate=0.05)
    s = Sensor(sp, terr, 140.0, 1, np.random.default_rng(3))
    x, y = _open_cell(terr)
    ppl = _one_person(terr, spec, x + 40, y + 30)
    gz = float(terr.z[int(y // 50), int(x // 50)])
    n_det = 0
    for _ in range(300):
        f, _ = s.frame(0.0, 0, x, y, gz + 90, ppl, gz)
        assert set(f.detections) <= set(f.cell_idx.tolist())
        n_det += len(f.detections)
    assert n_det > 0


def test_radio_range():
    r = Radio(3000.0)
    nodes = {0: (0.0, 0.0), 1: (0.99 * 3000.0, 0.0), 2: (1.01 * 3000.0, 0.0), BASE_ID: (100.0, 0.0)}
    r.send(0, "hello", 0.0)
    inbox = r.deliver(nodes)
    assert [m.payload for m in inbox[1]] == ["hello"]
    assert inbox[2] == []
    assert [m.payload for m in inbox[BASE_ID]] == ["hello"]
    assert inbox[0] == []
    r.send(2, "far", 1.0, to=0)
    assert r.deliver(nodes)[0] == []


class FakeFlood:
    def __init__(self, depth_fn, speed=0.0):
        self.f = depth_fn
        self.speed = speed

    def depth(self, x, y, t):
        return np.full(np.shape(x), self.f(t))

    def velocity(self, x, y, t):
        return np.full(np.shape(x), self.speed), np.zeros(np.shape(x))

    def depth_grid(self, t):
        return np.full((200, 200), self.f(t), np.float32)


def test_people_roof_and_danger(terr_spec):
    terr, spec = terr_spec
    x, y = _open_cell(terr)
    p = _one_person(terr, spec, x, y, cat=0)
    fl = FakeFlood(lambda t: 0.0 if t < 600 else 1.5, speed=1.0)  # HR = 1.5*1.5+1 = 3.25
    for t in np.arange(0, 3600, 10.0):
        p.step(t, fl, 10.0)
    assert p.state[0] == ROOF
    assert p.t_danger[0] == pytest.approx(3000.0, abs=20)
    assert p.survival()[0] == pytest.approx(math.exp(-3000 / 2400), abs=0.01)
    assert p.t_affected[0] == pytest.approx(600.0)


def test_people_walk_to_high_ground(terr_spec):
    terr, spec = terr_spec
    iy, ix = np.argwhere(terr.land & (terr.hand < 1.0) & (terr.dist_to_river < 300))[0]
    x, y = (ix + 0.5) * 50, (iy + 0.5) * 50
    p = _one_person(terr, spec, x, y, cat=2, mobile=True)
    fl = FakeFlood(lambda t: 0.2)
    for t in np.arange(0, 7200, 10.0):
        p.step(t, fl, 10.0)
    assert p.state[0] == SAFE_HIGH
    p2 = _one_person(terr, spec, x, y, cat=2, mobile=True)
    fl2 = FakeFlood(lambda t: 1.0, speed=0.5)
    p2.step(0.0, fl2, 10.0)
    assert p2.state[0] == IN_WATER
