import hashlib

import numpy as np
import pytest

from floodsim.config import Config, ScenarioBounds, ScenarioSpec, sample_raw_spec, Bounds
from floodsim.hydrology import compute_hydrology, fraction_draining_to, priority_flood
from floodsim.scenario import sample_spec, terrain_for
from floodsim.terrain import WATER

CFG = Config()


@pytest.fixture(scope="module")
def specs():
    return [sample_spec(s, cfg=CFG) for s in range(6)]


def test_sampler_respects_bounds():
    b = ScenarioBounds()
    for seed in range(200):
        s = sample_raw_spec(seed, b)
        for name, val in vars(s).items():
            bnd = getattr(b, name, None)
            if isinstance(bnd, Bounds) and name != "rain_peak_mmh":
                assert bnd.contains(val) or (name == "levee_height_m" and s.flood_type == "breach"), (name, val)
        assert all(b.town_radius_m.contains(r) for r in s.town_radii_m)
        assert len(s.town_radii_m) == s.n_towns
        assert abs(sum(s.category_weights) - 1) < 1e-9
        assert 10 * 1.0 <= s.rain_peak_mmh <= 60 * 1.4 + 1e-9
        assert 0 <= s.w_star_ms <= 2.5 * 0.7 * 1.4 + 1e-9


def _hash(*arrays):
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def test_determinism():
    s1 = sample_spec(11, cfg=CFG)
    s2 = sample_spec(11, cfg=CFG)
    assert s1 == s2
    from floodsim.terrain import build_terrain
    from floodsim.rng import stream
    a = build_terrain(s1, CFG.world, stream(11, "terrain", s1.attempt))
    b = build_terrain(s1, CFG.world, stream(11, "terrain", s1.attempt))
    assert _hash(a.z, a.hand, a.landcover, a.road_mask) == _hash(b.z, b.hand, b.landcover, b.road_mask)


def test_drainage_to_outlet(specs):
    for s in specs:
        t, _ = terrain_for(s, CFG)
        assert fraction_draining_to(t, t.land, s.outlet_edge) >= 0.8


def test_filled_dem_has_no_interior_pits(specs):
    t, _ = terrain_for(specs[0], CFG)
    zf = priority_flood(t.z.astype(np.float64))
    n = zf.shape[0]
    inner = zf[1:-1, 1:-1]
    lowest_nb = np.full(inner.shape, np.inf)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                lowest_nb = np.minimum(lowest_nb, zf[1 + dy:n - 1 + dy, 1 + dx:n - 1 + dx])
    assert np.all(lowest_nb < inner)
    assert np.all(zf >= t.z - 1e-6)


def test_hand_properties(specs):
    for s in specs[:3]:
        t, _ = terrain_for(s, CFG)
        assert np.all(t.hand >= 0)
        assert np.all(t.hand[t.drainage_mask] == 0)
        assert np.all(t.drainage_mask[t.river_cells])


def test_synthetic_bowl_hydrology():
    # a tilted plane with a pit: the pit is filled and everything drains south
    n = 40
    y, x = np.mgrid[0:n, 0:n]
    z = 0.5 * y + 0.01 * np.abs(x - 20)
    z[20, 20] -= 5
    h = compute_hydrology(z.astype(np.float32), 50.0, 0.1)
    assert h.depression_depth[20, 20] > 4
    assert fraction_draining_to(h, np.ones((n, n), bool), "S") > 0.95


def test_terrain_layers_consistent(specs):
    t, _ = terrain_for(specs[1], CFG)
    assert np.all(t.landcover[t.river_cells] == WATER)
    assert np.allclose(t.surface, t.z + t.obstacle_h)
    assert t.road_mask.any() and t.towns
    assert abs(t.population.sum() - 1) < 1e-4
    bx, by = t.base_xy
    assert 0 < bx < t.domain_m and 0 < by < t.domain_m
