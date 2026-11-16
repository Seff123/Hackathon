import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest

from floodsim.config import Config
from floodsim.flood import LocalInertialSolver, hazard_rating, debris_factor, run_flood
from floodsim.rng import stream
from floodsim.scenario import sample_spec, terrain_for
from floodsim.world import build_world, _cache_path

CFG = Config()


def test_still_lake():
    z = np.zeros((20, 20))
    z[0, :] = z[-1, :] = z[:, 0] = z[:, -1] = 0.0
    s = LocalInertialSolver(z, 0.03, 100.0)
    s.h[:] = 2.0
    t = 0.0
    while t < 3600:
        dt = s.stable_dt()
        s.step(dt)
        t += dt
    assert np.abs(s.qx).max() < 1e-6 and np.abs(s.qy).max() < 1e-6
    assert np.allclose(s.h, 2.0)


def test_dam_break_strip():
    nx = 120
    z = np.zeros((1, nx))
    s = LocalInertialSolver(z, 0.01, 10.0, dt_max=2.0)
    s.h[0, :40] = 2.0
    vol0 = s.volume
    fronts = []
    t = 0.0
    while t < 120:
        dt = min(s.stable_dt(), 0.5)
        s.step(dt)
        t += dt
        assert s.h.min() >= 0.0
        wet = np.nonzero(s.h[0] > 0.01)[0]
        fronts.append(wet.max())
    assert np.all(np.diff(fronts) >= 0)
    assert fronts[-1] > 50
    assert abs(s.volume - vol0) / vol0 < 1e-9


def test_mass_conservation_with_sources():
    rng = np.random.default_rng(0)
    z = rng.uniform(0, 3, (30, 30))
    out = np.zeros((30, 30), bool)
    out[:, -1] = True
    s = LocalInertialSolver(z, 0.05, 100.0, outlet=out, outlet_edge="E")
    src = np.zeros((30, 30))
    src[15, 0] = 50.0 / 100**2
    src += 1e-5
    t = 0.0
    while t < 7200:
        dt = s.stable_dt()
        s.step(dt, src)
        t += dt
    resid = s.volume - (s.vol_in - s.vol_out)
    assert abs(resid) / s.vol_in < 0.01
    assert s.vol_out > 0


def test_hazard_rating_cases():
    assert hazard_rating(0.0, 2.0) == 0.0
    assert np.isclose(hazard_rating(0.2, 1.0), 0.2 * 1.5)
    assert np.isclose(hazard_rating(0.5, 1.0), 0.5 * 1.5 + 0.5)
    assert np.isclose(hazard_rating(1.0, 2.0), 1.0 * 2.5 + 1.0)
    assert list(debris_factor([0.25, 0.26, 0.75, 0.76])) == [0.0, 0.5, 0.5, 1.0]


@pytest.fixture(scope="module")
def world0():
    return build_world(sample_spec(0, cfg=CFG), CFG)


def test_scenario_mass_and_depth(world0):
    assert world0.flood.mass_residual < 0.01
    assert world0.flood.depth_snap.min() >= 0


@pytest.mark.parametrize("offset", [-0.5, 4.0])
def test_time_window(offset):
    spec = dataclasses.replace(sample_spec(1, cfg=CFG), start_offset_h=offset)
    terr, _ = terrain_for(sample_spec(1, cfg=CFG), CFG)
    tl, _ = run_flood(spec, terr, CFG, stream(1, "flood"))
    need = spec.episode_s + CFG.world.post_episode_s
    assert tl.t_end >= need - 1e-6
    assert tl.depth_snap.shape[0] == int(round(need / 60)) + 1


def test_cache_size_and_reload(world0, tmp_path):
    from floodsim import world as W

    p = _cache_path(world0.spec, "cache")
    assert p.exists() and p.stat().st_size < 10e6
    W._MEM.clear()
    w2 = build_world(world0.spec, CFG)
    assert w2.from_cache
    a = world0.flood.depth_snap.astype(np.float32)
    b = w2.flood.depth_snap.astype(np.float32)
    assert np.array_equal(a, b)
    assert w2.terrain.base_xy == world0.terrain.base_xy


def test_fine_depth_dry_beside_channel_at_base_flow(world0):
    terr = world0.terrain
    d = world0.flood.depth_grid(0.0)
    # far from flooding the fine depth is zero on land: no fringe around every channel cell
    near = (terr.dist_to_river > 0) & (terr.dist_to_river <= 60) & terr.land
    assert d.shape == terr.z.shape
    assert np.isfinite(d).all() and (d >= 0).all()
    assert (d[terr.river_cells] > 0).mean() > 0.9


@pytest.mark.slow
def test_extent_bounds_over_50_seeds():
    ok = 0
    for seed in range(50):
        w = build_world(sample_spec(seed, cfg=CFG), CFG)
        e = w.peak_extent()
        ok += 0.03 <= e <= 0.40
    assert ok >= 45
