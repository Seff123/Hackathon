import os

import numpy as np
import pytest

from floodsim.config import FloodParams
from floodsim.flood import LocalInertialSolver, SolverState, build_flood, cache_path, run_flood
from floodsim.people import hazard_rating
from floodsim.scenario import sample_spec
from floodsim.terrain import generate_terrain
from floodsim import rng as R

from conftest import get_world, slow_enabled


def test_still_lake():
    rng = np.random.default_rng(0)
    z = rng.uniform(0, 2, (20, 20))
    h = 5.0 - z
    solver = LocalInertialSolver(z, np.full_like(z, 0.05), 100.0, FloodParams())
    st = SolverState(h=h.copy(), qx=np.zeros((20, 21)), qy=np.zeros((21, 20)), t=0.0)
    while st.t < 3600:
        solver.step(st, solver.max_dt(st.h))
    assert np.abs(st.qx).max() < 1e-6 and np.abs(st.qy).max() < 1e-6
    assert np.allclose(st.h + z, 5.0, atol=1e-9)


def test_dam_break_1d():
    nx = 120
    z = np.zeros((1, nx))
    h = np.where(np.arange(nx) < 30, 4.0, 0.0)[None, :].astype(float)
    solver = LocalInertialSolver(z, np.full_like(z, 0.03), 100.0, FloodParams())
    st = SolverState(h=h, qx=np.zeros((1, nx + 1)), qy=np.zeros((2, nx)), t=0.0)
    front_prev = 30
    vol0 = st.h.sum()
    for _ in range(300):
        solver.step(st, solver.max_dt(st.h))
        assert st.h.min() >= 0
        wet = np.flatnonzero(st.h[0] > 1e-3)
        front = wet.max()
        assert front >= front_prev
        front_prev = front
    assert front_prev > 40
    assert abs(st.h.sum() - vol0) / vol0 < 1e-9


def test_hazard_rating_cases():
    # HR = d (v + 0.5) + DF
    assert hazard_rating(0.2, 1.0) == pytest.approx(0.2 * 1.5 + 0.0)
    assert hazard_rating(0.5, 1.0) == pytest.approx(0.5 * 1.5 + 0.5)
    assert hazard_rating(1.0, 2.0) == pytest.approx(1.0 * 2.5 + 1.0)
    assert hazard_rating(0.0, 3.0) == 0.0


def test_mass_conservation(cfg):
    w = get_world(1, cfg)
    assert w.flood.mass["residual_frac"] < 0.01 or abs(w.flood.mass.get("residual", 0)) < 1.0


def test_time_window(cfg):
    for off in (-0.5, 4.0):
        spec = sample_spec(2, cfg=cfg, start_offset_h=off)
        ter = generate_terrain(spec, cfg.world, R.streams(2, spec.attempt)["terrain"])
        tl = run_flood(spec, ter, cfg, R.stream(2, spec.attempt, "flood"))
        assert tl.t_end == pytest.approx(cfg.world.episode_s + cfg.world.post_episode_s)
        onset = -off * 3600
        assert tl.mass["t_start"] <= min(onset, 0) - cfg.world.spinup_s + 1e-6
        assert tl.depth_snaps.shape[0] == int((cfg.world.episode_s + cfg.world.post_episode_s) / 60) + 1


def test_cache_size_and_reload(cfg, cache_dir):
    w = get_world(1, cfg)
    p = cache_path(w.spec, cache_dir)
    assert p.exists() and p.stat().st_size < 10e6
    ter = w.terrain
    fresh = run_flood(w.spec, ter, cfg, R.stream(w.spec.seed, w.spec.attempt, "flood"))
    loaded, cached = build_flood(w.spec, ter, cfg, cache_dir=cache_dir)
    assert cached
    assert np.array_equal(loaded.depth_snaps, fresh.depth_snaps.astype(np.float16))


def test_timeline_api(cfg):
    w = get_world(1, cfg)
    fl = w.flood
    d = fl.depth_grid50(600.0)
    assert d.shape == (200, 200) and d.min() >= 0
    hz = fl.hazard_grid(600.0)
    assert hz.shape == (100, 100) and np.isfinite(hz).all()
    x, y = np.array([1000.0, 5000.0]), np.array([1000.0, 5000.0])
    assert fl.depth(x, y, 0.0).shape == (2,)
    u, v = fl.velocity(x, y, 100.0)
    assert np.isfinite(u).all()


@pytest.mark.skipif(not slow_enabled(), reason="slow")
def test_extent_bounds_50_seeds(cfg):
    ok = 0
    for seed in range(50):
        w = get_world(seed, cfg)
        land = ~w.terrain.river_mask
        ext = ((w.flood.max_depth50() > cfg.flood.wet_depth) & land).sum() / land.sum()
        ok += 0.03 <= ext <= 0.40
    assert ok >= 45, ok
