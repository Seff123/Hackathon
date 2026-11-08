import copy
import math

import numpy as np
import pytest

from floodsim.config import SimConfig
from floodsim.controllers import Controller, ScriptedSweep, strip_patterns
from floodsim.flood import FloodTimeline
from floodsim.people import PeopleState
from floodsim.scenario import World
from floodsim.sensor import Sensor, SensorFrame
from floodsim.sim import PlannerView, Sim
from floodsim.terrain import Terrain
from floodsim.weather import Thermal, Weather

from conftest import get_world, skip_perf, slow_enabled

TRUTH_TYPES = (World, FloodTimeline, PeopleState, Weather, Thermal, Terrain, Sim, Sensor)


def _walk(obj, seen, depth=0):
    if id(obj) in seen or depth > 8:
        return
    seen.add(id(obj))
    assert not isinstance(obj, TRUTH_TYPES), f"truth object reachable: {type(obj).__name__}"
    if callable(obj) and getattr(obj, "__closure__", None):
        for c in obj.__closure__:
            _walk(c.cell_contents, seen, depth + 1)
    if isinstance(obj, dict):
        for v in obj.values():
            _walk(v, seen, depth + 1)
    elif isinstance(obj, (list, tuple, set)):
        for v in list(obj)[:50]:
            _walk(v, seen, depth + 1)
    elif hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            _walk(v, seen, depth + 1)
    elif hasattr(obj, "__dataclass_fields__"):
        for f in obj.__dataclass_fields__:
            _walk(getattr(obj, f), seen, depth + 1)


class Spy(Controller):
    name = "spy"

    def __init__(self):
        self.inner = ScriptedSweep()
        self.views = []
        self.params = self.inner.params

    def reset(self, views):
        self.views.append(views)
        return self.inner.reset(views)

    def decide(self, views_due):
        self.views.append(views_due)
        return self.inner.decide(views_due)

    def observe(self, views_all):
        if len(self.views) < 40:
            self.views.append(views_all)


def test_information_boundary(cfg):
    w = get_world(2, cfg)
    spy = Spy()
    Sim(w, spy, SimConfig(fast=True, max_steps=900), n_drones=3).run()
    assert len(spy.views) > 3
    for vs in spy.views:
        for v in vs.values():
            assert isinstance(v, PlannerView)
            _walk(v, set())


def test_own_frames_in_view(cfg):
    w = get_world(2, cfg)
    spy = Spy()
    sim = Sim(w, spy, SimConfig(fast=True, max_steps=600), n_drones=2)
    sim.run()
    v = [vs[0] for vs in spy.views if 0 in vs][-1]
    assert v.frames is sim.frames[0] or v.frames == sim.frames[0]
    assert len(v.frames) > 50
    assert any(len(f.detections) == 0 for f in v.frames)
    assert all(isinstance(f, SensorFrame) and not hasattr(f, "pd") for f in v.frames)
    assert all(f.drone_id == 0 for f in v.frames)


def test_battery_swap_and_relaunch(cfg):
    w = get_world(2, cfg)
    sim = Sim(w, ScriptedSweep(), SimConfig(fast=True), n_drones=2)
    res = sim.run()
    ev = res.safety_events
    lands = [e for e in ev if e["kind"] == "landed"]
    rel = [e for e in ev if e["kind"] == "relaunch"]
    assert lands
    for e in lands:
        nxt = [r for r in rel if r["drone"] == e["drone"] and r["t"] > e["t"]]
        if cfg.world.episode_s - (e["t"] + w.spec.charge_time_min * 60) >= 20 * 60:
            assert nxt and nxt[0]["t"] == pytest.approx(e["t"] + w.spec.charge_time_min * 60, abs=1.5)
        else:
            assert not nxt


def test_fast_mode_identical_metrics(cfg):
    w = get_world(3, cfg)
    a = Sim(w, ScriptedSweep(), SimConfig(fast=True, max_steps=2400), n_drones=3).run()
    b = Sim(w, ScriptedSweep(), SimConfig(fast=False, max_steps=2400), n_drones=3).run()
    assert a.metrics == b.metrics
    assert len(b.trajectory) > len(a.trajectory)


def test_determinism(cfg):
    w = get_world(3, cfg)
    a = Sim(w, ScriptedSweep(), SimConfig(fast=True, max_steps=3000), n_drones=4).run()
    b = Sim(w, ScriptedSweep(), SimConfig(fast=True, max_steps=3000), n_drones=4).run()
    assert a.metrics == b.metrics
    assert np.array_equal(a.trajectory, b.trajectory)
    assert a.finds == b.finds


def test_scripted_partition(cfg):
    w = get_world(1, cfg)
    t = w.terrain
    mask = (t.hand < 10.0) & (t.landcover != 1)
    for n in (2, 5, 8):
        pats, sid = strip_patterns(mask, 50.0, n, 0.7, 80.0)
        counts = np.bincount(sid, minlength=n)
        assert counts.sum() == mask.sum()            # every floodplain cell in exactly one strip
        assert np.all(np.abs(counts - mask.sum() / n) <= 0.05 * mask.sum() / n + 1)
        assert all(len(p) >= 2 for p in pats)


@pytest.mark.skipif(not slow_enabled(), reason="slow")
def test_integration_scripted():
    from floodsim.config import load_config
    cfg = load_config()
    for seed in range(5):
        w = get_world(seed, cfg)
        for n in (2, 5, 8):
            r = Sim(w, ScriptedSweep(), SimConfig(fast=True), n_drones=n).run()
            for k, v in r.metrics.items():
                if isinstance(v, (int, float)):
                    assert math.isfinite(v), (seed, n, k)


@pytest.mark.skipif(not slow_enabled(), reason="slow")
def test_swarm_separation_50_runs():
    from floodsim.config import load_config
    cfg = load_config()
    bad = []
    for seed in range(50):
        w = get_world(seed, cfg)
        r = Sim(w, ScriptedSweep(), SimConfig(fast=True, max_steps=3600), n_drones=8).run()
        if r.metrics["midair_collisions"] or r.metrics["separation_breaches"]:
            bad.append((seed, r.metrics["midair_collisions"], r.metrics["separation_breaches"]))
    assert not bad, bad


@pytest.mark.skipif(skip_perf(), reason="FLOODSIM_SKIP_PERF=1")
def test_performance(cfg, tmp_path):
    import time
    from floodsim.scenario import build_world, sample_spec
    spec = sample_spec(7, cfg=cfg)
    t0 = time.perf_counter()
    build_world(spec, cache_dir=str(tmp_path), cfg=cfg)
    assert time.perf_counter() - t0 < 8.0
    t0 = time.perf_counter()
    w = build_world(spec, cache_dir=str(tmp_path), cfg=cfg)
    assert time.perf_counter() - t0 < 1.0
    t0 = time.perf_counter()
    Sim(w, ScriptedSweep(), SimConfig(fast=True), n_drones=8).run()
    assert time.perf_counter() - t0 < 6.0
