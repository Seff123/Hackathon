import dataclasses
import math
import os
import time

import numpy as np
import pytest

from floodsim.config import Config, SimConfig, TeamInputs
from floodsim.controllers import ScriptedSweep, partition_floodplain
from floodsim.scenario import sample_spec
from floodsim.sensor import SensorFrame
from floodsim.sim import PlannerView, Sim
from floodsim.world import build_world

from helpers import reachable_truth

CFG = Config()


def world(seed, cfg=CFG):
    return build_world(sample_spec(seed, cfg=cfg), cfg)


def run(seed, n, cfg=CFG, **sim_kw):
    w = world(seed, cfg)
    sc = dataclasses.replace(cfg.sim, **sim_kw)
    return Sim(w, ScriptedSweep(cfg.controller), sc, n_drones=n, cfg=cfg)


def test_partition_equal_and_complete():
    rng = np.random.default_rng(0)
    from scipy import ndimage

    mask = ndimage.gaussian_filter(rng.random((200, 200)), 6) > 0.5
    for n in (2, 5, 8):
        labels, _ = partition_floodplain(mask, n, 0.7, 50.0)
        assert np.all(labels[mask] >= 0) and np.all(labels[~mask] == -1)
        counts = np.bincount(labels[mask], minlength=n)
        assert counts.max() <= counts.mean() * 1.05 and counts.min() >= counts.mean() * 0.95


def test_information_boundary():
    sim = run(2, 3, episode_s=600.0, fast=True)
    seen = []
    orig = sim.controller.decide

    def spy(views):
        seen.extend(views.values())
        return orig(views)

    sim.controller.decide = spy
    sim.run()
    assert seen
    for v in seen[:5] + seen[-5:]:
        assert isinstance(v, PlannerView)
        assert reachable_truth(v) is None, reachable_truth(v)


def test_own_frames_in_view():
    sim = run(2, 2, episode_s=900.0, fast=True)
    last = {}
    orig = sim.controller.decide

    def spy(views):
        last.update(views)
        return orig(views)

    sim.controller.decide = spy
    sim.run()
    for i, v in last.items():
        assert all(isinstance(f, SensorFrame) and f.drone_id == i for f in v.frames)
        assert any(len(f.detections) == 0 for f in v.frames)  # empty frames are included
        assert not any(hasattr(f, "p_d") or hasattr(f, "pd") for f in v.frames)
        seqs = [f.seq for f in v.frames]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    # every frame the sensor produced reached its drone (the final step may not have a decision yet)
    for i in range(2):
        assert len(sim.frames[i]) == int(sim.sensor.seq[i])


def test_fast_mode_identical_metrics_and_determinism():
    a = run(4, 3, episode_s=2400.0, fast=False).run()
    b = run(4, 3, episode_s=2400.0, fast=True).run()
    c = run(4, 3, episode_s=2400.0, fast=True).run()
    assert a.metrics == b.metrics
    assert b.metrics == c.metrics
    assert len(a.trajectory["t"]) > len(b.trajectory["t"])


def test_battery_swap_and_relaunch():
    cfg = Config(team=TeamInputs(battery_life_min=40.0, charge_time_min=10.0))
    sim = run(5, 3, cfg=cfg, episode_s=5400.0, fast=True)
    res = sim.run()
    landings = [(t, i) for (t, kind, i, info) in res.safety_events if kind == "landed"]
    assert landings, "short battery should force returns"
    relaunch = {i: [] for i in range(3)}
    for t, i in sim.relaunches:
        relaunch[i].append(t)
    for t_land, i in landings:
        ready = t_land + 600.0
        should = 5400.0 - ready >= cfg.sim.relaunch_min_remaining_s
        got = any(abs(tr - ready) <= 1.0 for tr in relaunch[i])
        assert got == should, (t_land, i, relaunch[i])


@pytest.mark.slow
@pytest.mark.parametrize("n", [2, 5, 8])
def test_integration_scripted(n):
    for seed in range(5):
        res = run(seed, n, fast=True).run()
        for k, v in res.metrics.items():
            assert np.isfinite(v), (seed, n, k, v)
        assert res.metrics["mid_air_collisions"] == 0


@pytest.mark.perf
def test_performance_budget():
    if os.environ.get("FLOODSIM_SKIP_PERF") == "1":
        pytest.skip("FLOODSIM_SKIP_PERF=1")
    from floodsim import world as W

    w = world(0)
    W._MEM.clear()
    t0 = time.perf_counter()
    build_world(w.spec, CFG)  # cached load
    cached = time.perf_counter() - t0
    sc = dataclasses.replace(CFG.sim, fast=True)
    t0 = time.perf_counter()
    Sim(w, ScriptedSweep(CFG.controller), sc, n_drones=8).run()
    ep8 = time.perf_counter() - t0
    msg = f"cached build {cached:.2f} s (budget 0.5), 8-drone fast episode {ep8:.1f} s (budget 6)"
    over = cached > 0.5 or ep8 > 6.0
    if over and os.environ.get("FLOODSIM_STRICT_PERF") != "1":
        pytest.xfail("soft performance budget exceeded: " + msg)
    assert not over, msg
