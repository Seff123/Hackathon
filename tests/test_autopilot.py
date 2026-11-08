import math

import numpy as np
import pytest

from floodsim.autopilot import RTB_M, THERMAL, SEARCH
from floodsim.goals import RTB, Loiter, Search, Transit, lawnmower_legs

from synth import Env, u10_for


def test_crosswind_transit():
    env = Env(u10=u10_for(10.0, 175.0), wind_to=math.pi / 2, base=(500, 5000))
    c = env.core
    c.place(0, 1000, 5000, 180, 0.0, V=15)
    c.auto.set_goal(0, Transit(x=6000, y=5000))
    env.run(1500, lambda e: any(ev.kind == "arrived" for ev in e.core.auto.events))
    assert any(ev.kind == "arrived" for ev in c.auto.events)
    assert c.log.crashes == 0


def _thermal_env(w=2.0, rt=150.0, zi=600.0, u10=0.0, drones=1):
    return Env(thermals=[[5000.0, 5000.0, w, rt]], w_star=1.0, zi=zi, u10=u10, base=(1000, 1000),
               drones=drones)


def test_thermal_climb():
    # core strength = 2 m/s above this airframe's circling sink (the plan's test used a
    # 2 m/s core for the 5 kg placeholder, whose circling sink was ~0.6 m/s)
    from floodsim.glider import Performance
    from floodsim.config import Config

    cfg = Config()
    perf = Performance(cfg.glider)
    bank = math.radians(cfg.autopilot.thermal_bank_deg)
    v = cfg.autopilot.thermal_speed_factor * float(perf.stall_speed(bank))
    env = _thermal_env(w=2.0 + float(perf.sink(v, bank)))
    c = env.core
    c.place(0, 5050.0, 5000.0, 300.0, math.pi / 2, V=11.0, E=0.5 * c.perf.capacity_wh)
    c.auto.set_goal(0, Loiter(x=5050.0, y=5000.0, agl=300.0, allow_soaring=True))
    c.auto.enter_thermal(0, 5050.0, 5000.0, 0.0, c.s)
    z0 = c.s.z[0]
    env.run(60)
    assert c.auto.mode[0] == THERMAL
    assert (c.s.z[0] - z0) / 60.0 > 0.5
    assert c.auto.counts["stall_protect"][0] == 0


def test_thermal_entry_and_speed():
    """Flying through lift with soaring allowed: enters, climbs, never trips stall protection."""
    env = _thermal_env(w=2.5)
    c = env.core
    c.place(0, 4400.0, 5000.0, 250.0, 0.0, V=15.0, E=0.5 * c.perf.capacity_wh)
    c.auto.set_goal(0, Transit(x=8000.0, y=5000.0, band=(200, 300), allow_soaring=True))
    entered = False
    for _ in range(400):
        env.run(1)
        entered |= c.auto.mode[0] == THERMAL
    assert entered
    assert c.auto.counts["stall_protect"][0] == 0


def test_one_drone_per_thermal():
    env = _thermal_env(w=2.5, drones=2)
    c = env.core
    for i in range(2):
        c.place(i, 4400.0, 5000.0 + 60 * i, 250.0 + 60 * i, 0.0, V=15.0, E=0.5 * c.perf.capacity_wh)
    c.auto.set_goal(0, Loiter(x=5000.0, y=5000.0, agl=250.0, allow_soaring=True))
    c.auto.enter_thermal(0, 5000.0, 5000.0, 0.0, c.s)
    c.place(1, 4000.0, 4900.0, 320.0, 0.0, V=15.0, E=0.5 * c.perf.capacity_wh)
    c.auto.set_goal(1, Transit(x=8000.0, y=5000.0, band=(300, 400), allow_soaring=True))
    passed_through = False
    for _ in range(400):
        env.run(1)
        passed_through |= abs(c.s.x[1] - 5000.0) < 100.0
        if c.auto.mode[1] == THERMAL and c.auto.mode[0] == THERMAL:
            d = math.hypot(c.auto.ox[1] - c.auto.ox[0], c.auto.oy[1] - c.auto.oy[0])
            assert d >= env.cfg.autopilot.thermal_exclusion_m
    assert passed_through


def _ridge(n=200, cell=50.0, height=200.0, sigma=500.0, x0=5000.0):
    xs = (np.arange(n) + 0.5) * cell
    X, _ = np.meshgrid(xs, xs)
    return (height * np.exp(-((X - x0) ** 2) / (2 * sigma**2))).astype(np.float32)


@pytest.mark.slow
def test_terrain_safety_towards_ridge():
    rng = np.random.default_rng(1)
    crashes = 0
    for run in range(50):
        h = rng.uniform(100, 280)
        sig = rng.uniform(350, 900)
        env = Env(z=_ridge(height=h, sigma=sig), u10=rng.uniform(0, 6), wind_to=rng.uniform(-np.pi, np.pi),
                  base=(1000, 5000), seed=run)
        c = env.core
        y0 = rng.uniform(3000, 7000)
        c.place(0, 1500.0, y0, rng.uniform(60, 120), rng.uniform(-0.4, 0.4), V=15.0)
        if run % 2:
            legs = np.array([[[1500.0, y0], [9000.0, y0]]])
            c.auto.set_goal(0, Search(legs=legs, agl=60.0, speed=13.0))
        else:
            c.auto.set_goal(0, Transit(x=9000.0, y=y0, band=(60, 110)))
        env.run(900)
        crashes += c.log.crashes
    assert crashes == 0


def _rtb_runs(n_runs, wind_from_base=False, seed=0):
    rng = np.random.default_rng(seed)
    ok = 0
    results = []
    for run in range(n_runs):
        base = (1500.0, 1500.0)
        ang = rng.uniform(0.2, 1.3)
        dist = rng.uniform(3000, 8000)
        x, y = base[0] + dist * math.cos(ang), base[1] + dist * math.sin(ang)
        if wind_from_base:
            u10, wto = 8.0, ang  # blows from the base towards the drone: headwind home
        else:
            u10, wto = rng.uniform(1, 8), rng.uniform(-np.pi, np.pi)
        env = Env(u10=u10, wind_to=wto, base=base, seed=run)
        c = env.core
        cap = c.perf.capacity_wh
        c.place(0, x, y, 180.0, rng.uniform(-np.pi, np.pi), V=15.0, E=rng.uniform(0.45, 0.6) * cap)
        c.auto.set_goal(0, Loiter(x=x, y=y, agl=150.0))
        env.run(9000, lambda e: not e.core.s.active[0])
        landed = bool(c.s.landed[0])
        good = landed and c.s.E[0] >= c.auto.reserve_wh - 1e-6 and c.log.crashes == 0
        ok += good
        results.append((landed, float(c.s.E[0]), c.auto.reserve_wh))
    return ok, results


@pytest.mark.slow
def test_rtb_lands_with_reserve():
    ok, res = _rtb_runs(50)
    assert ok >= 48, res


@pytest.mark.slow
def test_rtb_strong_wind_from_base():
    ok, res = _rtb_runs(50, wind_from_base=True, seed=7)
    assert ok >= 48, res


def test_landing_profile():
    env = Env(u10=3.0, wind_to=0.3, base=(2000.0, 2000.0))
    c = env.core
    c.place(0, 5000.0, 4000.0, 200.0, 0.0, V=15.0)
    c.auto.set_goal(0, RTB())
    base = np.array([2000.0, 2000.0])
    worst_floor_dist = 0.0
    max_r = 0.0
    while c.s.active[0] and env.t < 3000:
        env.run(1)
        d = math.hypot(c.s.x[0] - base[0], c.s.y[0] - base[1])
        h = c.s.z[0] - env.surface_at(c.s.x[:1], c.s.y[:1])[0]
        if h < 30.0 and c.s.active[0]:
            worst_floor_dist = max(worst_floor_dist, d)
        if c.auto.rtb_phase[0] == 1 and h < 100:
            max_r = max(max_r, d)
    assert c.s.landed[0] and c.log.crashes == 0
    assert worst_floor_dist <= 150.0
    assert max_r <= 100.0
    assert math.hypot(c.s.x[0] - base[0], c.s.y[0] - base[1]) <= 100.0


def test_search_over_forest_and_town_no_interventions():
    n = 200
    obst = np.zeros((n, n), np.float32)
    obst[:, 60:100] = 25.0  # forest
    obst[:, 100:130] = 20.0  # buildings
    env = Env(obstacles=obst, u10=3.0, wind_to=np.pi / 2, base=(1000, 5000))
    c = env.core
    c.place(0, 2000.0, 4000.0, 70.0, 0.0, V=13.0)
    legs = lawnmower_legs(2500.0, 4000.0, 5000.0, 600.0, 80.0, 0.0)
    c.auto.set_goal(0, Search(legs=legs, agl=60.0, speed=13.0))
    env.run(240)  # settle onto the first leg at 60 m before checking
    c.auto.counts["terrain_warning"][:] = 0
    env.run(1800)
    assert c.auto.counts["terrain_warning"][0] == 0
    assert c.log.crashes == 0


def test_base_zone_mass_landing():
    env = Env(u10=4.0, wind_to=1.0, base=(3000.0, 3000.0), drones=8)
    c = env.core
    rng = np.random.default_rng(3)
    for i in range(8):
        a = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(1200, 2000)
        c.place(i, 3000 + r * math.cos(a), 3000 + r * math.sin(a), 150.0 + 50 * (i % 5), a + np.pi, V=15.0)
        c.auto.set_goal(i, RTB())
    env.run(1800, lambda e: not e.core.s.active.any())
    assert c.s.landed.all(), c.log.events
    assert c.log.crashes == 0


@pytest.mark.slow
def test_swarm_separation():
    breaches = collisions = 0
    for run in range(50):
        rng = np.random.default_rng(100 + run)
        env = Env(u10=rng.uniform(1, 7), wind_to=rng.uniform(-np.pi, np.pi), base=(5000.0, 5000.0),
                  drones=8, seed=run)
        c = env.core
        for i in range(8):
            x, y = rng.uniform(1500, 8500, 2)
            gx, gy = rng.uniform(1500, 8500, 2)
            c.place(i, x, y, 150.0 + 50 * (i % 5) + rng.uniform(-20, 20), math.atan2(gy - y, gx - x), V=15.0)
            c.auto.set_goal(i, Transit(x=gx, y=gy))  # random crossing routes
        env.run(700)
        breaches += c.log.separation_breaches
        collisions += c.log.collisions
    assert collisions == 0
    assert breaches == 0
