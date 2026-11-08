import math

import numpy as np
import pytest

from floodsim.autopilot import RTBM, SEARCH, THERMAL, VERIFY
from floodsim.config import AutopilotParams, GliderParams
from floodsim.goals import RTB, Search, Transit

from harness import Harness, flat


def test_crosswind_transit():
    h = Harness(flat(), wind=(0.0, 10.0))
    h.place(0, 2000.0, 3000.0, 200.0, psi=0.0, goal=Transit(xy=(7000.0, 3000.0)))
    evs = h.run(1500, until=lambda hh, e: any(k == "arrived" for _, k, _ in e))
    assert any(k == "arrived" for _, k, _ in evs)
    assert not h.crashed.any()


def test_thermal_climb():
    gp = GliderParams()
    th = (5000.0, 5000.0, 2.0, 150.0)
    h = Harness(flat(), thermals=[th], zi=600.0)
    # enters 50 m from the centre, flying tangentially; battery below E_soar so soaring is allowed
    goal = Transit(xy=(9000.0, 4950.0), allow_soaring=True)
    h.place(0, 5000.0, 4950.0, 50.0 + 298.0, psi=0.0, E=0.5 * gp.capacity_wh, goal=goal)
    h.ap.d[0].band = (250.0, 400.0)
    h.run(8)
    assert h.ap.d[0].mode == THERMAL
    assert math.hypot(h.ap.d[0].th_cx - 5000.0, h.ap.d[0].th_cy - 5000.0) < 60.0
    z0, t0 = h.gs.z[0], h.t
    for _ in range(80):
        h.step()
        if h.ap.d[0].mode != THERMAL:
            break
    rate = (h.gs.z[0] - z0) / (h.t - t0)
    assert rate > 0.5, rate
    assert h.ap.counters["stall_protect"][0] == 0


def test_one_drone_per_thermal():
    gp = GliderParams()
    th = (5000.0, 5000.0, 2.0, 150.0)
    h = Harness(flat(), n=2, thermals=[th])
    goal = Transit(xy=(9000.0, 9000.0), allow_soaring=True)
    h.place(0, 5000.0, 4910.0, 348.0, psi=math.pi / 2, E=0.5 * gp.capacity_wh, goal=goal)
    h.ap.d[0].band = (250.0, 400.0)
    h.run(20)
    assert h.ap.d[0].mode == THERMAL
    h.place(1, 4800.0, 4950.0, 300.0, psi=0.0, E=0.5 * gp.capacity_wh, goal=Transit(xy=(9000.0, 4950.0), allow_soaring=True))
    h.ap.d[1].band = (200.0, 400.0)
    for _ in range(60):
        h.step()
        assert h.ap.d[1].mode != THERMAL


def _ridge(n=200, h=150.0, x0=6000.0, width=600.0):
    xs = (np.arange(n) + 0.5) * 50
    X, _ = np.meshgrid(xs, xs)
    return 50.0 + h * np.exp(-((X - x0) / width) ** 2)


def test_terrain_safety_50_runs():
    rng = np.random.default_rng(0)
    ter = _ridge()
    for k in range(50):
        wind = (rng.uniform(-6, 6), rng.uniform(-6, 6))
        h = Harness(ter, wind=wind)
        y = rng.uniform(2000, 8000)
        h.place(0, 3000.0, y, 50.0 + rng.uniform(35, 90), psi=0.0, goal=Transit(xy=(9000.0, y)))
        h.ap.d[0].band = (rng.uniform(40, 80), rng.uniform(90, 140))
        h.run(900, until=lambda hh, e: hh.gs.x[0] > 8000)
        assert not h.crashed.any(), k


def test_search_over_forest_and_town_no_interventions():
    n = 200
    ter = flat(n, 50.0)
    obst = np.zeros((n, n))
    obst[60:120, 40:80] = 25.0     # forest
    obst[60:120, 100:140] = 20.0   # buildings
    h = Harness(ter + obst, terrain=ter)
    h.place(0, 1500.0, 2900.0, 50.0 + 60.0 + 25.0, psi=0.0,
            goal=Search(box=(1500.0, 3000.0, 7500.0, 6000.0), agl=60.0, speed=13.0, leg_angle_deg=0.0))
    h.ap.set_goal(0, Search(waypoints=[(1500.0, 3000.0 + k * 300.0) if j == 0 else (7500.0, 3000.0 + k * 300.0)
                                       for k in range(6) for j in ((0, 1) if k % 2 == 0 else (1, 0))],
                            agl=60.0, speed=13.0), h.gs, 0.0, 0.0)
    h.run(2500)
    assert h.ap.counters["terrain_warning"][0] == 0
    assert not h.crashed.any()


def _rtb_run(rng, wind, ter=None, base=(5000.0, 5000.0)):
    gp = GliderParams()
    ter = flat() if ter is None else ter
    h = Harness(ter, wind=wind, base=base)
    ang = rng.uniform(0, 2 * math.pi)
    dist = rng.uniform(2000, 4500)
    x, y = base[0] + dist * math.cos(ang), base[1] + dist * math.sin(ang)
    # energy a little above what the return rule needs, so the rule fires soon
    er = h.ap.e_return_wh(x, y, wind[0], wind[1])[0]
    E0 = er + rng.uniform(1.0, 8.0)
    h.place(0, x, y, 50.0 + 84.0, psi=rng.uniform(-3, 3), E=E0,
            goal=Search(box=(x - 1500, y - 1500, x + 1500, y + 1500), agl=84.0, speed=13.0))
    h.run(4000)
    return h


def test_rtb_lands_with_reserve():
    rng = np.random.default_rng(1)
    ok = 0
    for k in range(50):
        wind = tuple(rng.uniform(-5, 5, 2))
        h = _rtb_run(rng, wind)
        ok += bool(h.landed[0] and h.land_E[0] >= h.gp.reserve_wh)
        assert not h.crashed[0], k
    assert ok >= 48


def test_rtb_strong_wind_from_base():
    rng = np.random.default_rng(2)
    ok = 0
    from floodsim.controllers import wind_at_height
    for k in range(50):
        h0 = Harness(flat())
        ang = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(2000, 4500)
        # 8 m/s at 10 m blowing from the base towards the drone (headwind on the way home)
        base = (5000.0, 5000.0)
        x, y = base[0] + dist * math.cos(ang), base[1] + dist * math.sin(ang)
        wind = (8.0 * math.cos(ang), 8.0 * math.sin(ang))
        gp = GliderParams()
        h = Harness(flat(), wind=wind_at_height(*wind, 84.0), base=base)
        er = h.ap.e_return_wh(x, y, *h.wind)[0]
        h.place(0, x, y, 134.0, psi=0.0, E=er + rng.uniform(1.0, 6.0),
                goal=Search(box=(x - 1000, y - 1000, x + 1000, y + 1000), agl=84.0, speed=17.0))
        h.run(5000)
        ok += bool(h.landed[0] and h.land_E[0] >= gp.reserve_wh)
    assert ok >= 48


def test_landing_profile():
    rng = np.random.default_rng(3)
    for k in range(10):
        h = _rtb_run(rng, (rng.uniform(-4, 4), rng.uniform(-4, 4)))
        assert h.landed[0]
        bx, by = h.base
        for t, xs, ys, zs, modes, _, _ in h.log:
            if modes[0] == RTBM:
                hs = zs[0] - h.ap.surf_at(xs[0], ys[0])
                if hs < 30.0:
                    assert math.hypot(xs[0] - bx, ys[0] - by) <= 150.0
        lx, ly = h.land_xy[0]
        assert math.hypot(lx - bx, ly - by) <= 100.0


def test_orbit_radius_at_base():
    rng = np.random.default_rng(4)
    h = _rtb_run(rng, (0.0, 0.0))
    bx, by = h.base
    near = [math.hypot(xs[0] - bx, ys[0] - by) for t, xs, ys, zs, m, _, _ in h.log[-60:]]
    assert max(near) <= 120.0
    assert AutopilotParams().rtb_orbit_r <= 80.0


def test_base_zone_8_drones_land():
    gp = GliderParams()
    h = Harness(flat(), n=8)
    rng = np.random.default_rng(5)
    for i in range(8):
        ang = 2 * math.pi * i / 8
        d = 1000.0 + 150.0 * i
        x, y = 5000 + d * math.cos(ang), 5000 + d * math.sin(ang)
        h.place(i, x, y, 150.0 + 10 * i, psi=ang + math.pi, E=gp.capacity_wh * 0.5, goal=RTB())
    h.run(1500)
    assert h.landed.all() and not h.crashed.any()
