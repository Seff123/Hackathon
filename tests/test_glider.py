import math

import numpy as np
import pytest

from floodsim.config import Config, ControllerParams, GliderParams, TeamInputs
from floodsim.glider import GliderState, Performance, glider_step, terrain_contact

CFG = Config()
GP = CFG.glider
PERF = Performance(GP)
CALM = (np.zeros(1), np.zeros(1), np.zeros(1))


def fly(V, phi, tau, seconds, perf=PERF, z=500.0):
    s = GliderState.at(1, 0.0, 0.0, z, V=V, E=perf.capacity_wh)
    traj = []
    for _ in range(int(seconds)):
        glider_step(s, np.array([V]), np.array([phi]), np.array([tau]), CALM, perf, 1.0)
        traj.append((s.x[0], s.y[0], s.z[0], s.E[0]))
    return s, np.array(traj)


def test_still_air_glide_at_best_glide_speed():
    k = GP.k_induced
    cl_star = math.sqrt(GP.cd0 / k)
    v_md = math.sqrt(2 * GP.weight_n / (GP.rho * GP.wing_area_m2 * cl_star))
    ld = 1 / (2 * math.sqrt(GP.cd0 * k))
    _, tr = fly(v_md, 0.0, 0.0, 60)
    sink = -(tr[-1, 2] - tr[0, 2]) / 59
    assert sink == pytest.approx(v_md / ld, rel=0.03)
    # sanity numbers for the 5 kg placeholder
    assert ld == pytest.approx(23.6, rel=0.01) and v_md == pytest.approx(10.2, rel=0.02)


def test_steady_turn_radius():
    V, phi = 12.0, math.radians(30)
    period = 2 * math.pi * V / (9.81 * math.tan(phi))
    _, tr = fly(V, phi, PERF.level_throttle(V, phi), int(round(period * 5)))
    pts = tr[:, :2]
    c = pts.mean(axis=0)
    r = np.hypot(*(pts - c).T).mean()
    assert r == pytest.approx(V * V / (9.81 * math.tan(phi)), rel=0.03)


def test_climb_rate_and_level_energy():
    V = 12.0
    _, tr = fly(V, 0.0, 1.0, 30)
    climb = (tr[-1, 2] - tr[0, 2]) / 29
    expect = (GP.thrust_max_n - float(GP.drag(V))) * V / GP.weight_n
    assert climb == pytest.approx(expect, rel=0.05)
    assert expect == pytest.approx(2.9, rel=0.05)
    V = 14.0
    s, tr = fly(V, 0.0, float(PERF.level_throttle(V)), 3600)
    used = PERF.capacity_wh - s.E[0]
    pred = float(GP.drag(V)) * V / GP.prop_eff + GP.payload_w
    assert used == pytest.approx(pred, rel=0.10)
    assert abs(tr[-1, 2] - tr[0, 2]) < 1.0


@pytest.mark.parametrize("mass", [3.0, 5.0, 10.0, 15.0])
def test_battery_life_input(mass):
    gp = GliderParams.from_team(TeamInputs(mass_kg=mass, battery_life_min=150.0))
    perf = Performance(gp)
    tau = float(perf.level_throttle(gp.v_ref))
    assert tau <= 1.0
    s = GliderState.at(1, 0, 0, 1000.0, V=gp.v_ref, E=perf.capacity_wh)
    t = 0
    while s.E[0] > 0 and t < 20000:
        glider_step(s, np.array([gp.v_ref]), np.zeros(1), np.array([tau]), CALM, perf, 1.0)
        t += 1
    assert t / 60 == pytest.approx(150.0, rel=0.02)


def test_battery_size_independent_of_search_speed():
    a = Config(controller=ControllerParams(sweep_speed_min=11.0)).glider.battery_capacity_wh
    b = Config(controller=ControllerParams(sweep_speed_min=16.0)).glider.battery_capacity_wh
    assert a == b


def test_crash_into_hill():
    s = GliderState.at(1, 0.0, 0.0, 50.0, V=15.0, E=100.0)

    def surface(x, y):
        return np.where(np.asarray(x) > 500.0, 120.0, 0.0)

    hit = False
    for _ in range(60):
        glider_step(s, np.array([15.0]), np.zeros(1), np.array([float(PERF.level_throttle(15.0))]), CALM, PERF, 1.0)
        if terrain_contact(s, surface)[0]:
            hit = True
            break
    assert hit


def test_performance_table_matches_polar():
    for V in (10.0, 13.7, 21.0):
        for b in (0.0, 0.3, 0.6):
            assert PERF.sink(V, b) == pytest.approx(float(GP.drag(V, b)) * V / GP.weight_n, rel=0.01)
    assert 18 <= PERF.fast_climb_speed <= 24
