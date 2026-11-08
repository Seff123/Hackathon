import math

import numpy as np
import pytest

from floodsim.config import G, GliderParams, TeamInputs
from floodsim.glider import GliderState, PerfTable, glider_step


def _fly(gp, V, phi, tau, steps, wind=(0.0, 0.0, 0.0)):
    tab = PerfTable(gp)
    s = GliderState.create(1, 0.0, 0.0, 1000.0, gp.capacity_wh)
    s.airborne[:] = True
    xs, ys, zs = [], [], []
    for _ in range(steps):
        glider_step(s, np.array([V]), np.array([phi]), np.array([tau]), tuple(np.array([w]) for w in wind), gp, tab, 1.0)
        xs.append(s.x[0]); ys.append(s.y[0]); zs.append(s.z[0])
    return np.array(xs), np.array(ys), np.array(zs), s


def test_best_glide():
    gp = GliderParams()
    tab = PerfTable(gp)
    V = tab.v_best_glide
    _, _, z, _ = _fly(gp, V, 0.0, 0.0, 100)
    sink = (z[0] - z[-1]) / 99
    q = 0.5 * 1.225 * V * V * gp.wing_area
    cl = gp.weight / q
    ld = cl / (gp.cd0 + gp.k * cl * cl)
    assert sink == pytest.approx(V / ld, rel=0.03)
    assert ld == pytest.approx(23.6, abs=0.3) and V == pytest.approx(10.2, abs=0.2)


def test_steady_turn_radius():
    gp = GliderParams()
    V, phi = 12.0, math.radians(30)
    x, y, _, _ = _fly(gp, V, phi, 0.0, 400)
    r_meas = 0.5 * (x.max() - x.min() + y.max() - y.min()) / 2
    assert r_meas == pytest.approx(V * V / (G * math.tan(phi)), rel=0.03)


def test_climb_and_power():
    gp = GliderParams()
    V = gp.climb_speed
    _, _, z, _ = _fly(gp, V, 0.0, 1.0, 60)
    rate = (z[-1] - 1000.0) / 60
    assert rate == pytest.approx((gp.thrust_max - float(gp.drag(V))) * V / gp.weight, rel=0.05)
    # 1 h of level flight at 14 m/s uses the drag-polar energy prediction
    Vc = 14.0
    tau = float(gp.drag(Vc)) / gp.thrust_max
    _, _, z, s = _fly(gp, Vc, 0.0, tau, 3600)
    used = gp.capacity_wh - s.E[0]
    assert used == pytest.approx(float(gp.level_power_w(Vc)), rel=0.10)
    assert abs(z[-1] - 1000.0) < 5.0


@pytest.mark.parametrize("mass", [2.0, 5.0, 12.0, 20.0])
def test_battery_life_input(mass):
    team = TeamInputs(mass_kg=mass, battery_life_min=150.0)
    gp = GliderParams.from_team(team)
    tau = float(gp.drag(gp.v_ref)) / gp.thrust_max
    assert tau <= 1.0
    s = GliderState.create(1, 0, 0, 5000.0, gp.capacity_wh)
    s.airborne[:] = True
    tab = PerfTable(gp)
    t = 0
    zero = tuple(np.zeros(1) for _ in range(3))
    while s.E[0] > 0 and t < 20000:
        glider_step(s, np.array([gp.v_ref]), np.zeros(1), np.array([tau]), zero, gp, tab, 1.0)
        t += 1
    assert t / 60 == pytest.approx(150.0, rel=0.02)


def test_battery_size_fixed_by_reference_speed():
    gp = GliderParams()
    cap = gp.capacity_wh
    from floodsim.config import ControllerParams
    _ = ControllerParams(search_speed_min=16.0)
    assert GliderParams().capacity_wh == cap
    assert cap == pytest.approx(180.0, rel=0.02)


def test_sanity_numbers():
    gp = GliderParams()
    assert float(gp.stall_speed(0.0)) == pytest.approx(8.3, abs=0.1)
    assert float(gp.sink(10.5, math.radians(35))) == pytest.approx(0.55, abs=0.06)
    assert float(gp.level_power_w(14.0)) == pytest.approx(79.0, abs=3.0)
