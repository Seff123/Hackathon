import math

import numpy as np

from floodsim.config import Config, WeatherParams
from floodsim.weather import Weather, thermal_profile

WP = WeatherParams()


def flat(n=80, cell=50.0):
    return np.zeros((n, n), np.float32), cell


def test_log_profile_increases_to_600m():
    z, c = flat()
    w = Weather(z, c, 5.0, 0.0, 1.0, 0.0, 1000.0, np.zeros((0, 4)), WP)
    hs = np.arange(2.0, 600.0, 10.0)
    sp = w.speed(hs)
    assert np.all(np.diff(sp) > 0)
    assert math.isclose(float(w.speed(10.0)), 5.0, rel_tol=1e-9)
    assert w.speed(800.0) == w.speed(600.0)
    # 8 m/s at 10 m gives at most ~14 m/s at 300 m
    w8 = Weather(z, c, 8.0, 0.0, 1.0, 0.0, 1000.0, np.zeros((0, 4)), WP)
    assert float(w8.speed(300.0)) < 15.0


def test_ridge_lift_sign_on_gaussian_hill():
    n, c = 120, 50.0
    X, Y = np.meshgrid((np.arange(n) + 0.5) * c, (np.arange(n) + 0.5) * c)
    z = 150.0 * np.exp(-((X - 3000) ** 2 + (Y - 3000) ** 2) / (2 * 600.0**2))
    # wind blowing towards +x (from the west): windward slope is west of the hill
    w = Weather(z, c, 6.0, 0.0, 1.0, 0.0, 1000.0, np.zeros((0, 4)), WP)
    xw, xl, y = 2400.0, 3600.0, 3000.0
    zw = float(w.z[int(y // c), int(xw // c)]) + 80
    zl = float(w.z[int(y // c), int(xl // c)]) + 80
    assert w.wind(xw, y, zw)[2] > 0.2
    assert w.wind(xl, y, zl)[2] < -0.2
    assert w.wind(xl, y, zl)[2] >= WP.ridge_sink_clip


def test_thermal_shape():
    z, c = flat()
    th = np.array([[2000.0, 2000.0, 2.0, 150.0]])
    w = Weather(z, c, 0.0, 0.0, 1.0, 1.5, 600.0, th, WP)
    rs = np.array([0, 50, 100, 150, 250, 400.0])
    ws = w.thermal(2000.0 + rs, np.full(rs.shape, 2000.0), np.full(rs.shape, 0.25 * 600))
    assert np.argmax(ws) == 0 and np.all(np.diff(ws) < 0)
    assert abs(thermal_profile(1 / 4.4) - 1.0) < 1e-9
    assert thermal_profile(0.0) == 0 and thermal_profile(0.95) == 0
    # profile peak at the core is w_core
    assert abs(float(w.thermal(2000.0, 2000.0, 600 / 4.4)[0]) - 2.0) < 1e-6
