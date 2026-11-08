import dataclasses
from types import SimpleNamespace

import numpy as np

from floodsim.scenario import sample_spec
from floodsim.terrain import OPEN
from floodsim.weather import Weather, thermal_shape


def _hill_weather(cfg, wind_from_deg=270.0, wstar=0.0):
    n = 200
    xs = (np.arange(n) + 0.5) * 50
    X, Y = np.meshgrid(xs, xs)
    z = 50 + 150 * np.exp(-((X - 5000) ** 2 + (Y - 5000) ** 2) / (2 * 800 ** 2))
    ter = SimpleNamespace(z=z.astype(np.float32), landcover=np.full((n, n), OPEN, np.int8), cell_size=50.0, n=n)
    spec = dataclasses.replace(sample_spec(1, cfg=cfg), wind10_ms=6.0, wind_from_deg=wind_from_deg,
                               wstar_ms=wstar, thermal_density_km2=0.0, ridge_k=1.0)
    return Weather(spec, ter, cfg, np.random.default_rng(0)), z


def test_log_profile_increases(cfg):
    w, _ = _hill_weather(cfg)
    s = w.speed_at(np.array([10, 50, 100, 300, 599]))
    assert np.all(np.diff(s) > 0)
    assert abs(w.speed_at(10.0) - 6.0) < 1e-9
    assert w.speed_at(800.0) == w.speed_at(600.0)


def test_ridge_lift_sign(cfg):
    w, z = _hill_weather(cfg, wind_from_deg=270.0)   # wind from the west, blowing east
    zz = 80.0
    up = w.wind(np.array([4300.0]), np.array([5000.0]), np.array([z[100, 86] + zz]))[2][0]   # west slope: windward
    lee = w.wind(np.array([5700.0]), np.array([5000.0]), np.array([z[100, 114] + zz]))[2][0]
    assert up > 0.2 and lee < -0.2


def test_thermal_shape(cfg):
    w, _ = _hill_weather(cfg)
    from floodsim.weather import Thermal
    w.thermal_list = [Thermal(2000.0, 2000.0, 2.0, 150.0)]
    w._tx, w._ty, w._tw, w._tr = (np.array([2000.0]), np.array([2000.0]), np.array([2.0]), np.array([150.0]))
    r = np.array([0.0, 50.0, 100.0, 200.0, 400.0])
    wz = w.thermal_w(2000.0 + r, np.full(5, 2000.0), np.full(5, 0.23 * w.zi))
    assert np.argmax(wz) == 0 and np.all(np.diff(wz) < 0)
    assert abs(thermal_shape(0.25 / 1.1) - 1.0) < 1e-9
    assert thermal_shape(0.0) == 0 and thermal_shape(0.95) == 0
