"""Synthetic environments for autopilot tests."""

import math

import numpy as np

from floodsim.config import AutopilotParams, Config, ControllerParams, WeatherParams
from floodsim.flight import FlightCore
from floodsim.grid import bilinear
from floodsim.weather import Weather


class Env:
    def __init__(self, z=None, n=200, cell=50.0, obstacles=None, u10=0.0, wind_to=0.0, thermals=None,
                 w_star=0.0, zi=1000.0, base=(1000.0, 1000.0), drones=1, seed=0, cfg=None,
                 ap=None, cp=None, k_ridge=1.0):
        cfg = cfg or Config()
        if z is None:
            z = np.zeros((n, n), np.float32)
        self.z = np.asarray(z, np.float32)
        n = self.z.shape[0]
        self.cell = cell
        self.L = n * cell
        self.surface = self.z + (0 if obstacles is None else np.asarray(obstacles, np.float32))
        th = np.zeros((0, 4)) if thermals is None else np.asarray(thermals, float)
        self.weather = Weather(self.z, cell, u10, wind_to, k_ridge, w_star, zi, th, WeatherParams())
        self.cfg = cfg
        self.core = FlightCore(
            drones, cfg.glider, ap or cfg.autopilot, cp or cfg.controller,
            self.surface_at, self.ground_at, self.L, base, cfg.team.ceiling_agl_m,
            self.weather.wind_at, self.mean_wind_at, np.random.default_rng(seed), zi=zi,
            mean_wind_agl=self.weather.mean_wind)
        self.t = 0.0

    def surface_at(self, x, y):
        return bilinear(self.surface, x, y, self.cell)

    def ground_at(self, x, y):
        return bilinear(self.z, x, y, self.cell)

    def mean_wind_at(self, x, y, z):
        return self.weather.mean_wind(np.asarray(z) - self.ground_at(x, y))

    def run(self, seconds, until=None):
        events = []
        for _ in range(int(seconds)):
            events += self.core.step(self.t)
            self.t += 1.0
            if until is not None and until(self):
                break
        return events


def u10_for(speed_at_h, h, z0=0.1):
    """10 m wind that gives `speed_at_h` at height h (log profile)."""
    return speed_at_h * math.log(10 / z0) / math.log(h / z0)
