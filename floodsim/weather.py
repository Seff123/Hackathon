"""Weather truth: constant mean wind with a log profile, ridge lift, fixed thermals."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from .config import Config, ScenarioSpec, WeatherParams
from .grid import bilinear, cell_centres
from .terrain import FOREST, OPEN, URBAN, WATER

# normalisation for g(s) = s^(1/3) (1 - 1.1 s): maximum at s = 1/4.4
_S_PEAK = 1.0 / 4.4
_G_MAX = _S_PEAK ** (1 / 3) * (1 - 1.1 * _S_PEAK)


def thermal_profile(s):
    """Vertical shape g(z/zi), peak 1, zero at the ground and above 1/1.1 zi."""
    s = np.asarray(s, float)
    ok = (s > 0) & (s < 1 / 1.1)
    sc = np.where(ok, s, 0.5)
    return np.where(ok, np.cbrt(sc) * (1 - 1.1 * sc) / _G_MAX, 0.0)


class Weather:
    """Truth-side wind field. Controllers never see this object."""

    def __init__(self, z: np.ndarray, cell: float, u10: float, wind_to_rad: float, k_ridge: float,
                 w_star: float, zi: float, thermals: np.ndarray, wp: WeatherParams, storm=None,
                 t_offset: float = 0.0):
        self.cell = float(cell)
        self.z = np.asarray(z, np.float32)
        self.u10 = float(u10)
        self.theta = float(wind_to_rad)
        self.k_ridge = float(k_ridge)
        self.w_star = float(w_star)
        self.zi = float(zi)
        self.wp = wp
        self.thermals = np.asarray(thermals, np.float64).reshape(-1, 4)  # x, y, w_core, r_t
        self.storm = storm
        self.t_offset = t_offset  # episode time -> storm time (seconds since onset)
        self._ln10 = math.log(10.0 / wp.z0_m)
        sig = wp.ridge_smooth_m / self.cell
        zs = ndimage.gaussian_filter(self.z.astype(np.float64), sig, mode="nearest")
        gy, gx = np.gradient(zs, self.cell)
        self.gx = gx.astype(np.float32)
        self.gy = gy.astype(np.float32)
        win = max(int(round(wp.relief_window_m / self.cell)) | 1, 3)
        relief = ndimage.maximum_filter(self.z, win) - ndimage.minimum_filter(self.z, win)
        self.h_ridge = np.maximum(wp.ridge_h_factor * relief, wp.ridge_h_min_m).astype(np.float32)
        self.ex, self.ey = math.cos(self.theta), math.sin(self.theta)
        self._stack = np.stack([self.z, self.gx, self.gy, self.h_ridge]).astype(np.float64)

    # ---------------------------------------------------------------- profile
    def speed(self, z_agl):
        z = np.minimum(np.maximum(np.asarray(z_agl, float), 2.0), self.wp.profile_top_m)
        return self.u10 * np.log(z / self.wp.z0_m) / self._ln10

    def mean_wind(self, z_agl):
        s = self.speed(z_agl)
        return s * self.ex, s * self.ey

    # ---------------------------------------------------------------- components
    def ridge(self, x, y, z_agl):
        gx = bilinear(self.gx, x, y, self.cell)
        gy = bilinear(self.gy, x, y, self.cell)
        hr = bilinear(self.h_ridge, x, y, self.cell)
        s = self.speed(z_agl)
        w = self.k_ridge * s * (self.ex * gx + self.ey * gy) * np.exp(-np.maximum(z_agl, 0.0) / hr)
        return np.maximum(w, self.wp.ridge_sink_clip)

    def thermal(self, x, y, z_agl):
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        z_agl = np.broadcast_to(np.asarray(z_agl, float), x.shape)
        if len(self.thermals) == 0 or self.w_star <= 0:
            return np.zeros(x.shape)
        th = self.thermals
        r2 = (x[:, None] - th[None, :, 0]) ** 2 + (y[:, None] - th[None, :, 1]) ** 2
        rad = np.exp(-r2 / th[None, :, 3] ** 2)
        return thermal_profile(z_agl / self.zi) * (rad * th[None, :, 2]).sum(axis=1)

    def wind_at(self, x, y, z):
        """(u, v, w) at world positions (arrays), z in metres above datum."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        zg, gx, gy, hr = bilinear(self._stack, x, y, self.cell)
        z_agl = np.asarray(z, float) - zg
        sp = self.speed(z_agl)
        wr = self.k_ridge * sp * (self.ex * gx + self.ey * gy) * np.exp(-np.maximum(z_agl, 0.0) / hr)
        w = np.maximum(wr, self.wp.ridge_sink_clip)
        if len(self.thermals) and self.w_star > 0:
            w = w + self.thermal(x, y, z_agl).reshape(np.shape(x))
        return sp * self.ex, sp * self.ey, w

    def wind(self, x, y, z, t=0.0):
        u, v, w = self.wind_at(np.array([x]), np.array([y]), np.array([z]))
        return float(u[0]), float(v[0]), float(w[0])

    def wind_grid(self, z_agl, t=0.0):
        n = self.z.shape[0]
        X, Y = cell_centres(n, self.cell)
        u, v = self.mean_wind(np.full(X.shape, z_agl))
        w = self.ridge(X, Y, np.full(X.shape, z_agl)) + self.thermal(X.ravel(), Y.ravel(), z_agl).reshape(X.shape)
        return u, v, w

    def rain(self, x, y, t):
        if self.storm is None:
            return np.zeros(np.shape(x))
        return self.storm.rate(np.asarray(x, float), np.asarray(y, float), t + self.t_offset)

    def thermal_list(self):
        """Debug only."""
        return self.thermals.copy()

    # ---------------------------------------------------------------- cache io
    def to_arrays(self):
        return {"thermals": self.thermals}

    @classmethod
    def from_arrays(cls, a, terr, spec: ScenarioSpec, cfg: Config, storm=None):
        return cls(terr.z, terr.cell_size, spec.wind10_ms, spec.wind_to_rad, spec.k_ridge, spec.w_star_ms,
                   spec.zi_m, a["thermals"], cfg.weather, storm=storm, t_offset=spec.start_offset_h * 3600.0)


def place_thermals(spec: ScenarioSpec, terr, wp: WeatherParams, rng) -> np.ndarray:
    area_km2 = (terr.domain_m / 1000.0) ** 2
    n = int(round(spec.thermal_density_km2 * area_km2))
    if n == 0 or spec.w_star_ms <= 0:
        return np.zeros((0, 4))
    lw = wp.thermal_landcover_weight
    w = np.select([terr.landcover == URBAN, terr.landcover == FOREST, terr.landcover == WATER],
                  [lw["urban"], lw["forest"], lw["water"]], lw["open"]).ravel().astype(float)
    idx = rng.choice(w.size, size=n, p=w / w.sum())
    iy, ix = np.divmod(idx, terr.n)
    x = (ix + rng.uniform(0, 1, n)) * terr.cell_size
    y = (iy + rng.uniform(0, 1, n)) * terr.cell_size
    core = rng.uniform(*wp.thermal_core_factor, n) * spec.w_star_ms
    r = np.clip(rng.uniform(*wp.thermal_radius_frac, n) * spec.zi_m, *wp.thermal_radius_clamp)
    return np.stack([x, y, core, r], axis=1)


def build_weather(spec: ScenarioSpec, terr, cfg: Config, rng, storm=None) -> Weather:
    th = place_thermals(spec, terr, cfg.weather, rng)
    return Weather(terr.z, terr.cell_size, spec.wind10_ms, spec.wind_to_rad, spec.k_ridge, spec.w_star_ms,
                   spec.zi_m, th, cfg.weather, storm=storm, t_offset=spec.start_offset_h * 3600.0)
