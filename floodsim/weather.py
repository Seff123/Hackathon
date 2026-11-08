"""Weather (truth): constant mean wind with a log profile, ridge lift, fixed thermal hotspots, storm rain."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

from .config import Config, ScenarioSpec
from .grid import nearest
from .terrain import FOREST, OPEN, URBAN, WATER, Terrain

G_PEAK_S = 0.25 / 1.1   # argmax of s^(1/3)(1 - 1.1 s)
G_NORM = G_PEAK_S ** (1 / 3) * (1 - 1.1 * G_PEAK_S)


def thermal_shape(s):
    """Vertical profile g(s) proportional to s^(1/3)(1 - 1.1 s), normalised so g_max = 1."""
    s = np.asarray(s, dtype=float)
    ok = (s > 0) & (s < 1 / 1.1)
    return np.where(ok, np.cbrt(np.clip(s, 0, None)) * (1 - 1.1 * s) / G_NORM, 0.0)


@dataclass
class Thermal:
    x: float
    y: float
    w_core: float
    r_t: float


class Weather:
    def __init__(self, spec: ScenarioSpec, terrain: Terrain, cfg: Config, rng: np.random.Generator):
        wp = cfg.weather
        self.p = wp
        self.u10 = float(spec.wind10_ms)
        self.theta = spec.wind_to_rad
        self.cx, self.cy = math.cos(self.theta), math.sin(self.theta)
        self.zi = float(spec.zi_m)
        self.wstar = float(spec.wstar_ms)
        self.k_r = float(spec.ridge_k)
        self.cell = terrain.cell_size
        self.terrain_z = terrain.z.astype(np.float64)
        zs = gaussian_filter(self.terrain_z, wp.ridge_smooth_m / self.cell)
        gy, gx = np.gradient(zs, self.cell)
        self.gx, self.gy = gx, gy
        size = int(2 * wp.ridge_relief_radius_m / self.cell) + 1
        relief = maximum_filter(self.terrain_z, size=size) - minimum_filter(self.terrain_z, size=size)
        self.H_r = np.maximum(0.8 * relief, 50.0)
        self.log_ref = math.log(10.0 / wp.z0)
        # thermals: density x area hotspots weighted by land cover, fixed for the episode
        area_km2 = (cfg.world.domain_m / 1000.0) ** 2
        n_th = int(round(spec.thermal_density_km2 * area_km2))
        cw = wp.thermal_cover_weight
        wmap = np.select([terrain.landcover == URBAN, terrain.landcover == OPEN, terrain.landcover == FOREST,
                          terrain.landcover == WATER], [cw["urban"], cw["open"], cw["forest"], cw["water"]], 0.0)
        self.thermal_list: list[Thermal] = []
        if n_th > 0 and self.wstar > 0 and wmap.sum() > 0:
            idx = rng.choice(wmap.size, size=n_th, p=(wmap / wmap.sum()).reshape(-1))
            n = terrain.n
            for k in idx:
                iy, ix = divmod(int(k), n)
                rt = float(np.clip(rng.uniform(*wp.thermal_radius_frac) * self.zi, *wp.thermal_radius_clamp))
                self.thermal_list.append(Thermal(x=(ix + rng.uniform()) * self.cell, y=(iy + rng.uniform()) * self.cell,
                                                 w_core=float(rng.uniform(*wp.thermal_core) * self.wstar), r_t=rt))
        self._tx = np.array([t.x for t in self.thermal_list])
        self._ty = np.array([t.y for t in self.thermal_list])
        self._tw = np.array([t.w_core for t in self.thermal_list])
        self._tr = np.array([t.r_t for t in self.thermal_list])
        from .flood import make_storm
        from . import rng as rngmod
        self.storm = make_storm(spec, cfg, rngmod.stream(spec.seed, spec.attempt, "flood"))

    # ------------------------------------------------------------------ profile
    def speed_at(self, z_agl):
        z = np.clip(np.asarray(z_agl, dtype=float), 2.0, self.p.profile_top_m)
        return self.u10 * np.log(z / self.p.z0) / self.log_ref

    def mean_wind(self, z_agl):
        s = self.speed_at(z_agl)
        return s * self.cx, s * self.cy

    # ------------------------------------------------------------------ point queries
    def wind(self, x, y, z, t: float = 0.0):
        """(u, v, w) at world points; arrays over drones. z is altitude (m above datum)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        zg = nearest(self.terrain_z, x, y, self.cell)
        z_agl = np.asarray(z, dtype=float) - zg
        s = self.speed_at(z_agl)
        u, v = s * self.cx, s * self.cy
        w = self.ridge(x, y, z_agl, u, v) + self.thermal_w(x, y, z_agl)
        return u, v, w

    wind_at = wind

    def ridge(self, x, y, z_agl, u, v):
        gx = nearest(self.gx, x, y, self.cell)
        gy = nearest(self.gy, x, y, self.cell)
        hr = nearest(self.H_r, x, y, self.cell)
        w = self.k_r * (u * gx + v * gy) * np.exp(-np.maximum(z_agl, 0) / hr)
        return np.maximum(w, self.p.ridge_min_sink)

    def thermal_w(self, x, y, z_agl):
        if len(self._tx) == 0:
            return np.zeros(np.shape(x))
        dx = np.asarray(x)[..., None] - self._tx
        dy = np.asarray(y)[..., None] - self._ty
        r2 = dx * dx + dy * dy
        prof = thermal_shape(np.asarray(z_agl) / self.zi)
        return (self._tw * np.exp(-r2 / self._tr ** 2)).sum(-1) * prof

    def rain(self, x, y, t: float):
        return self.storm.rain(x, y, t)

    def thermals(self) -> list[Thermal]:
        return list(self.thermal_list)

    # ------------------------------------------------------------------ grids for plots
    def wind_grid(self, z_agl: float, t: float = 0.0, cell: float | None = None):
        n = self.terrain_z.shape[0]
        xs = (np.arange(n) + 0.5) * self.cell
        X, Y = np.meshgrid(xs, xs)
        Z = self.terrain_z + z_agl
        return self.wind(X, Y, Z, t)
