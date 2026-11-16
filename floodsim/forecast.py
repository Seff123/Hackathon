"""FloodForecast: the planner-visible, deliberately imperfect flood picture.

Built once at world build from truth, then held only as plain arrays (no
reference to FloodTimeline), so nothing here can leak ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.special import ndtr

from .config import Config
from .flood import hazard_rating
from .grid import cell_centres
from .terrain import FOREST, URBAN, WATER


@dataclass
class StaticMaps:
    """Planner-visible static layers (copies of terrain arrays, read-only)."""

    cell: float
    z: np.ndarray
    surface: np.ndarray
    landcover: np.ndarray
    road_mask: np.ndarray
    population: np.ndarray
    hand: np.ndarray
    depression_depth: np.ndarray
    drain_id: np.ndarray
    flow_acc: np.ndarray
    dist_to_river: np.ndarray
    base_xy: tuple

    @classmethod
    def from_terrain(cls, t) -> "StaticMaps":
        def ro(a):
            a = np.array(a, copy=True)
            a.setflags(write=False)
            return a

        return cls(cell=t.cell_size, z=ro(t.z), surface=ro(t.surface), landcover=ro(t.landcover),
                   road_mask=ro(t.road_mask), population=ro(t.population), hand=ro(t.hand),
                   depression_depth=ro(t.depression_depth), drain_id=ro(t.drain_id),
                   flow_acc=ro(t.flow_acc), dist_to_river=ro(t.dist_to_river), base_xy=tuple(t.base_xy))


@dataclass
class RainForecast:
    """Noisy storm model (radar-extrapolated): a few floats, no truth reference."""

    x_mid: float
    y_mid: float
    vx: float
    vy: float
    t_start: float  # episode time
    duration: float
    radius: float
    peak_ms: float

    def rate(self, X, Y, t):
        u = (t - self.t_start) / self.duration
        if self.peak_ms <= 0 or u < 0 or u > 1:
            return np.zeros(np.shape(X))
        env = math.sin(math.pi * u) ** 2
        dtm = t - (self.t_start + 0.5 * self.duration)
        cx, cy = self.x_mid + self.vx * dtm, self.y_mid + self.vy * dtm
        return self.peak_ms * env * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * self.radius**2))

    def cumulative(self, X, Y, t0, t1, n=24):
        if t1 <= t0 or self.peak_ms <= 0:
            return np.zeros(np.shape(X))
        ts = np.linspace(t0, t1, n + 1)
        mids = 0.5 * (ts[1:] + ts[:-1])
        out = np.zeros(np.shape(X))
        for tm in mids:
            out += self.rate(X, Y, tm) * (ts[1] - ts[0])
        return out


@dataclass
class FloodForecast:
    maps: StaticMaps
    drains: np.ndarray  # flat indices of drainage cells
    stage_fc: np.ndarray  # float16 [n_issue, n_lead, n_drains] forecast stage above drain bed
    obs_extent: np.ndarray  # packed bits [n_obs, ...]: noisy observed wet/dry
    obs_vel: np.ndarray  # float16 [n_vel, 2, nyc, nxc] noisy velocity
    sigma0: float
    growth_mph: float
    issue_dt: float
    lead_step: float
    obs_dt: float
    vel_dt: float
    observed_wet_floor: float
    rain: RainForecast | None
    runoff: np.ndarray
    pluvial_accum: float
    pluvial_min_dep: float
    vel_factor: int = 2
    wet_depth: float = 0.1
    _cache: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ helpers
    @property
    def shape(self):
        return self.maps.hand.shape

    def sigma(self, lead_s: float) -> float:
        return self.sigma0 + self.growth_mph * lead_s / 3600.0

    def _bucket(self, t_now, lead_s):
        i = int(np.clip(t_now // self.issue_dt, 0, self.stage_fc.shape[0] - 1))
        j = int(np.clip(round(lead_s / self.lead_step), 0, self.stage_fc.shape[1] - 1))
        return i, j

    def _obs_index(self, t_now):
        return int(np.clip(t_now // self.obs_dt, 0, self.obs_extent.shape[0] - 1))

    def observed_extent(self, t_now) -> np.ndarray:
        k = self._obs_index(t_now)
        n = self.shape[0] * self.shape[1]
        return np.unpackbits(self.obs_extent[k])[:n].reshape(self.shape).astype(bool)

    def _stage_cells(self, i, j):
        st = np.full(self.maps.hand.size + 1, -np.inf, np.float32)
        lookup = self._cache.get("lookup")
        if lookup is None:
            lookup = np.full(self.maps.hand.size, -1, np.int64)
            lookup[self.drains] = np.arange(len(self.drains))
            self._cache["lookup"] = lookup
        pos = lookup[self.maps.drain_id.ravel()]
        vals = self.stage_fc[i, j].astype(np.float32)
        s = np.where(pos >= 0, vals[np.maximum(pos, 0)], -np.inf)
        return s.reshape(self.shape)

    def _pluvial(self, t_now, lead_s):
        if self.rain is None or self.rain.peak_ms <= 0:
            return None
        X, Y = self._cache.setdefault("XY", cell_centres(self.shape[0], self.maps.cell))
        rain = self.rain.cumulative(X, Y, self.rain.t_start, t_now + lead_s)
        est = rain * self.runoff * np.minimum(self.maps.flow_acc, self.pluvial_accum)
        dep = np.maximum(self.maps.depression_depth, self.pluvial_min_dep)
        sig = 0.5 * self.sigma(lead_s)
        p = ndtr((est - dep - self.wet_depth) / sig)
        return np.where(self.maps.depression_depth > 0.05, p, 0.0), np.maximum(est - dep, 0.0)

    # ------------------------------------------------------------------ API
    def prob_flooded(self, t_now, lead_s=0.0) -> np.ndarray:
        i, j = self._bucket(t_now, lead_s)
        key = ("p", i, j, self._obs_index(t_now))
        if key in self._cache:
            return self._cache[key]
        sig = self.sigma(j * self.lead_step)
        s = self._stage_cells(i, j)
        a = (s - self.maps.hand - self.wet_depth) / sig
        p = ndtr(a)
        pl = self._pluvial(i * self.issue_dt, j * self.lead_step)
        if pl is not None:
            p = 1 - (1 - p) * (1 - pl[0])
        p = np.where(self.observed_extent(t_now), np.maximum(p, self.observed_wet_floor), p)
        p = np.where(self.maps.landcover == WATER, 1.0, p).astype(np.float32)
        self._remember(key, p)
        return p

    def expected_depth(self, t_now, lead_s=0.0) -> np.ndarray:
        i, j = self._bucket(t_now, lead_s)
        key = ("d", i, j)
        if key in self._cache:
            return self._cache[key]
        sig = self.sigma(j * self.lead_step)
        s = self._stage_cells(i, j)
        a = np.where(np.isfinite(s), (s - self.maps.hand) / sig, -50.0)
        d = np.maximum(sig * (np.exp(-0.5 * a * a) / math.sqrt(2 * math.pi) + a * ndtr(a)), 0.0)
        pl = self._pluvial(i * self.issue_dt, j * self.lead_step)
        if pl is not None:
            d = np.maximum(d, pl[1])
        d = d.astype(np.float32)
        self._remember(key, d)
        return d

    def observed_velocity(self, t_now):
        k = int(np.clip(t_now // self.vel_dt, 0, self.obs_vel.shape[0] - 1))
        key = ("v", k)
        if key not in self._cache:
            u = ndimage.zoom(self.obs_vel[k, 0].astype(np.float32), self.vel_factor, order=1)
            v = ndimage.zoom(self.obs_vel[k, 1].astype(np.float32), self.vel_factor, order=1)
            self._remember(key, (u, v))
        return self._cache[key]

    def expected_hazard(self, t_now, lead_s=0.0) -> np.ndarray:
        d = self.expected_depth(t_now, lead_s)
        u, v = self.observed_velocity(t_now)
        return hazard_rating(d, np.hypot(u, v)).astype(np.float32)

    def rain_estimate(self, t_now) -> np.ndarray:
        X, Y = self._cache.setdefault("XY", cell_centres(self.shape[0], self.maps.cell))
        if self.rain is None:
            return np.zeros(self.shape, np.float32)
        return self.rain.rate(X, Y, t_now).astype(np.float32)

    def _remember(self, key, val):
        if len(self._cache) > 200:
            keep = {k: self._cache[k] for k in ("lookup", "XY") if k in self._cache}
            self._cache.clear()
            self._cache.update(keep)
        self._cache[key] = val

    # ------------------------------------------------------------------ cache io
    def to_arrays(self) -> dict:
        r = self.rain
        return {
            "drains": self.drains, "stage_fc": self.stage_fc, "obs_extent": self.obs_extent,
            "obs_vel": self.obs_vel,
            "scalars": np.array([self.sigma0, self.growth_mph, self.issue_dt, self.lead_step, self.obs_dt,
                                 self.vel_dt, self.observed_wet_floor, self.pluvial_accum,
                                 self.pluvial_min_dep, self.vel_factor]),
            "rain": np.array([r.x_mid, r.y_mid, r.vx, r.vy, r.t_start, r.duration, r.radius, r.peak_ms])
            if r is not None else np.zeros(0),
        }

    @classmethod
    def from_arrays(cls, a: dict, terr, cfg: Config) -> "FloodForecast":
        sc = a["scalars"]
        rain = RainForecast(*map(float, a["rain"])) if a["rain"].size else None
        return cls(maps=StaticMaps.from_terrain(terr), drains=a["drains"], stage_fc=a["stage_fc"],
                   obs_extent=a["obs_extent"], obs_vel=a["obs_vel"], sigma0=float(sc[0]),
                   growth_mph=float(sc[1]), issue_dt=float(sc[2]), lead_step=float(sc[3]),
                   obs_dt=float(sc[4]), vel_dt=float(sc[5]), observed_wet_floor=float(sc[6]),
                   rain=rain, runoff=_runoff_map(terr, cfg), pluvial_accum=float(sc[7]),
                   pluvial_min_dep=float(sc[8]), vel_factor=int(sc[9]), wet_depth=cfg.world.wet_depth_m)


def _runoff_map(terr, cfg):
    rc = cfg.world.runoff_coef
    lc = terr.landcover
    return np.select([lc == URBAN, lc == FOREST, lc == WATER], [rc["urban"], rc["forest"], rc["water"]],
                     rc["open"]).astype(np.float32)


def smooth_unit_field(rng, n, cell, corr_len):
    """Gaussian random field with unit variance and corr(corr_len) = 1/e."""
    w = rng.standard_normal((n, n))
    sig = corr_len / (2.0 * cell)  # kernel sigma in cells: corr(d) = exp(-d^2 / 4 sigma^2)
    f = ndimage.gaussian_filter(w, sig, mode="wrap")
    return f / max(f.std(), 1e-12)


def build_forecast(world) -> "FloodForecast":
    from .rng import stream

    spec, cfg, terr, fl = world.spec, world.cfg, world.terrain, world.flood
    fp = cfg.forecast
    rng = stream(spec.seed, "forecast", spec.attempt)
    n = terr.n
    ep = world.episode_s
    drains = np.flatnonzero(terr.drainage_mask)
    zf = terr.z_fill.ravel()[drains].astype(np.float32)
    z = terr.z.ravel()[drains].astype(np.float32)
    n_issue = int(ep // fp.issue_dt_s) + 1
    n_lead = int(round(fp.max_lead_s / fp.lead_step_s)) + 1
    # true stage above the drainage cell's filled bed at every 5 min to episode end + max lead
    times = np.arange(0.0, ep + fp.max_lead_s + 1e-6, fp.issue_dt_s)
    true_stage = np.zeros((len(times), len(drains)), np.float32)
    for k, t in enumerate(times):
        d = fl.depth_grid(min(t, fl.t_end)).ravel()[drains]
        true_stage[k] = np.where(d > 0, d + z - zf, fp.dry_stage_m)
    err = smooth_unit_field(rng, n, terr.cell_size, fp.corr_length_m).ravel()[drains].astype(np.float32)
    bias = spec.fc_bias_m
    stage_fc = np.zeros((n_issue, n_lead, len(drains)), np.float16)
    step = int(round(fp.lead_step_s / fp.issue_dt_s))
    for i in range(n_issue):
        for j in range(n_lead):
            L = j * fp.lead_step_s
            sig = spec.fc_sigma0_m + spec.fc_growth_mph * L / 3600.0
            ts = true_stage[min(i + j * step, len(times) - 1)]
            stage_fc[i, j] = ts + bias + sig * err
    # observed extent every 10 min: wet/dry correct with p_obs
    n_obs = int(ep // fp.obs_dt_s) + 1
    obs = []
    for k in range(n_obs):
        wet = fl.depth_grid(k * fp.obs_dt_s) > cfg.world.wet_depth_m
        flip = rng.random(wet.shape) > fp.p_obs_correct
        obs.append(np.packbits((wet ^ flip).ravel()))
    obs_extent = np.stack(obs)
    # observed velocity every 10 min (coarse grid), x (1 +/- noise)
    n_vel = int(ep // fp.velocity_dt_s) + 1
    vel = np.zeros((n_vel, 2) + fl.z_coarse.shape, np.float16)
    for k in range(n_vel):
        u, v = fl.coarse_velocity_grid(k * fp.velocity_dt_s)
        fac = 1.0 + rng.uniform(-fp.velocity_noise, fp.velocity_noise, u.shape)
        vel[k, 0] = u * fac
        vel[k, 1] = v * fac
    rain = None
    st = world.storm
    if spec.has_rain and st.peak_ms > 0:
        t_ep = spec.start_offset_h * 3600.0
        e = rng.normal(0, 1, 4)
        rain = RainForecast(st.x_mid + 500 * e[0], st.y_mid + 500 * e[1], st.vx * (1 + 0.1 * e[2]),
                            st.vy * (1 + 0.1 * e[2]), st.t_start - t_ep, st.duration, st.radius,
                            st.peak_ms * float(np.clip(1 + fp.rain_noise * e[3], 0.5, 1.5)))
    return FloodForecast(
        maps=StaticMaps.from_terrain(terr), drains=drains, stage_fc=stage_fc, obs_extent=obs_extent,
        obs_vel=vel, sigma0=spec.fc_sigma0_m, growth_mph=spec.fc_growth_mph, issue_dt=fp.issue_dt_s,
        lead_step=fp.lead_step_s, obs_dt=fp.obs_dt_s, vel_dt=fp.velocity_dt_s,
        observed_wet_floor=fp.observed_wet_floor, rain=rain, runoff=_runoff_map(terr, cfg),
        pluvial_accum=fp.pluvial_accum_cells, pluvial_min_dep=fp.pluvial_min_depression_m,
        vel_factor=fl.fine_factor,
    )


def brier_score(world, lead_s: float, zone_hand_m: float | None = None) -> float:
    """Brier score of prob_flooded against truth over land cells with HAND < zone."""
    fc, fl, terr = world.forecast, world.flood, world.terrain
    zone_hand_m = zone_hand_m or world.cfg.forecast.floodplain_hand_m
    zone = terr.land & (terr.hand < zone_hand_m)
    errs = []
    for t in np.arange(0.0, world.episode_s - lead_s + 1e-6, fc.issue_dt):
        p = fc.prob_flooded(t, lead_s)
        truth = fl.depth_grid(t + lead_s) > world.cfg.world.wet_depth_m
        errs.append(np.mean((p[zone] - truth[zone]) ** 2))
    return float(np.mean(errs))
