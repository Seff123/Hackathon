"""FloodForecast: the planner-visible, deliberately imperfect flood picture.

Everything is precomputed at world build as plain arrays (observed extents, stage forecasts at drainage
cells, noisy velocities, rain accumulations). FloodForecast never holds a reference to the FloodTimeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr

from .config import Config, ScenarioSpec
from .grid import upsample2
from .terrain import FOREST, URBAN, Terrain

SQ2PI = 1.0 / math.sqrt(2 * math.pi)


def _phi(x):
    return SQ2PI * np.exp(-0.5 * x * x)


@dataclass
class FloodForecast:
    # static maps (planner-visible)
    hand: np.ndarray
    drain_k: np.ndarray            # [n, n] index into drainage list (-1: none)
    drain_scale: np.ndarray        # [ND] forecast-error scale f(d) (1 on the main river)
    depression_depth: np.ndarray
    pond_gain: np.ndarray          # runoff coeff x min(upstream cells, cap)
    land: np.ndarray
    # precomputed noisy products
    stage_fc: np.ndarray           # [n_issue, n_lead, ND] float16 forecast stage (m above drainage bed)
    obs_extent: np.ndarray         # [n_obs, n, n] bool
    obs_vel: np.ndarray            # [n_vobs, 2, n/2, n/2] float16 (100 m)
    rain_fc: np.ndarray            # [n_rain, n/2, n/2] float16 forecast accumulated rain (m), every issue step
    # scalars
    sigma0: float
    growth_per_s: float
    issue_dt: float
    lead_dt: float
    n_lead: int
    extent_dt: float
    vel_dt: float
    wet_depth: float
    sigma_floor: float
    pluvial_sigma: float
    observed_wet_p: float
    has_rain: bool
    brier: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ helpers
    def sigma(self, lead_s: float) -> float:
        """Stage forecast error sigma(L) = sigma0 + growth x L (main river)."""
        return float(self.sigma0 + self.growth_per_s * max(lead_s, 0.0))

    def _ij(self, t_now: float, lead_s: float):
        i = int(np.clip(np.floor(t_now / self.issue_dt + 1e-9), 0, self.stage_fc.shape[0] - 1))
        j = int(np.clip(round(lead_s / self.lead_dt), 0, self.n_lead - 1))
        return i, j

    def _get(self, key, fn):
        if key not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[key] = fn()
        return self._cache[key]

    def _fluvial_mu_sigma(self, i: int, j: int, stage_offset: float):
        s = self.stage_fc[i, j].astype(np.float32) + np.float32(stage_offset) * self.drain_scale
        lead = j * self.lead_dt
        sig_d = np.maximum(self.sigma(lead) * self.drain_scale, self.sigma_floor)
        k = np.maximum(self.drain_k, 0)
        mu = np.where(self.drain_k >= 0, s[k] - self.hand, -99.0).astype(np.float32)
        sig = sig_d[k].astype(np.float32)
        return mu, sig

    def _pond(self, i: int, j: int):
        if not self.has_rain:
            return None
        r = upsample2(self.rain_fc[min(i + j, self.rain_fc.shape[0] - 1)].astype(np.float32))
        pond = np.minimum(np.maximum(r, 0) * self.pond_gain, self.depression_depth)
        return np.where(self.depression_depth > 0.05, pond, 0.0).astype(np.float32)

    # ------------------------------------------------------------------ API
    def observed_extent(self, t_now: float) -> np.ndarray:
        k = int(np.clip(np.floor(t_now / self.extent_dt + 1e-9), 0, self.obs_extent.shape[0] - 1))
        return self.obs_extent[k]

    def observed_velocity(self, t_now: float):
        k = int(np.clip(np.floor(t_now / self.vel_dt + 1e-9), 0, self.obs_vel.shape[0] - 1))

        def f():
            v = self.obs_vel[k].astype(np.float32)
            return upsample2(v[0]), upsample2(v[1])
        return self._get(("vel", k), f)

    def prob_flooded(self, t_now: float, lead_s: float, stage_offset: float = 0.0) -> np.ndarray:
        i, j = self._ij(t_now, lead_s)
        ko = int(np.floor(t_now / self.extent_dt + 1e-9))

        def f():
            mu, sig = self._fluvial_mu_sigma(i, j, stage_offset)
            p = ndtr((mu - self.wet_depth) / sig)
            pond = self._pond(i, j)
            if pond is not None:
                sp = self.pluvial_sigma + 0.3 * pond
                pp = np.where(self.depression_depth > 0.05, ndtr((pond - self.wet_depth) / sp), 0.0)
                p = 1.0 - (1.0 - p) * (1.0 - pp)
            p = np.where(self.observed_extent(t_now), np.maximum(p, self.observed_wet_p), p)
            return np.where(self.land, p, 1.0).astype(np.float32)
        return self._get(("p", i, j, ko, round(stage_offset, 4)), f)

    def expected_depth(self, t_now: float, lead_s: float, stage_offset: float = 0.0) -> np.ndarray:
        i, j = self._ij(t_now, lead_s)

        def f():
            mu, sig = self._fluvial_mu_sigma(i, j, stage_offset)
            z = mu / sig
            e = mu * ndtr(z) + sig * _phi(z)
            pond = self._pond(i, j)
            if pond is not None:
                e = np.maximum(e, pond)
            return np.maximum(e, 0).astype(np.float32)
        return self._get(("d", i, j, round(stage_offset, 4)), f)

    def expected_hazard(self, t_now: float, lead_s: float, stage_offset: float = 0.0) -> np.ndarray:
        i, j = self._ij(t_now, lead_s)

        def f():
            mu, sig = self._fluvial_mu_sigma(i, j, stage_offset)
            ed = self.expected_depth(t_now, lead_s, stage_offset)
            u, v = self.observed_velocity(t_now)
            sp = np.hypot(u, v)
            p25 = ndtr((mu - 0.25) / sig)
            p75 = ndtr((mu - 0.75) / sig)
            pond = self._pond(i, j)
            if pond is not None:
                p25 = np.maximum(p25, (pond > 0.25).astype(np.float32))
                p75 = np.maximum(p75, (pond > 0.75).astype(np.float32))
            return (ed * (sp + 0.5) + 0.5 * p25 + 0.5 * p75).astype(np.float32)
        return self._get(("hr", i, j, round(stage_offset, 4)), f)


def _ar1_profile(terrain: Terrain, didx: np.ndarray, rng, corr_m: float) -> np.ndarray:
    """Unit-variance AR(1) error along the drainage network (correlation length corr_m)."""
    n = terrain.n
    pos = -np.ones(n * n, dtype=np.int64)
    pos[didx] = np.arange(len(didx))
    down = terrain.flow_dir.reshape(-1)
    zf = terrain.z_fill.reshape(-1)
    order = didx[np.argsort(zf[didx], kind="stable")]       # downstream first
    e = np.zeros(len(didx))
    xi = rng.standard_normal(len(didx))
    c = terrain.cell_size
    for cidx in order.tolist():
        k = pos[cidx]
        d = down[cidx]
        if d >= 0 and pos[d] >= 0:
            dy, dx = divmod(cidx, n)[0] - divmod(d, n)[0], cidx % n - d % n
            step = c * math.hypot(dy, dx)
            rho = math.exp(-step / corr_m)
            e[k] = rho * e[pos[d]] + math.sqrt(1 - rho * rho) * xi[k]
        else:
            e[k] = xi[k]
    return e


def build_forecast(spec: ScenarioSpec, terrain: Terrain, flood, cfg: Config, rng, cache_dir=None) -> FloodForecast:
    fc = cfg.forecast
    wc = cfg.world
    n = terrain.n
    didx = np.flatnonzero(terrain.drainage_mask.reshape(-1))
    nd = len(didx)
    kmap = -np.ones(n * n, dtype=np.int64)
    kmap[didx] = np.arange(nd)
    drain = terrain.drain_id.reshape(-1)
    drain_k = np.where(drain >= 0, kmap[np.maximum(drain, 0)], -1).reshape(n, n)
    area_km2 = terrain.flow_acc.reshape(-1)[didx] * terrain.cell_size ** 2 / 1e6
    is_river = terrain.river_mask.reshape(-1)[didx]
    scale = np.where(is_river, 1.0, np.minimum(1.0, np.sqrt(area_km2 / fc.trib_ref_area_km2))).astype(np.float32)
    e = _ar1_profile(terrain, didx, rng, fc.ar_corr_m)
    bias = spec.fc_bias_m
    s0, g = spec.fc_sigma0_m, spec.fc_growth_mph / 3600.0

    n_issue = int(round(wc.episode_s / fc.issue_every_s)) + 1
    n_lead = int(round(fc.max_lead_s / fc.lead_step_s)) + 1
    n_times = n_issue + n_lead - 1
    stage_true = np.zeros((n_times, nd), dtype=np.float32)
    for k in range(n_times):
        stage_true[k] = flood.depth_grid50(k * fc.issue_every_s).reshape(-1)[didx]
    leads = np.arange(n_lead) * fc.lead_step_s
    sig = s0 + g * leads
    stage_fc = np.zeros((n_issue, n_lead, nd), dtype=np.float16)
    err = scale[None, :] * (bias + sig[:, None] * e[None, :])           # [n_lead, nd]
    for i in range(n_issue):
        stage_fc[i] = np.maximum(stage_true[i:i + n_lead] + err, 0.0).astype(np.float16)

    # observed extent every 10 min (each cell correct with p = 0.95)
    n_obs = int(round(wc.episode_s / fc.extent_refresh_s)) + 1
    obs = np.zeros((n_obs, n, n), dtype=bool)
    for k in range(n_obs):
        wet = flood.depth_grid50(k * fc.extent_refresh_s) > cfg.flood.wet_depth
        flip = rng.uniform(size=(n, n)) > fc.extent_correct_p
        obs[k] = wet ^ flip
    # noisy velocity every 10 min (100 m)
    n_v = int(round(wc.episode_s / fc.velocity_refresh_s)) + 1
    vel = np.zeros((n_v, 2, n // 2, n // 2), dtype=np.float16)
    for k in range(n_v):
        u, v = flood.velocity_grid(k * fc.velocity_refresh_s)
        f = np.clip(1.0 + fc.velocity_noise * rng.standard_normal(u.shape), 0.0, 2.0)
        vel[k, 0] = (u * f).astype(np.float16)
        vel[k, 1] = (v * f).astype(np.float16)
    # rain forecast: accumulated storm rain x (1 + scenario error), every issue step out to the last lead
    from .flood import make_storm
    from . import rng as rngmod
    storm = make_storm(spec, cfg, rngmod.stream(spec.seed, spec.attempt, "flood"))
    has_rain = storm.active
    if has_rain:
        rerr = 1.0 + fc.rain_noise * float(np.clip(rng.standard_normal(), -2.5, 2.5))
        xs = (np.arange(n // 2) + 0.5) * cfg.world.flood_cell_m
        X, Y = np.meshgrid(xs, xs)
        rain = np.zeros((n_times, n // 2, n // 2), dtype=np.float16)
        for k in range(n_times):
            rain[k] = (storm.accumulated(X, Y, k * fc.issue_every_s, n_sub=12) * rerr).astype(np.float16)
    else:
        rain = np.zeros((1, n // 2, n // 2), dtype=np.float16)
    lc = terrain.landcover
    fp = cfg.flood
    cr = np.where(lc == URBAN, fp.runoff_urban, np.where(lc == FOREST, fp.runoff_forest, fp.runoff_open))
    pond_gain = (cr * np.minimum(terrain.flow_acc, fc.pluvial_area_cap)).astype(np.float32)
    ff = FloodForecast(
        hand=terrain.hand.astype(np.float32), drain_k=drain_k, drain_scale=scale,
        depression_depth=terrain.depression_depth.astype(np.float32), pond_gain=pond_gain,
        land=~terrain.river_mask, stage_fc=stage_fc, obs_extent=obs, obs_vel=vel, rain_fc=rain,
        sigma0=s0, growth_per_s=g, issue_dt=fc.issue_every_s, lead_dt=fc.lead_step_s, n_lead=n_lead,
        extent_dt=fc.extent_refresh_s, vel_dt=fc.velocity_refresh_s, wet_depth=cfg.flood.wet_depth,
        sigma_floor=fc.sigma_floor_m, pluvial_sigma=fc.pluvial_sigma_m, observed_wet_p=fc.observed_wet_p,
        has_rain=has_rain)
    ff.brier = forecast_brier(ff, flood, terrain, wc.episode_s, cfg.flood.wet_depth)
    return ff


def forecast_brier(ff: FloodForecast, flood, terrain: Terrain, episode_s: float, wet_depth: float,
                   leads_min=(15, 30, 60)) -> dict:
    """Brier score of prob_flooded against truth at several leads, over land cells and issue times."""
    land = ~terrain.river_mask
    out = {}
    for L in leads_min:
        errs = []
        for t in np.arange(0.0, episode_s + 1, 1800.0):
            p = ff.prob_flooded(t, L * 60.0)
            truth = flood.depth_grid50(t + L * 60.0) > wet_depth
            errs.append(float(((p - truth) ** 2)[land].mean()))
        out[int(L)] = float(np.mean(errs))
    ff._cache.clear()
    return out
