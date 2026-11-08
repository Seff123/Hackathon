"""Belief: one Poisson intensity grid per behaviour component, looks, finds and prediction (section 5).

lam[k] = expected number of *unfound* people in each cell if everyone followed component k (each component
starts at total N_hat). Every update is linear per cell and identical across components, so the scenario
posterior only needs per-component accumulators (section 6)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .params import COMPONENTS, PlannerParams
from .scenarios import posterior, scenario_grid


def component_maps(maps, forecast, t0: float, edge_lead_s: float = 3600.0) -> np.ndarray:
    """Normalised component maps [K, n, n] from planner-visible layers only."""
    bd = maps.building_density.astype(np.float64)
    hand = maps.hand.astype(np.float64)
    land = maps.landcover != 1
    road = maps.road_mask.astype(np.float64)
    home = bd
    home_low = bd * (hand < 5.0)
    roads = road * (1.0 + 2.0 * (hand < 3.0))
    open_ = ((maps.landcover == 0) & (maps.dist_to_river <= 2000.0)).astype(np.float64)
    pf = forecast.prob_flooded(t0, edge_lead_s)
    wet = (pf > 0.5) & land
    if wet.any():
        dist = distance_transform_edt(~wet, sampling=maps.cell)
        high = ((hand >= 3.0) & (dist <= 1000.0) & land & ~wet).astype(np.float64)
    else:
        high = (hand >= 3.0).astype(np.float64) * land
    anywhere = land.astype(np.float64)
    comps = np.stack([home, home_low, roads, open_, high, anywhere])
    comps = comps * land[None]
    sums = comps.sum(axis=(1, 2), keepdims=True)
    # an empty component (e.g. no low homes) falls back to "anywhere" so the grid stays a valid density
    comps = np.where(sums > 0, comps, anywhere[None])
    comps = comps + 1e-9 * land[None]
    return comps / comps.sum(axis=(1, 2), keepdims=True)


class ComponentBelief:
    def __init__(self, comps: np.ndarray, n_hat: float, params: PlannerParams):
        self.p = params
        self.lam = (n_hat * comps).astype(np.float64)                       # [K, n, n]
        K = self.lam.shape[0]
        self.K = K
        self.W, self.S, self.log_prior = scenario_grid(params.prior_mean, params.scenario_step, params.scenario_scales,
                                                       params.scenario_concentration, params.scale_sigma)
        self.A = np.zeros(K)
        self.det_ll = np.zeros(len(self.S))
        self.n_hat = n_hat
        self.lost_mass = np.zeros(K)
        self.mob = self.p.mobile_fraction * self.lam[3].copy()            # mobile part of the open component
        n = self.lam.shape[1]
        self.pend_pd = np.zeros((K, n * n), dtype=np.float32)
        self.pend_open = np.full(n * n, np.inf)
        self.n_looks = 0
        self._post_cache = None

    # ------------------------------------------------------------------ scenario posterior
    def post(self) -> np.ndarray:
        if self._post_cache is None:
            if self.p.scenarios:
                self._post_cache = posterior(self.log_prior, self.W, self.S, self.A, self.det_ll)
            else:
                self._post_cache = np.exp(self.log_prior - self.log_prior.max())
                self._post_cache /= self._post_cache.sum()
        return self._post_cache

    def w_eff(self) -> np.ndarray:
        """Effective component weights: posterior mean of s_j w_jk (fixed prior mean if scenarios are off)."""
        if not self.p.scenarios:
            return np.asarray(self.p.prior_mean, dtype=float)
        return (self.post() * self.S) @ self.W

    def count_scale(self) -> float:
        return float(self.post() @ self.S)

    def intensity(self) -> np.ndarray:
        return np.tensordot(self.w_eff(), self.lam, axes=1)

    def _changed(self):
        self._post_cache = None

    # ------------------------------------------------------------------ looks
    def queue_look(self, cells: np.ndarray, pd_k: np.ndarray, t: float) -> None:
        """Revisit window: a cell gets at most one update per min_revisit_s; keep the best p_hat_d in it."""
        op = self.pend_open[cells]
        new = ~np.isfinite(op)
        self.pend_open[cells[new]] = t
        self.pend_pd[:, cells] = np.maximum(self.pend_pd[:, cells], pd_k)

    def flush(self, t: float, force: bool = False) -> int:
        due = np.flatnonzero(self.pend_open <= (t + 1e9 if force else t - self.p.min_revisit_s))
        if len(due) == 0:
            return 0
        self.look_flat(due, self.pend_pd[:, due].astype(np.float64))
        self.pend_pd[:, due] = 0.0
        self.pend_open[due] = np.inf
        return len(due)

    def look_flat(self, cells: np.ndarray, pd_k: np.ndarray) -> None:
        lam = self.lam.reshape(self.K, -1)
        lam_c = lam[:, cells]
        self.A += (pd_k * lam_c).sum(axis=1)
        lam[:, cells] = lam_c * (1.0 - pd_k)
        mob = self.mob.reshape(-1)
        mob[cells] *= (1.0 - pd_k[3])
        self.n_looks += len(cells)
        self._changed()

    def look(self, iy, ix, pd_k):
        n = self.lam.shape[1]
        self.look_flat(np.asarray(iy) * n + np.asarray(ix), np.asarray(pd_k, dtype=np.float64))

    def lam_at(self, cell: int) -> np.ndarray:
        return self.lam.reshape(self.K, -1)[:, cell].copy()

    # ------------------------------------------------------------------ finds
    def confirmed_find(self, lam_at_look_k: np.ndarray, pd_k_cell: np.ndarray) -> None:
        """The person was already removed by the look's (1 - pd) update: no '-1' here."""
        lam_j = self.S * (self.W @ (pd_k_cell * lam_at_look_k))
        self.det_ll += np.log(np.maximum(lam_j, 1e-12))
        self._changed()

    def rejected(self, cell: int, pd_verify: float) -> None:
        self.look_flat(np.array([cell]), np.full((self.K, 1), pd_verify))

    def p_real(self, lam_k: np.ndarray, pd_k: np.ndarray, kappa: float) -> float:
        m = float(self.w_eff() @ (pd_k * lam_k))
        return m / (kappa + m)

    # ------------------------------------------------------------------ prediction (open component only)
    def predict(self, dt: float, forecast, t: float, maps) -> float:
        """Mobile people of the open component walk uphill (along grad HAND) where P_flood > 0.5, spread by
        diffusion, and ride fast deep water. Forward (push) advection conserves mass; what leaves the domain
        goes to lost_mass. Mass where P_flood <= 0.5 does not move. Returns mass lost this step."""
        p = self.p
        c = maps.cell
        pf = forecast.prob_flooded(t, 0.0)
        active = (pf > 0.5) & (self.mob > 0)
        if not active.any():
            return 0.0
        gy, gx = np.gradient(maps.hand.astype(np.float64), c)
        g = np.hypot(gx, gy)
        sp = p.walk_speed * dt / c
        ux = np.where(g > 1e-6, gx / np.maximum(g, 1e-9), 0.0) * sp
        uy = np.where(g > 1e-6, gy / np.maximum(g, 1e-9), 0.0) * sp
        ed = forecast.expected_depth(t, 0.0)
        vu, vv = forecast.observed_velocity(t)
        fast = (ed > p.water_drift_depth) & (np.hypot(vu, vv) > p.water_drift_speed)
        mov = np.where(active, self.mob, 0.0)
        stay = self.mob - mov
        out, lost = _push(mov, ux, uy, fast, vu * p.drift_fraction * dt / c, vv * p.drift_fraction * dt / c,
                          min(1.0, p.water_drift_rate_per_min * dt / 60.0))
        sigma = 0.5 * p.walk_speed * dt / c
        diffused = gaussian_filter(out, sigma, mode="constant")
        lost += out.sum() - diffused.sum()
        new_mob = stay + diffused
        self.lost_mass[3] += lost
        self.lam[3] = np.maximum(self.lam[3] + new_mob - self.mob, 0.0)
        self.mob = new_mob
        self._changed()
        return float(lost)


def _push(mass, ux, uy, fast, wu, wv, frac_fast):
    """Forward bilinear splatting of `mass` by displacement (ux, uy) cells; in `fast` cells a fraction
    frac_fast of the mass instead moves with (wu, wv). Returns (new grid, mass that left the domain)."""
    n = mass.shape[0]
    out = np.zeros_like(mass)
    lost = 0.0
    iy, ix = np.nonzero(mass > 0)
    if len(iy) == 0:
        return out, 0.0
    m = mass[iy, ix]
    parts = []
    fm = fast[iy, ix]
    walk_m = np.where(fm, m * (1 - frac_fast), m)
    parts.append((walk_m, ux[iy, ix], uy[iy, ix]))
    if fm.any():
        parts.append((np.where(fm, m * frac_fast, 0.0), wu[iy, ix], wv[iy, ix]))
    for mm, dx, dy in parts:
        fx, fy = ix + dx, iy + dy
        x0, y0 = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
        ax, ay = fx - x0, fy - y0
        for oy, ox, w in ((0, 0, (1 - ax) * (1 - ay)), (0, 1, ax * (1 - ay)), (1, 0, (1 - ax) * ay), (1, 1, ax * ay)):
            yy, xx = y0 + oy, x0 + ox
            ok = (yy >= 0) & (yy < n) & (xx >= 0) & (xx < n)
            np.add.at(out, (yy[ok], xx[ok]), (mm * w)[ok])
            lost += float((mm * w)[~ok].sum())
    return out, lost
