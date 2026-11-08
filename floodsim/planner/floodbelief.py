"""Learning the forecast's scenario-wide stage bias from the observed extent (PLANNER_PLAN section 7)."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


class BiasBelief:
    def __init__(self, params, seed: int = 0):
        self.p = params
        self.b = np.asarray(params.bias_grid, dtype=float)
        self.logp = -0.5 * (self.b / params.bias_prior_sd) ** 2
        self.rng = np.random.default_rng(seed)
        self.last_t = -1.0
        self.history = []

    def post(self) -> np.ndarray:
        q = np.exp(self.logp - self.logp.max())
        return q / q.sum()

    def mean(self) -> float:
        return float(self.post() @ self.b) if self.p.bias_learning else 0.0

    def update(self, forecast, t: float, wet_depth: float = 0.1) -> bool:
        """Call when a new observed extent is available at time t (every 10 min)."""
        if not self.p.bias_learning or t <= self.last_t + 1e-6:
            return False
        self.last_t = t
        obs = forecast.observed_extent(t).reshape(-1)
        used = False
        for L in self.p.bias_leads_min:
            Ls = L * 60.0
            if t - Ls < 0:
                continue
            mu, sig, f = forecast.fluvial_terms(t - Ls, Ls)
            mu, sig, f = mu.reshape(-1), sig.reshape(-1), f.reshape(-1)
            p0 = ndtr((mu - wet_depth) / sig)
            cand = np.flatnonzero((p0 > 0.05) & (p0 < 0.95) & (f > 0))
            if len(cand) == 0:
                continue
            if len(cand) > self.p.bias_max_cells:
                cand = self.rng.choice(cand, self.p.bias_max_cells, replace=False)
            m, s, ff, o = mu[cand], sig[cand], f[cand], obs[cand]
            pb = ndtr((m[None] + self.b[:, None] * ff[None] - wet_depth) / s[None])
            pobs = np.clip(0.95 * pb + 0.05 * (1 - pb), 1e-6, 1 - 1e-6)
            ll = (o[None] * np.log(pobs) + (1 - o[None]) * np.log(1 - pobs)).sum(axis=1)
            self.logp += self.p.bias_tau * ll
            used = True
        if used:
            self.logp -= self.logp.max()
            self.history.append((t, self.mean()))
        return used
