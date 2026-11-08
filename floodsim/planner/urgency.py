"""Urgency as marginal survival (PLANNER_PLAN section 8, Appendix B)."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def survival_value(rate_leads, lead_s, t_now, t_end_plus2h, d_past, t_crit=2400.0, tail_factor=1.0):
    """Marginal survival gained by confirming a person in each cell at each lead time.

    rate_leads: [L, ...] danger rate (0-1) at forecast leads lead_s (seconds, starting at 0).
    Returns V[L, ...] = S(at lead) - S(if never found), and cumulative danger D[L, ...]."""
    lead_s = np.asarray(lead_s, dtype=float)
    dt = np.diff(lead_s).reshape((-1,) + (1,) * (rate_leads.ndim - 1))
    D = np.concatenate([np.zeros_like(rate_leads[:1]),
                        np.cumsum(0.5 * (rate_leads[1:] + rate_leads[:-1]) * dt, axis=0)])
    tail_s = max(t_end_plus2h - (t_now + lead_s[-1]), 0.0)
    D_end = D[-1] + tail_factor * rate_leads[-1] * tail_s
    return np.exp(-(d_past + D) / t_crit) - np.exp(-(d_past + D_end) / t_crit), D


def danger_rate(p_hr_low, p_hr_high, p_exposed):
    return p_exposed * p_hr_low + (1 - p_exposed) * p_hr_high


def p_hr_exceeds(p_flood, e_hr, theta, sigma_hr):
    """P(HR > theta) ~ P_flood * Phi((E[HR | wet] - theta) / sigma_HR)."""
    e_wet = e_hr / np.maximum(p_flood, 0.05)
    return p_flood * ndtr((e_wet - theta) / sigma_hr)


def coarsen(a: np.ndarray, f: int = 2) -> np.ndarray:
    n = a.shape[-1]
    return a.reshape(a.shape[:-2] + (n // f, f, n // f, f)).mean(axis=(-3, -1))


class UrgencyTables:
    """Per-class value tables V[class, lead, cell] at 100 m, refreshed every 5-min forecast bucket.
    Class 0 = sheltered (home, home_low, road; HR > 2), class 1 = exposed (open, high_ground, anywhere; HR > 0.75)."""

    def __init__(self, params, horizon_s: float, lead_dt: float = 300.0, max_lead: float = 3600.0):
        self.p = params
        self.horizon = horizon_s
        self.leads = np.arange(0.0, max_lead + 1e-6, lead_dt)
        self.d_past = None
        self.t_last = None
        self.V = None
        self.rate0 = None
        self.t_built = None

    def rates(self, forecast, t, stage_offset):
        """Danger rate per class at every lead, 100 m: [2, L, 100, 100].
        P(HR > theta) is evaluated at 50 m then block-averaged."""
        p = self.p
        stride = 2
        pf, hr = forecast.products_all_leads(t, stage_offset, lead_stride=stride)
        sh = coarsen(p_hr_exceeds(pf, hr, 2.0, p.sigma_hr))
        ex = coarsen(p_hr_exceeds(pf, hr, 0.75, p.sigma_hr))
        r = np.stack([sh, ex]).astype(np.float32)               # [2, L/2+1, ...] at 0, 10, ..., 60 min
        full = np.empty((2, len(self.leads)) + r.shape[2:], np.float32)
        full[:, ::stride] = r
        full[:, 1::stride] = 0.5 * (r[:, :-1] + r[:, 1:])      # 5-min steps interpolated
        return full

    def update(self, forecast, t: float, stage_offset: float = 0.0):
        rates = self.rates(forecast, t, stage_offset)
        if self.d_past is None:
            self.d_past = np.zeros_like(rates[:, 0])
        elif self.t_last is not None and self.rate0 is not None:
            self.d_past += self.rate0 * (t - self.t_last)
        self.t_last = t
        self.rate0 = rates[:, 0].copy()
        self.last_rates = rates
        V = np.empty_like(rates)
        for c in range(2):
            V[c], _ = survival_value(rates[c], self.leads, t, self.horizon, self.d_past[c], self.p.t_crit_hat_s,
                                     self.p.tail_factor)
        self.V = np.maximum(V, 0.0) + self.p.eps_welfare
        self.t_built = t
        return self.V


def heuristic_U(forecast, t, params, stage_offset=0.0):
    """Appendix baseline: U = 1 + lambda_f P_flood(t+L) + lambda_h E[HR](t+L)/2 (100 m)."""
    pf = forecast.prob_flooded(t, params.lead_L_s, stage_offset)
    hr = forecast.expected_hazard(t, params.lead_L_s, stage_offset)
    return coarsen(1.0 + params.lambda_f * pf + params.lambda_h * hr / 2.0).astype(np.float32)
