"""Survivor likelihood: scenario grid, prior and posterior helpers (PLANNER_PLAN section 6, Appendix B)."""
from __future__ import annotations

import itertools

import numpy as np


def scenario_grid(prior_mean, step=0.25, scales=(0.5, 0.7, 1.0, 1.4, 2.0), conc=6.0, scale_sigma=0.3):
    """All weight vectors on a simplex grid x count scales, with a Dirichlet x lognormal prior.
    Returns W [J, K], S [J], log prior [J] (max 0)."""
    K = len(prior_mean)
    n = int(round(1 / step))
    ws = [np.array(c) / n for c in itertools.product(range(n + 1), repeat=K) if sum(c) == n]
    alpha = conc * np.asarray(prior_mean) + 1e-3
    W, S, LP = [], [], []
    for w in ws:
        for s in scales:
            lp = ((alpha - 1) * np.log(np.clip(w, 1e-3, None))).sum() - np.log(s) ** 2 / (2 * scale_sigma ** 2)
            W.append(w)
            S.append(s)
            LP.append(lp)
    LP = np.array(LP)
    return np.array(W), np.array(S, dtype=float), LP - LP.max()


def posterior(log_prior, W, S, A, det_ll):
    lp = log_prior - S * (W @ A) + det_ll
    p = np.exp(lp - lp.max())
    return p / p.sum()


def entropy_norm(p: np.ndarray) -> float:
    q = p[p > 0]
    return float(-(q * np.log(q)).sum() / np.log(len(p)))
