"""Reference maths for the FloodSim planner (prototype, verified on synthetic data).

Grids are [ny, nx]. Component grids are [K, ny, nx]. Times in seconds.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import norm

# ----------------------------------------------------------------------------- detection model
STATES = ("home_inside", "roof", "vehicle", "outdoor", "in_water")
STATE_SIZE = {"home_inside": 0.5, "roof": 0.5, "vehicle": 0.5, "outdoor": 0.5, "in_water": 0.3}
STATE_FACTOR = {"home_inside": 0.02, "roof": 1.0, "vehicle": 0.2, "outdoor": 1.0, "in_water": 0.8}


def pd_range(z_cam, size, R):
    """Environment's Johnson-style curve (section 11), before modifiers."""
    r = (size / 0.5) * (R / np.maximum(z_cam, 1.0))
    E = 2.7 + 0.7 * r
    return 0.95 * r ** E / (1.0 + r ** E)


def state_mix(component, p_deep):
    """Planner's guess of the state mix of people from one component in a cell.
    p_deep = P(depth >= 0.5 m). Returns dict state -> prob (sums to 1)."""
    if component in ("home", "home_low"):
        return {"home_inside": 1 - p_deep, "roof": p_deep}
    if component == "road":
        return {"vehicle": np.ones_like(p_deep)}
    # outdoor-like components (open land, high ground, uniform)
    return {"outdoor": 1 - p_deep, "in_water": p_deep}


def pd_hat(z_cam, R, cover_mod, component, p_deep):
    """Expected detection probability for a person from `component`, averaged over the state mix."""
    out = 0.0
    for s, w in state_mix(component, p_deep).items():
        out = out + w * pd_range(z_cam, STATE_SIZE[s], R) * STATE_FACTOR[s]
    return out * cover_mod


# ----------------------------------------------------------------------------- belief
class ComponentBelief:
    """Intensity grids per behaviour component + posterior over mixture scenarios.

    lam[k] = expected number of *unfound* people in each cell if every person followed component k
    (each starts with total N_hat). A scenario j = (weights w_j over components, count scale s_j).
    Scenario intensity = s_j * sum_k w_jk lam[k]. All updates are linear per cell, so the scenarios
    never need their own grids: the Poisson log-likelihood only needs per-component sums.
    """

    def __init__(self, comps: np.ndarray, n_hat: float, scen_w: np.ndarray, scen_s: np.ndarray,
                 log_prior: np.ndarray):
        self.lam = n_hat * comps / comps.sum(axis=(1, 2), keepdims=True)      # [K, ny, nx]
        self.W = scen_w                                                         # [J, K]
        self.s = scen_s                                                         # [J]
        self.log_prior = log_prior                                              # [J]
        self.A = np.zeros(self.lam.shape[0])      # cumulative sum_looks sum_cells pd*lam_k
        self.det_ll = np.zeros(len(scen_s))       # sum over confirmed finds of log(pd * lam_j)

    # -- scenario posterior
    def log_post(self):
        return self.log_prior - self.s * (self.W @ self.A) + self.det_ll

    def post(self):
        lp = self.log_post()
        p = np.exp(lp - lp.max())
        return p / p.sum()

    def w_eff(self):
        """Effective component weights (posterior mean of s_j * w_jk)."""
        return (self.post() * self.s) @ self.W

    def intensity(self):
        return np.tensordot(self.w_eff(), self.lam, axes=1)

    # -- measurement updates
    def look(self, iy, ix, pd_k):
        """One look at cells (iy, ix) with per-component detection prob pd_k [K, n] (no detection)."""
        lam_c = self.lam[:, iy, ix]
        self.A += (pd_k * lam_c).sum(axis=1)
        self.lam[:, iy, ix] = lam_c * (1.0 - pd_k)

    def confirmed_find(self, lam_at_look_k, pd_k_cell):
        """Call when verification confirms a person detected in a look. lam_at_look_k = lam[:, c] just
        before that look (store it at detection time); pd_k_cell = pd per component for that look.
        The person was already removed from lam by the look's (1 - pd) update (Poisson thinning)."""
        lam_j = self.s * (self.W @ (pd_k_cell * lam_at_look_k))
        self.det_ll += np.log(np.maximum(lam_j, 1e-12))

    def rejected(self, iy, ix, pd_verify_k):
        """Verification orbit found nobody: it is just another (very good) look."""
        self.look(np.atleast_1d(iy), np.atleast_1d(ix), pd_verify_k[:, None])


def scenario_grid(prior_mean, step=0.25, scales=(0.5, 0.7, 1.0, 1.4, 2.0), conc=6.0, scale_sigma=0.3):
    """All weight vectors on a simplex grid x count scales, with a Dirichlet x lognormal prior."""
    K = len(prior_mean)
    n = int(round(1 / step))
    ws = [np.array(c) / n for c in itertools.product(range(n + 1), repeat=K) if sum(c) == n]
    W, S, LP = [], [], []
    alpha = conc * np.asarray(prior_mean) + 1e-3
    for w in ws:
        for s in scales:
            wc = np.clip(w, 1e-3, None)
            lp = ((alpha - 1) * np.log(wc)).sum() - (np.log(s) ** 2) / (2 * scale_sigma ** 2)
            W.append(w); S.append(s); LP.append(lp)
    LP = np.array(LP)
    return np.array(W), np.array(S), LP - LP.max()


# ----------------------------------------------------------------------------- urgency value
def danger_rate(p_hr_low, p_hr_high, comp_state_exposed):
    """Expected fraction of each second that counts as danger time for a person in the cell.
    p_hr_low = P(HR > 0.75), p_hr_high = P(HR > 2); comp_state_exposed = P(exposed) for the cell's mix."""
    return comp_state_exposed * p_hr_low + (1 - comp_state_exposed) * p_hr_high


def survival_value(rate_leads, lead_s, t_now, t_end_plus2h, d_past, t_crit=2400.0, tail_factor=1.0):
    """Marginal survival gained by confirming a person in each cell at each lead time.

    rate_leads: [L, ...] danger rate (0-1) at forecast leads lead_s (seconds, starting at 0).
    Returns V[L, ...] = S(at lead) - S(if never found), and cumulative danger D[L, ...].
    Beyond the last lead the rate is held at tail_factor * last rate until t_end_plus2h.
    """
    dt = np.diff(lead_s).reshape((-1,) + (1,) * (rate_leads.ndim - 1))
    D = np.concatenate([np.zeros_like(rate_leads[:1]),
                        np.cumsum(0.5 * (rate_leads[1:] + rate_leads[:-1]) * dt, axis=0)])
    tail_s = max(t_end_plus2h - (t_now + lead_s[-1]), 0.0)
    D_end = D[-1] + tail_factor * rate_leads[-1] * tail_s
    return np.exp(-(d_past + D) / t_crit) - np.exp(-(d_past + D_end) / t_crit), D
