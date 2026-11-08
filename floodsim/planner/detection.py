"""State-aware detection model the planner uses (PLANNER_PLAN section 4).

The planner cannot know a person's state, so it averages the environment's range curve over a state mix
per behaviour component, driven by the forecast at the time of the look."""
from __future__ import annotations

import numpy as np

from ..sensor import pd_range
from .params import COMPONENTS

STATE_SIZE = {"home_inside": 0.5, "roof": 0.5, "vehicle": 0.5, "outdoor": 0.5, "in_water": 0.3}
STATE_FACTOR = {"home_inside": 0.02, "roof": 1.0, "vehicle": 0.2, "outdoor": 1.0, "in_water": 0.8}


def state_mix(component, p_deep):
    if component in ("home", "home_low"):
        return {"home_inside": 1 - p_deep, "roof": p_deep}
    if component == "road":
        return {"vehicle": np.ones_like(p_deep)}
    return {"outdoor": 1 - p_deep, "in_water": p_deep}


def pd_hat(z_cam, R, cover_mod, component, p_deep):
    """Reference implementation (Appendix B): expected detection probability for one component."""
    out = 0.0
    for s, w in state_mix(component, p_deep).items():
        out = out + w * pd_range(z_cam, STATE_SIZE[s], R) * STATE_FACTOR[s]
    return out * cover_mod


def p_deep_from_forecast(p_flood: np.ndarray, exp_depth: np.ndarray) -> np.ndarray:
    """P(depth >= 0.5 m) = P_flood x sigmoid((E[d | wet] - 0.5) / 0.15)."""
    ed_wet = exp_depth / np.maximum(p_flood, 0.05)
    return (p_flood / (1.0 + np.exp(-(ed_wet - 0.5) / 0.15))).astype(np.float32)


def pd_components(z_cam_bld, z_cam_gnd, R, cover, p_deep, state_aware: bool = True) -> np.ndarray:
    """Vectorised p_hat_d for all 6 components -> [K, ...].

    z_cam_bld: camera range to people at home / on roofs (above roof level);
    z_cam_gnd: camera range to people on the ground."""
    if not state_aware:   # appendix model: average target 0.4 m, m_state 0.8, same for every component
        p = pd_range(z_cam_gnd, 0.4, R) * 0.8 * cover
        return np.broadcast_to(p, (len(COMPONENTS),) + np.shape(p)).astype(np.float32)
    in_bld = pd_range(z_cam_bld, 0.5, R)
    land = pd_range(z_cam_gnd, 0.5, R)
    water = pd_range(z_cam_gnd, 0.3, R)
    home = ((1 - p_deep) * in_bld * 0.02 + p_deep * in_bld * 1.0) * cover
    road = land * 0.2 * cover
    outd = ((1 - p_deep) * land * 1.0 + p_deep * water * 0.8) * cover
    return np.stack([home, home, road, outd, outd, outd]).astype(np.float32)
