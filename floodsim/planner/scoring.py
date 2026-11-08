"""Choosing a goal: block tables, travel and search time, energy, height choice (PLANNER_PLAN section 9)."""
from __future__ import annotations

import math

import numpy as np

from ..config import RHO
from .params import EXPOSED, SHELTERED

TAN30 = math.tan(math.radians(30))


def block_sum(a: np.ndarray, f: int) -> np.ndarray:
    """Sum f x f blocks over the last two axes."""
    n = a.shape[-1]
    return a.reshape(a.shape[:-2] + (n // f, f, n // f, f)).sum(axis=(-3, -1))


def log_factor(z_agl, z0=0.1, top=600.0):
    z = np.clip(z_agl, 2.0, top)
    return np.log(z / z0) / math.log(10.0 / z0)


def drag(gp, V):
    q = 0.5 * RHO * np.asarray(V, float) ** 2 * gp.wing_area
    cl = np.minimum(gp.weight / q, gp.cl_max)
    return q * (gp.cd0 + gp.k * cl * cl)


def level_power(gp, V):
    return drag(gp, V) * np.asarray(V, float) / gp.prop_eff + gp.payload_w


def search_speed(params, wind10, h):
    u = math.hypot(*wind10) * float(log_factor(h))
    return min(max(params.search_speed_min, u + params.search_speed_margin), params.search_speed_cap)


def search_time(params, h, V, block_m):
    spacing = 2 * h * TAN30 * (1 - params.overlap)
    n_legs = math.ceil(block_m / spacing)
    return block_m * block_m / (spacing * V) + params.leg_turn_s * n_legs, spacing


def travel(gx, gy, x, y, wind_t, ap):
    """Distance, time and cruise speed from (x, y) to block centres with the autopilot's transit speed rule."""
    dx, dy = gx - x, gy - y
    dist = np.hypot(dx, dy)
    chi = np.arctan2(dy, dx)
    along = wind_t[0] * np.cos(chi) + wind_t[1] * np.sin(chi)
    head = -along
    cruise = np.minimum(ap.transit_speed + 0.5 * np.maximum(head, 0.0), ap.transit_speed_cap)
    gs = np.maximum(2.0, cruise + along)
    return dist, dist / gs, cruise


def e_return(gp, ap, bx, by, x, y, wind_t):
    """Autopilot return-energy rule (Wh) from points (x, y): includes x1.3 and the reserve."""
    dx, dy = bx - x, by - y
    dist = np.hypot(dx, dy)
    chi = np.arctan2(dy, dx)
    head = -(wind_t[0] * np.cos(chi) + wind_t[1] * np.sin(chi))
    vc = np.minimum(ap.transit_speed + 0.5 * np.maximum(head, 0.0), ap.transit_speed_cap)
    gs = np.maximum(ap.rtb_min_gs, vc - head)
    return dist / gs * level_power(gp, vc) / 3600.0 * ap.rtb_energy_factor + gp.reserve_wh


def block_table(X: np.ndarray, V: np.ndarray, f: int) -> np.ndarray:
    """X [2 classes, H, ny, nx] expected detections per cell; V [2, L, ny, nx] value -> B [H, L, nb, nb]."""
    prod = X[:, :, None] * V[:, None]          # [2, H, L, ny, nx]
    return block_sum(prod.sum(axis=0), f)


def class_sums(Wk: np.ndarray, pd_h: np.ndarray) -> np.ndarray:
    """Wk [K, ny, nx] weighted intensity; pd_h [H, K, ny, nx] -> X [2, H, ny, nx]."""
    sh = (Wk[list(SHELTERED)][None] * pd_h[:, list(SHELTERED)]).sum(axis=1)
    ex = (Wk[list(EXPOSED)][None] * pd_h[:, list(EXPOSED)]).sum(axis=1)
    return np.stack([sh, ex])
