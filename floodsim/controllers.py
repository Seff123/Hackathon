"""Controller interface and the scripted sweep that exercises the environment.

Controllers only ever see PlannerView objects (never the world)."""
from __future__ import annotations

import math

import numpy as np

from .config import ControllerParams
from .goals import RTB, Goal, Search, Transit, Verify  # noqa: F401  (re-exported)


def wind_at_height(u10: float, v10: float, z_agl: float, z0: float = 0.1, top: float = 600.0):
    """Log-law scaling of a 10 m wind vector to height z_agl (public physics, planner-visible)."""
    z = min(max(z_agl, 2.0), top)
    f = math.log(z / z0) / math.log(10.0 / z0)
    return u10 * f, v10 * f


class Controller:
    name = "base"

    def reset(self, views: dict) -> dict:
        return {}

    def decide(self, views_due: dict) -> dict:
        return {}

    def observe(self, views_all: dict) -> None:
        """Called every sensor tick with a view for every drone (planner hook)."""
        return None

    def messages_out(self) -> list:
        """(sender, payload, recipient|None) tuples to hand to the radio."""
        return []

    def params_dict(self) -> dict:
        p = getattr(self, "params", None)
        if p is None:
            return {}
        from .config import to_jsonable
        return to_jsonable(p)


def search_height(cp: ControllerParams, camera_range: float) -> float:
    return max(cp.search_height_min, cp.search_height_frac * camera_range)


def search_speed(cp: ControllerParams, wind10: tuple, agl: float) -> float:
    u, v = wind_at_height(wind10[0], wind10[1], agl)
    return min(max(cp.search_speed_min, math.hypot(u, v) + cp.search_speed_margin), cp.search_speed_cap)


def _clip_xy(p, domain, margin):
    return (min(max(p[0], margin), domain - margin), min(max(p[1], margin), domain - margin))


def strip_patterns(mask: np.ndarray, cell: float, n_strips: int, leg_angle: float, spacing: float,
                   domain: float | None = None, margin: float = 100.0):
    """Split mask cells into n equal-area strips along the leg-normal axis; boustrophedon legs per strip.

    Returns (list of waypoint lists, strip id per mask cell [same order as np.nonzero(mask)])."""
    iy, ix = np.nonzero(mask)
    domain = domain if domain is not None else mask.shape[0] * cell
    x = (ix + 0.5) * cell
    y = (iy + 0.5) * cell
    ca, sa = math.cos(leg_angle), math.sin(leg_angle)
    d = x * ca + y * sa           # along-leg coordinate
    nn = -x * sa + y * ca         # across-leg coordinate
    if len(x) == 0:
        return [[] for _ in range(n_strips)], np.zeros(0, int)
    order = np.argsort(nn, kind="stable")
    sid = np.empty(len(x), dtype=int)
    bounds = np.linspace(0, len(x), n_strips + 1).round().astype(int)
    for k in range(n_strips):
        sid[order[bounds[k]:bounds[k + 1]]] = k
    patterns = []
    for k in range(n_strips):
        sel = sid == k
        if not sel.any():
            patterns.append([])
            continue
        nk, dk = nn[sel], d[sel]
        lines = np.arange(nk.min() + spacing / 2, nk.max() + spacing / 2 + 1e-6, spacing)
        wps = []
        flip = False
        for nl in lines:
            on = np.abs(nk - nl) <= max(spacing / 2, cell / 2)
            if not on.any():
                continue
            d0, d1 = dk[on].min() - cell / 2, dk[on].max() + cell / 2
            a = _clip_xy((d0 * ca - nl * sa, d0 * sa + nl * ca), domain, margin)
            b = _clip_xy((d1 * ca - nl * sa, d1 * sa + nl * ca), domain, margin)
            wps += [b, a] if flip else [a, b]
            flip = not flip
        patterns.append(wps)
    return patterns, sid


class ScriptedSweep(Controller):
    """Lawnmower over the floodplain (HAND < 10 m): one equal-area strip per drone, crosswind legs.
    Verifies every detection, soars opportunistically, relies on the autopilot's energy rule for RTB."""
    name = "scripted"

    def __init__(self, params: ControllerParams | None = None):
        self.params = params or ControllerParams()
        self.patterns = {}
        self.progress = {}
        self.verify_queue = {}
        self.verified = {}
        self.current = {}

    def reset(self, views: dict) -> dict:
        cp = self.params
        v0 = next(iter(views.values()))
        maps = v0.maps
        n = len(views)
        mask = (maps.hand < cp.floodplain_hand_m) & (maps.landcover != 1)
        wind_dir = math.atan2(v0.wind_est[1], v0.wind_est[0])
        leg = wind_dir + math.radians(cp.leg_angle_deg)
        agl = search_height(cp, v0.camera_range_m)
        spacing = 2 * agl * math.tan(math.radians(30)) * (1 - cp.overlap)
        pats, _ = strip_patterns(mask, maps.cell, n, leg, spacing)
        self.agl = agl
        self.spacing = spacing
        goals = {}
        for i, v in views.items():
            wps = pats[i] if pats[i] else [v.base_xy]
            # start at the strip end nearest base
            bx, by = v.base_xy
            if math.hypot(wps[-1][0] - bx, wps[-1][1] - by) < math.hypot(wps[0][0] - bx, wps[0][1] - by):
                wps = wps[::-1]
            self.patterns[i] = wps
            self.progress[i] = 0
            self.verify_queue[i] = []
            self.verified[i] = {}
            goals[i] = self._search_goal(i, v)
        return goals

    def _search_goal(self, i, v) -> Search:
        cp = self.params
        sp = search_speed(cp, v.wind_est, self.agl)
        g = Search(waypoints=self.patterns[i], agl=self.agl, spacing=self.spacing, speed=sp,
                   start_index=self.progress[i], allow_soaring=True, tag="strip")
        self.current[i] = "search"
        return g

    def decide(self, views_due: dict) -> dict:
        goals = {}
        for i, v in views_due.items():
            st = v.own
            if st.goal_kind == "Search" and st.wpi is not None:
                self.progress[i] = st.wpi
            if "search_done" in v.reasons:
                self.progress[i] = 0
            for det in v.new_detections:
                cell = det[1]
                last = self.verified[i].get(cell, -1e9)
                if v.t - last > 120.0 and cell not in self.verify_queue[i]:
                    self.verify_queue[i].append(cell)
            if not st.airborne:
                continue
            if self.current.get(i) == "verify" and "verify_done" not in v.reasons:
                continue
            if self.verify_queue[i] and self.params.verify_all:
                cell = self.verify_queue[i].pop(0)
                self.verified[i][cell] = v.t
                goals[i] = Verify(cell=cell)
                self.current[i] = "verify"
            elif self.current.get(i) != "search" or "search_done" in v.reasons or "relaunch" in v.reasons:
                goals[i] = self._search_goal(i, v)
        return goals
