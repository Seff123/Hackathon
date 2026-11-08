"""Controller interface and the scripted sweep that exercises the environment.

Controllers only ever receive PlannerViews (see sim.py), never the world.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import ControllerParams
from .goals import RTB, Goal, Search, Verify


class Controller:
    name = "base"

    def reset(self, views: dict) -> dict[int, Goal]:
        raise NotImplementedError

    def decide(self, views_due: dict) -> dict[int, Goal]:
        raise NotImplementedError

    def params(self) -> dict:
        return {}


def partition_floodplain(mask: np.ndarray, n_strips: int, theta: float, cell: float):
    """Split mask cells into n strips of equal cell count, cut across the wind axis.

    theta: direction (rad) of the strip-stacking axis (the wind direction, so the
    crosswind legs run along each strip). Returns labels (-1 outside the mask) and
    the u-coordinate cut points.
    """
    iy, ix = np.nonzero(mask)
    x = (ix + 0.5) * cell
    y = (iy + 0.5) * cell
    u = x * math.cos(theta) + y * math.sin(theta)
    order = np.argsort(u, kind="stable")
    labels = np.full(mask.shape, -1, np.int64)
    k = np.zeros(len(u), np.int64)
    bounds = np.linspace(0, len(u), n_strips + 1).round().astype(int)
    for s in range(n_strips):
        k[order[bounds[s]:bounds[s + 1]]] = s
    labels[iy, ix] = k
    cuts = [float(u[order[b]]) if b < len(u) else float(u.max()) for b in bounds[:-1]] + [float(u.max())]
    return labels, cuts


def strip_legs(labels: np.ndarray, strip: int, theta: float, cell: float, spacing: float,
               gap_m: float = 300.0) -> np.ndarray:
    """Boustrophedon legs (crosswind) covering one strip's cells."""
    iy, ix = np.nonzero(labels == strip)
    if len(iy) == 0:
        return np.zeros((0, 2, 2))
    x = (ix + 0.5) * cell
    y = (iy + 0.5) * cell
    ca, sa = math.cos(theta), math.sin(theta)
    u = x * ca + y * sa  # along-wind (row) coordinate
    w = -x * sa + y * ca  # crosswind (leg) coordinate
    u0, u1 = u.min() - cell / 2, u.max() + cell / 2
    n_rows = max(int(math.ceil((u1 - u0) / spacing)), 1)
    legs = []
    flip = False
    for r in range(n_rows):
        uc = u0 + (r + 0.5) * (u1 - u0) / n_rows
        sel = np.abs(u - uc) <= 0.5 * (u1 - u0) / n_rows + cell / 2
        if not sel.any():
            continue
        ws = np.sort(w[sel])
        breaks = np.nonzero(np.diff(ws) > gap_m)[0]
        segs = np.split(ws, breaks + 1)
        segs = [(sg[0] - cell / 2, sg[-1] + cell / 2) for sg in segs]
        if flip:
            segs = [(b, a) for a, b in segs[::-1]]
        for a, b in segs:
            pa = (uc * ca - a * sa, uc * sa + a * ca)
            pb = (uc * ca - b * sa, uc * sa + b * ca)
            legs.append((pa, pb))
        flip = not flip
    return np.asarray(legs, float)


class ScriptedSweep(Controller):
    """Equal-area crosswind strips over the floodplain, one per drone; verifies
    every detection; soars opportunistically below 60% battery."""

    name = "scripted"

    def __init__(self, cp: ControllerParams | None = None, z0: float = 0.1):
        self.cp = cp or ControllerParams()
        self.z0 = z0
        self.legs: dict[int, np.ndarray] = {}
        self.leg_idx: dict[int, int] = {}
        self.queue: dict[int, list[int]] = {}
        self.verifying: dict[int, int | None] = {}
        self.done: dict[int, bool] = {}
        self.claimed: set[int] = set()
        self.agl = 0.0
        self.speed = 0.0

    def params(self) -> dict:
        return dict(vars(self.cp))

    def _wind_at(self, view, h):
        u, v = view.wind10_est
        f = math.log(max(h, 2.0) / self.z0) / math.log(10.0 / self.z0)
        return u * f, v * f

    def reset(self, views: dict) -> dict[int, Goal]:
        cp = self.cp
        v0 = next(iter(views.values()))
        maps = v0.maps
        n = len(views)
        R = v0.team.camera_range_m
        self.agl = max(cp.sweep_agl_frac * R, cp.sweep_min_agl_m)
        uw, vw = self._wind_at(v0, self.agl)
        wspd = math.hypot(uw, vw)
        self.speed = min(max(cp.sweep_speed_min, wspd + cp.sweep_wind_margin), cp.sweep_speed_cap)
        wind_dir = math.atan2(vw, uw) if wspd > 0.1 else 0.0
        # legs run at leg_angle from the wind: rows stack along the direction normal to the legs
        leg_dir = wind_dir + math.radians(cp.leg_angle_deg)
        theta = leg_dir - math.pi / 2
        mask = (maps.hand < cp.floodplain_hand_m) & (maps.landcover != 0)
        spacing = 2 * self.agl * math.tan(math.radians(30)) * (1 - cp.overlap)
        labels, _ = partition_floodplain(mask, n, theta, maps.cell)
        self.labels = labels
        ids = sorted(views)
        goals = {}
        for k, i in enumerate(ids):
            legs = strip_legs(labels, k, theta, maps.cell, spacing)
            # start at the end of the strip nearest the base
            if len(legs):
                bx, by = v0.base_xy
                d0 = math.hypot(legs[0][0][0] - bx, legs[0][0][1] - by)
                d1 = math.hypot(legs[-1][1][0] - bx, legs[-1][1][1] - by)
                if d1 < d0:
                    legs = legs[::-1, ::-1]
            if len(legs):  # keep legs clear of the geofence margin
                m = cp.edge_margin_m
                legs = np.clip(legs, m, maps.hand.shape[0] * maps.cell - m)
            self.legs[i] = legs
            self.leg_idx[i] = 0
            self.queue[i] = []
            self.verifying[i] = None
            self.done[i] = len(legs) == 0
            goals[i] = self._search_goal(i)
        return goals

    def _search_goal(self, i):
        legs = self.legs[i]
        if len(legs) == 0:
            return None
        if self.leg_idx[i] >= len(legs):
            self.leg_idx[i] = 0  # pattern complete: start another pass
        return Search(legs=legs, agl=self.agl, speed=self.speed, start_leg=self.leg_idx[i],
                      allow_soaring=True)

    def decide(self, views_due: dict) -> dict[int, Goal]:
        goals = {}
        for i, v in views_due.items():
            st = v.status
            if st.get("mode") == "SEARCH":
                self.leg_idx[i] = max(self.leg_idx[i], int(st.get("leg_idx", 0)))
            if self.cp.verify_all:
                for (_, cell) in v.new_detections:
                    if cell not in self.claimed:
                        self.claimed.add(cell)
                        self.queue[i].append(cell)
            for (_, cell, _ok) in v.new_verifications:
                if self.verifying.get(i) == cell:
                    self.verifying[i] = None
            events = {e for e, _ in v.events}
            if "pattern_done" in events:
                self.leg_idx[i] = len(self.legs[i])
            if not v.own.airborne:
                if "landed" in events or not v.own.alive:
                    self.verifying[i] = None
                if v.own.alive and ("relaunch" in events):
                    goals[i] = self._search_goal(i) or RTB()
                continue
            if st.get("mode") in ("RTB", "LAUNCH"):
                continue
            if self.verifying.get(i) is None and self.queue[i]:
                cell = self.queue[i].pop(0)
                n = v.maps.hand.shape[1]
                cy, cx = divmod(cell, n)
                self.verifying[i] = cell
                goals[i] = Verify(x=(cx + 0.5) * v.maps.cell, y=(cy + 0.5) * v.maps.cell, cell=int(cell),
                                  allow_soaring=False)
                continue
            if self.verifying.get(i) is None and st.get("mode") in ("LOITER",):
                g = self._search_goal(i)
                if g is not None:
                    goals[i] = g
        return goals
