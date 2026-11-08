"""Nadir thermal camera: footprint, detection model, false alarms. Frames carry no detection probabilities."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import SensorParams
from .people import IN_WATER, STATE_FACTOR, PeopleState
from .terrain import FOREST, OPEN, URBAN, WATER, Terrain


def pd_range(z_cam, size, R):
    """Johnson-style curve: r = (size/0.5)(R/z), p = 0.95 r^E/(1 + r^E), E = 2.7 + 0.7 r."""
    r = (np.asarray(size, dtype=float) / 0.5) * (R / np.maximum(np.asarray(z_cam, dtype=float), 1.0))
    E = 2.7 + 0.7 * r
    rE = r ** E
    return 0.95 * rE / (1.0 + rE)


@dataclass
class SensorFrame:
    t: float
    drone_id: int
    seq: int
    cell_idx: np.ndarray
    camera_height_m: float
    detections: list = field(default_factory=list)


def cover_modifier(landcover: np.ndarray, sp: SensorParams) -> np.ndarray:
    m = np.ones(landcover.shape, dtype=np.float32)
    m[landcover == URBAN] = sp.cover_mod["urban"]
    m[landcover == FOREST] = sp.cover_mod["forest"]
    m[landcover == OPEN] = sp.cover_mod["open"]
    m[landcover == WATER] = sp.cover_mod["water"]
    return m


class Sensor:
    def __init__(self, sp: SensorParams, terrain: Terrain, camera_range: float, lambda_fa: float, rng):
        self.sp = sp
        self.R = float(camera_range)
        self.lambda_fa = float(lambda_fa)
        self.rng = rng
        self.cell = terrain.cell_size
        self.n = terrain.n
        self.terr = terrain.z
        self.cover = cover_modifier(terrain.landcover, sp)
        self.tan = sp.tan_half
        k = int(math.ceil(2 * self.R * self.tan / self.cell)) + 2
        oy, ox = np.mgrid[-k:k + 1, -k:k + 1]
        self.off_y = oy.reshape(-1)
        self.off_x = ox.reshape(-1)
        self.seq = {}

    def footprint(self, x: float, y: float, h: float) -> np.ndarray:
        """Flat indices of cells whose centre is inside the footprint circle, plus the nadir cell."""
        c, n = self.cell, self.n
        r = max(h, 0.0) * self.tan
        ix0, iy0 = int(x // c), int(y // c)
        iy = iy0 + self.off_y
        ix = ix0 + self.off_x
        cxw = (ix + 0.5) * c - x
        cyw = (iy + 0.5) * c - y
        inside = (cxw * cxw + cyw * cyw <= r * r) | ((self.off_x == 0) & (self.off_y == 0))
        inside &= (iy >= 0) & (iy < n) & (ix >= 0) & (ix < n)
        return (iy[inside] * n + ix[inside]).astype(np.int64)

    def pd_person(self, z_cam, in_water, state, cell):
        size = np.where(in_water, self.sp.water_size, self.sp.land_size)
        p = pd_range(z_cam, size, self.R) * self.cover.reshape(-1)[cell] * STATE_FACTOR[state]
        return np.clip(p, 0.0, 1.0)

    def frame(self, drone_id: int, x: float, y: float, z: float, t: float, people: PeopleState,
              people_elev: np.ndarray | None = None):
        """Returns (SensorFrame or None, truth list of (cell, person_id or -1))."""
        c, n = self.cell, self.n
        if not (0 <= x < n * c and 0 <= y < n * c):
            return None, []
        h = z - float(self.terr[int(y // c), int(x // c)])
        if h <= 0 or h > 2 * self.R:
            return None, []
        cells = self.footprint(x, y, h)
        r = h * self.tan
        truth = []
        dets = []
        act = np.flatnonzero(people.active())
        if len(act):
            px, py = people.x[act], people.y[act]
            dist2 = (px - x) ** 2 + (py - y) ** 2
            pcell = (np.clip((py // c).astype(np.int64), 0, n - 1) * n + np.clip((px // c).astype(np.int64), 0, n - 1))
            nadir = int(y // c) * n + int(x // c)
            seen = (dist2 <= r * r) | (pcell == nadir)
            if seen.any():
                idx = act[seen]
                elev = people_elev[idx] if people_elev is not None else self.terr.reshape(-1)[pcell[seen]]
                st = people.state[idx]
                pd = self.pd_person(z - elev, st == IN_WATER, st, pcell[seen])
                hit = self.rng.uniform(size=len(idx)) < pd
                for pid, cc in zip(idx[hit].tolist(), pcell[seen][hit].tolist()):
                    truth.append((cc, pid))
                    dets.append(cc)
        nfa = self.rng.poisson(self.lambda_fa * len(cells))
        if nfa:
            for cc in self.rng.choice(cells, size=nfa).tolist():
                truth.append((cc, -1))
                dets.append(cc)
        if dets:
            extra = np.setdiff1d(np.array(dets, dtype=np.int64), cells)
            if len(extra):
                cells = np.concatenate([cells, extra])
        dets = sorted(set(dets))
        s = self.seq.get(drone_id, 0)
        self.seq[drone_id] = s + 1
        return SensorFrame(t=t, drone_id=drone_id, seq=s, cell_idx=cells, camera_height_m=h, detections=dets), truth
