"""Nadir thermal (LWIR) camera: footprint, detections, false alarms, verification."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import SensorParams
from .people import IN_WATER, People
from .terrain import LANDCOVER_NAMES, Terrain


@dataclass(frozen=True)
class SensorFrame:
    """One camera frame: everything a planner needs, no detection probabilities."""

    t: float
    drone_id: int
    seq: int
    cell_idx: np.ndarray  # flat 50 m cell indices observed
    camera_height_m: float  # camera height above the ground at the nadir point
    detections: tuple = ()  # flat cell indices with a detection


def p_detect(range_m, camera_range_m, d_target=0.5, m_cover=1.0, m_state=1.0, sp: SensorParams | None = None):
    """Johnson-style detection probability from the range ratio (plan section 11)."""
    sp = sp or SensorParams()
    rng_m = np.maximum(np.asarray(range_m, float), 1e-3)
    r = (np.asarray(d_target, float) / sp.ref_target_m) * camera_range_m / rng_m
    E = sp.exp_a + sp.exp_b * r
    rE = r**E
    p = sp.p_max * rE / (1.0 + rE) * m_cover * m_state
    return np.clip(p, 0.0, 1.0)


class Sensor:
    def __init__(self, sp: SensorParams, terr: Terrain, camera_range_m: float, n_drones: int,
                 rng: np.random.Generator):
        self.sp = sp
        self.terr = terr
        self.R = camera_range_m
        self.rng = rng
        self.seq = np.zeros(n_drones, np.int64)
        self.tan = math.tan(math.radians(sp.fov_deg / 2.0))
        cov = sp.cover_mod
        self.cover = np.zeros(terr.landcover.shape, np.float32)
        for k, name in LANDCOVER_NAMES.items():
            self.cover[terr.landcover == k] = cov[name]
        self.cell = terr.cell_size
        self.n = terr.n
        self.max_height = sp.skip_above_range_factor * camera_range_m

    def footprint_cells(self, x, y, h):
        """Flat indices of cells whose centres lie inside the footprint (+ the nadir cell)."""
        c, n = self.cell, self.n
        r = h * self.tan
        ix0, iy0 = int(x // c), int(y // c)
        k = int(math.ceil(r / c)) + 1
        xs = np.arange(max(ix0 - k, 0), min(ix0 + k, n - 1) + 1)
        ys = np.arange(max(iy0 - k, 0), min(iy0 + k, n - 1) + 1)
        cx = (xs + 0.5) * c - x
        cy = (ys + 0.5) * c - y
        inside = (cx[None, :] ** 2 + cy[:, None] ** 2) <= r * r
        iy, ix = np.nonzero(inside)
        cells = ys[iy] * n + xs[ix]
        if 0 <= ix0 < n and 0 <= iy0 < n:
            nadir = iy0 * n + ix0
            if not np.any(cells == nadir):
                cells = np.append(cells, nadir)
        return cells, r

    def frame(self, t: float, drone_id: int, x: float, y: float, z: float, people: People,
              ground_z: float):
        """Take a frame. Returns (SensorFrame or None, truth list of (person, cell) detected)."""
        h = z - ground_z
        if h <= 0 or h > self.max_height:
            return None, []
        cells, r = self.footprint_cells(x, y, h)
        sp = self.sp
        det_cells: list[int] = []
        truth = []
        act = people.active
        if act.any():
            dx = people.x - x
            dy = people.y - y
            inside = act & (dx * dx + dy * dy <= r * r)
            if inside.any():
                k = np.nonzero(inside)[0]
                rng_m = z - people.elevation()[k]
                d_t = np.where(people.state[k] == IN_WATER, sp.target_water_m, sp.target_land_m)
                pc = people.cells()[k]
                p = p_detect(rng_m, self.R, d_t, self.cover.ravel()[pc], people.detect_factor()[k], sp)
                hit = self.rng.random(len(k)) < p
                for kk, c_ in zip(k[hit], pc[hit]):
                    det_cells.append(int(c_))
                    truth.append((int(kk), int(c_)))
        n_fa = self.rng.poisson(sp.false_alarm_rate * len(cells))
        if n_fa:
            det_cells.extend(int(c_) for c_ in self.rng.choice(cells, size=n_fa))
        if det_cells:
            extra = np.setdiff1d(np.asarray(det_cells), cells)
            if len(extra):
                cells = np.concatenate([cells, extra])
        self.seq[drone_id] += 1
        dets = tuple(dict.fromkeys(det_cells))  # unique, in order
        f = SensorFrame(t=t, drone_id=drone_id, seq=int(self.seq[drone_id]), cell_idx=cells.astype(np.int64),
                        camera_height_m=float(h), detections=dets)
        return f, truth

    def verify(self, cell: int, people: People, t: float):
        """Resolve a 60 s verification orbit over `cell`. Returns (confirmed, person index or None)."""
        act = np.nonzero(people.active)[0]
        here = act[people.cells()[act] == cell] if len(act) else act
        if len(here):
            if self.rng.random() < self.sp.verify_p_confirm:
                return True, int(here[0])
            return False, None
        if self.rng.random() < self.sp.verify_p_reject_fa:
            return False, None
        return True, None  # a false alarm wrongly confirmed
