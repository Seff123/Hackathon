"""People: hidden ground truth. Placement, behaviour state machine, danger time and survival."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PeopleParams, ScenarioSpec
from .terrain import OPEN, Terrain

HOME, VEHICLE_CAT, OUTDOOR_CAT = 0, 1, 2
CAT_NAMES = {HOME: "home", VEHICLE_CAT: "vehicle", OUTDOOR_CAT: "outdoor"}

HOME_INSIDE, ROOF, VEHICLE, OUTDOOR, SAFE_HIGH, IN_WATER = range(6)
STATE_NAMES = ["home_inside", "roof", "vehicle", "outdoor", "safe_high", "in_water"]
STATE_FACTOR = np.array([0.02, 1.0, 0.2, 1.0, 1.0, 0.8])
EXPOSED = np.array([False, False, False, True, False, True])
SHELTERED = np.array([True, True, True, False, False, False])


def hazard_rating(d, v):
    """Environment Agency FD2320 (simplified): HR = d (v + 0.5) + DF."""
    d = np.asarray(d, dtype=float)
    df = np.where(d <= 0.25, 0.0, np.where(d <= 0.75, 0.5, 1.0))
    return np.where(d > 0, d * (np.asarray(v, dtype=float) + 0.5) + df, 0.0)


def place_people(spec: ScenarioSpec, terrain: Terrain, rng: np.random.Generator, pp: PeopleParams):
    """Returns dict of arrays: x, y, cat, mobile."""
    n = terrain.n
    c = terrain.cell_size
    N = int(spec.n_targets)
    cats = rng.choice(3, size=N, p=np.asarray(spec.category_weights) / np.sum(spec.category_weights))
    hand = terrain.hand
    w_home = terrain.building_density * (1.0 + pp.home_low_boost * (hand < pp.home_low_hand))
    w_veh = terrain.road_mask * (1.0 + 3.0 * (hand < pp.vehicle_low_hand))
    w_out = ((terrain.landcover == OPEN) & (terrain.dist_to_river <= pp.outdoor_river_m)).astype(float)
    weights = {HOME: w_home, VEHICLE_CAT: w_veh, OUTDOOR_CAT: w_out}
    xs = np.zeros(N)
    ys = np.zeros(N)
    for i, k in enumerate(cats):
        w = weights[int(k)].reshape(-1).astype(float)
        if w.sum() <= 0:
            w = (terrain.landcover != 1).reshape(-1).astype(float)
        idx = rng.choice(w.size, p=w / w.sum())
        iy, ix = divmod(int(idx), n)
        xs[i] = (ix + rng.uniform(0.1, 0.9)) * c
        ys[i] = (iy + rng.uniform(0.1, 0.9)) * c
    mobile = rng.uniform(size=N) >= spec.immobile_fraction
    return {"x": xs, "y": ys, "cat": cats.astype(np.int64), "mobile": mobile}


@dataclass
class PeopleState:
    x: np.ndarray
    y: np.ndarray
    cat: np.ndarray
    state: np.ndarray
    mobile: np.ndarray
    walking: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    t_danger: np.ndarray
    t_affected: np.ndarray
    found: np.ndarray
    t_found: np.ndarray
    lost: np.ndarray
    surv_frozen: np.ndarray
    ever_affected: np.ndarray
    walk_speed: float
    drift_frac: float
    params: PeopleParams
    start_x: np.ndarray
    start_y: np.ndarray

    @property
    def n(self) -> int:
        return len(self.x)

    @staticmethod
    def create(placement: dict, spec: ScenarioSpec, pp: PeopleParams) -> "PeopleState":
        N = len(placement["x"])
        cat = placement["cat"]
        state = np.where(cat == HOME, HOME_INSIDE, np.where(cat == VEHICLE_CAT, VEHICLE, OUTDOOR)).astype(np.int64)
        nan = np.full(N, np.nan)
        return PeopleState(
            x=placement["x"].astype(float).copy(), y=placement["y"].astype(float).copy(), cat=cat.copy(),
            state=state, mobile=placement["mobile"].copy(), walking=np.zeros(N, bool), tx=nan.copy(),
            ty=nan.copy(), t_danger=np.zeros(N), t_affected=nan.copy(), found=np.zeros(N, bool),
            t_found=nan.copy(), lost=np.zeros(N, bool), surv_frozen=nan.copy(), ever_affected=np.zeros(N, bool),
            walk_speed=float(spec.walk_speed_ms), drift_frac=float(spec.drift_fraction), params=pp,
            start_x=placement["x"].astype(float).copy(), start_y=placement["y"].astype(float).copy())

    def survival(self) -> np.ndarray:
        s = np.exp(-self.t_danger / self.params.t_crit_s)
        s = np.where(self.found, self.surv_frozen, s)
        return np.where(self.lost & ~self.found, 0.0, s)

    def active(self) -> np.ndarray:
        return ~self.found & ~self.lost

    def elevation(self, terrain: Terrain) -> np.ndarray:
        """Elevation of the person (roof height for people at home or on roofs)."""
        from .grid import nearest
        z = nearest(terrain.z, self.x, self.y, terrain.cell_size)
        roof = nearest(terrain.obstacle_h, self.x, self.y, terrain.cell_size)
        on_bld = (self.state == HOME_INSIDE) | (self.state == ROOF)
        return z + np.where(on_bld, roof, 0.0)

    def step(self, t: float, dt: float, flood, terrain: Terrain, domain: float) -> None:
        pp = self.params
        act = self.active()
        if not act.any():
            return
        d = flood.depth(self.x, self.y, t)
        u, v = flood.velocity(self.x, self.y, t)
        sp = np.hypot(u, v)
        hr = hazard_rating(d, sp)
        newly = act & (d > pp.affected_depth) & np.isnan(self.t_affected)
        self.t_affected[newly] = t
        self.ever_affected |= newly

        st = self.state
        # home -> roof
        st[act & (st == HOME_INSIDE) & (d >= pp.roof_depth)] = ROOF
        # in water -> outdoor
        out_w = act & (st == IN_WATER) & (d < pp.leave_water_depth)
        st[out_w] = OUTDOOR
        # outdoor: start walking to higher ground
        outdoor = act & (st == OUTDOOR)
        if outdoor.any():
            need = outdoor & self.mobile & ~self.walking
            if need.any():
                wet_near = self._flood_near(flood, terrain, t, np.flatnonzero(need))
                for i in np.flatnonzero(need):
                    if d[i] > pp.walk_trigger_depth or wet_near[i]:
                        self._start_walk(i, terrain)
            # outdoor -> in water
            st[outdoor & (d > pp.in_water_depth)] = IN_WATER
        # walking
        wk = act & self.walking & (st == OUTDOOR)
        if wk.any():
            dx = self.tx[wk] - self.x[wk]
            dy = self.ty[wk] - self.y[wk]
            dist = np.hypot(dx, dy)
            step = self.walk_speed * dt
            arrive = dist <= step
            f = np.where(arrive, 1.0, step / np.maximum(dist, 1e-9))
            self.x[wk] += dx * f
            self.y[wk] += dy * f
            idx = np.flatnonzero(wk)[arrive]
            st[idx] = SAFE_HIGH
            self.walking[idx] = False
        # drifting in water
        iw = act & (st == IN_WATER)
        if iw.any():
            self.x[iw] += self.drift_frac * u[iw] * dt
            self.y[iw] += self.drift_frac * v[iw] * dt
            gone = iw & ((self.x < 0) | (self.x > domain) | (self.y < 0) | (self.y > domain))
            self.lost |= gone
            self.x = np.clip(self.x, 0.0, domain - 1e-6)
            self.y = np.clip(self.y, 0.0, domain - 1e-6)
        # danger time
        exposed = EXPOSED[st]
        sheltered = SHELTERED[st]
        danger = act & ((exposed & (hr > pp.hr_exposed)) | (sheltered & (hr > pp.hr_sheltered)))
        self.t_danger[danger] += dt

    def _flood_near(self, flood, terrain: Terrain, t: float, idx: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n, dtype=bool)
        wet = flood.depth_grid50(t) > self.params.affected_depth
        c = terrain.cell_size
        r = int(np.ceil(self.params.walk_trigger_flood_m / c))
        n = wet.shape[0]
        for i in idx:
            iy, ix = int(self.y[i] // c), int(self.x[i] // c)
            win = wet[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1]
            if win.any():
                yy, xx = np.nonzero(win)
                yy = (yy + max(0, iy - r) + 0.5) * c
                xx = (xx + max(0, ix - r) + 0.5) * c
                out[i] = np.min(np.hypot(xx - self.x[i], yy - self.y[i])) <= self.params.walk_trigger_flood_m
        return out

    def _start_walk(self, i: int, terrain: Terrain) -> None:
        c = terrain.cell_size
        n = terrain.n
        iy, ix = min(int(self.y[i] // c), n - 1), min(int(self.x[i] // c), n - 1)
        h0 = terrain.hand[iy, ix]
        ok = (terrain.hand >= h0 + self.params.high_ground_hand_m) & (terrain.landcover != 1)
        if not ok.any():
            return
        yy, xx = np.nonzero(ok)
        dd = (yy - iy) ** 2 + (xx - ix) ** 2
        k = int(np.argmin(dd))
        self.tx[i] = (xx[k] + 0.5) * c
        self.ty[i] = (yy[k] + 0.5) * c
        self.walking[i] = True

    def mark_found(self, i: int, t: float) -> None:
        if self.found[i]:
            return
        self.surv_frozen[i] = float(np.exp(-self.t_danger[i] / self.params.t_crit_s))
        self.found[i] = True
        self.t_found[i] = t
