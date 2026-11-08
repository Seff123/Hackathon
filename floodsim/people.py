"""People: placement, behaviour state machine, danger time and survival.

Truth only: nothing here is ever handed to a controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import PeopleParams, ScenarioSpec
from .grid import cell_centres, world_to_grid
from .terrain import OPEN, URBAN, Terrain

HOME, VEHICLE, OUTDOOR = 0, 1, 2
CATEGORY_NAMES = ("home", "vehicle", "outdoor")

# behaviour states
HOME_INSIDE, ROOF, VEHICLE_S, OUTDOOR_S, SAFE_HIGH, IN_WATER, LOST = range(7)
STATE_NAMES = ("HOME_INSIDE", "ROOF", "VEHICLE", "OUTDOOR", "SAFE_HIGH", "IN_WATER", "LOST")
SHELTERED = (HOME_INSIDE, ROOF, VEHICLE_S)
EXPOSED = (OUTDOOR_S, IN_WATER)


@dataclass
class Placement:
    x: np.ndarray
    y: np.ndarray
    category: np.ndarray
    mobile: np.ndarray


def _pick(rng, weights: np.ndarray, k: int, n: int, cell: float):
    w = weights.ravel().astype(float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    idx = rng.choice(w.size, size=k, p=w / w.sum())
    iy, ix = np.divmod(idx, n)
    x = (ix + rng.uniform(0.15, 0.85, k)) * cell
    y = (iy + rng.uniform(0.15, 0.85, k)) * cell
    return x, y


def place_people(spec: ScenarioSpec, terr: Terrain, pp: PeopleParams, rng: np.random.Generator) -> Placement:
    n, c = terr.n, terr.cell_size
    N = spec.n_targets
    cat = rng.choice(3, size=N, p=np.asarray(spec.category_weights) / np.sum(spec.category_weights))
    mobile = rng.random(N) >= spec.immobile_frac
    x = np.zeros(N)
    y = np.zeros(N)
    land = terr.land
    w_home = terr.building_density * (terr.landcover == URBAN) * (1 + pp.home_hand_boost * (terr.hand < pp.home_hand_m))
    road_ok = terr.road_mask & ~terr.bridge_mask & land
    w_veh = road_ok * (1 + pp.vehicle_boost * (terr.hand < pp.vehicle_hand_m))
    w_out = ((terr.landcover == OPEN) & ~terr.road_mask & (terr.dist_to_river <= pp.outdoor_river_m)).astype(float)
    if w_home.sum() <= 0:
        w_home = land.astype(float)
    if w_veh.sum() <= 0:
        w_veh = land.astype(float)
    if w_out.sum() <= 0:
        w_out = land.astype(float)
    for k, w in ((HOME, w_home), (VEHICLE, w_veh), (OUTDOOR, w_out)):
        sel = np.nonzero(cat == k)[0]
        if len(sel):
            x[sel], y[sel] = _pick(rng, w, len(sel), n, c)
    return Placement(x=x, y=y, category=cat.astype(np.int8), mobile=mobile)


class People:
    """Truth state of all targets; stepped every people_dt seconds."""

    def __init__(self, placement: Placement, spec: ScenarioSpec, terr: Terrain, pp: PeopleParams):
        self.pp = pp
        self.terr = terr
        N = len(placement.x)
        self.n = N
        self.x = placement.x.astype(float).copy()
        self.y = placement.y.astype(float).copy()
        self.category = placement.category.copy()
        self.mobile = placement.mobile.copy()
        self.state = np.select([self.category == HOME, self.category == VEHICLE],
                               [HOME_INSIDE, VEHICLE_S], OUTDOOR_S).astype(np.int8)
        self.walk_speed = spec.walk_speed_ms
        self.drift = spec.drift_frac
        self.walking = np.zeros(N, bool)
        self.tx = np.zeros(N)
        self.ty = np.zeros(N)
        self.t_danger = np.zeros(N)
        self.t_affected = np.full(N, np.nan)
        self.found = np.zeros(N, bool)
        self.t_found = np.full(N, np.nan)
        self.state_at_find = np.full(N, -1, np.int8)
        self.surv_frozen = np.full(N, np.nan)
        self.lost = np.zeros(N, bool)
        self.ever_flooded = np.zeros(N, bool)
        self.n_steps = 0

    # ------------------------------------------------------------- queries
    @property
    def active(self):
        """Not yet found and still on the map."""
        return ~self.found & ~self.lost

    def survival(self):
        p = np.exp(-self.t_danger / self.pp.t_crit_s)
        p = np.where(self.found, self.surv_frozen, p)
        return np.where(self.lost & ~self.found, 0.0, p)

    def elevation(self):
        """Height of the person's body for camera range: roof top on a roof, else ground."""
        c = self.terr.cell_size
        iy, ix = world_to_grid(self.x, self.y, c, self.terr.n)
        z = self.terr.z[iy, ix].astype(float)
        return np.where(self.state == ROOF, z + self.terr.raw_obstacle_h[iy, ix], z)

    def cells(self):
        iy, ix = world_to_grid(self.x, self.y, self.terr.cell_size, self.terr.n)
        return iy * self.terr.n + ix

    def detect_factor(self):
        f = self.pp.detect_factor
        table = np.array([f["HOME_INSIDE"], f["ROOF"], f["VEHICLE"], f["OUTDOOR"], f["SAFE_HIGH"],
                          f["IN_WATER"], 0.0])
        return table[self.state]

    def mark_found(self, k: int, t: float):
        self.found[k] = True
        self.t_found[k] = t
        self.state_at_find[k] = self.state[k]
        self.surv_frozen[k] = float(np.exp(-self.t_danger[k] / self.pp.t_crit_s))

    # ------------------------------------------------------------- dynamics
    def _start_walk(self, k):
        terr = self.terr
        c = terr.cell_size
        iy, ix = world_to_grid(self.x[k], self.y[k], c, terr.n)
        goal = terr.hand >= terr.hand[iy, ix] + self.pp.high_ground_hand_m
        goal &= terr.land
        if not goal.any():
            return
        gy, gx = np.nonzero(goal)
        d2 = (gx - ix) ** 2 + (gy - iy) ** 2
        j = int(np.argmin(d2))
        self.tx[k], self.ty[k] = (gx[j] + 0.5) * c, (gy[j] + 0.5) * c
        self.walking[k] = True

    def step(self, t: float, flood, dt: float):
        pp = self.pp
        act = self.active
        if not act.any():
            return
        d = flood.depth(self.x, self.y, t)
        u, v = flood.velocity(self.x, self.y, t)
        sp = np.hypot(u, v)
        hr = np.where(d > 0, d * (sp + 0.5) + np.where(d <= 0.25, 0.0, np.where(d <= 0.75, 0.5, 1.0)), 0.0)
        newly = act & np.isnan(self.t_affected) & (d > pp.affected_depth_m)
        self.t_affected = np.where(newly, t, self.t_affected)
        st = self.state
        exposed = (st == OUTDOOR_S) | (st == IN_WATER)
        sheltered = (st == HOME_INSIDE) | (st == ROOF) | (st == VEHICLE_S)
        danger = act & ((exposed & (hr > pp.hr_exposed)) | (sheltered & (hr > pp.hr_sheltered)))
        self.t_danger += danger * dt
        # transitions
        self.state = np.where(act & (st == HOME_INSIDE) & (d >= pp.roof_depth_m), ROOF, st).astype(np.int8)
        to_water = act & (self.state == OUTDOOR_S) & (d > pp.to_water_depth_m)
        self.state = np.where(to_water, IN_WATER, self.state).astype(np.int8)
        self.walking &= ~to_water
        out_water = act & (self.state == IN_WATER) & (d < pp.from_water_depth_m)
        self.state = np.where(out_water, OUTDOOR_S, self.state).astype(np.int8)
        # outdoor mobile people walk to higher ground when water is on them or nearby
        idle = act & (self.state == OUTDOOR_S) & self.mobile & ~self.walking
        if idle.any():
            wet_here = d > pp.walk_trigger_depth_m
            near = np.zeros(self.n, bool)
            grid = None
            for k in np.nonzero(idle & ~wet_here)[0]:
                if grid is None:
                    grid = flood.depth_grid(t) > pp.walk_trigger_depth_m
                c = self.terr.cell_size
                r = int(np.ceil(pp.walk_trigger_dist_m / c))
                iy, ix = world_to_grid(self.x[k], self.y[k], c, self.terr.n)
                sub = grid[max(iy - r, 0):iy + r + 1, max(ix - r, 0):ix + r + 1]
                near[k] = bool(sub.any())
            for k in np.nonzero(idle & (wet_here | near))[0]:
                self._start_walk(k)
        # walking
        w = act & self.walking & (self.state == OUTDOOR_S)
        if w.any():
            dx, dy = self.tx - self.x, self.ty - self.y
            dist = np.hypot(dx, dy)
            stepl = self.walk_speed * dt
            arrive = w & (dist <= stepl)
            f = np.where(dist > 0, np.minimum(stepl / np.maximum(dist, 1e-9), 1.0), 0.0)
            self.x = np.where(w, self.x + dx * f, self.x)
            self.y = np.where(w, self.y + dy * f, self.y)
            self.state = np.where(arrive, SAFE_HIGH, self.state).astype(np.int8)
            self.walking &= ~arrive
        # drifting in water
        wd = act & (self.state == IN_WATER)
        if wd.any():
            self.x = np.where(wd, self.x + self.drift * u * dt, self.x)
            self.y = np.where(wd, self.y + self.drift * v * dt, self.y)
            L = self.terr.domain_m
            off = wd & ((self.x < 0) | (self.y < 0) | (self.x >= L) | (self.y >= L))
            self.lost |= off
            self.state = np.where(off, LOST, self.state).astype(np.int8)
            self.x = np.clip(self.x, 0, L - 1e-3)
            self.y = np.clip(self.y, 0, L - 1e-3)
        self.ever_flooded |= act & (d > pp.affected_depth_m)
        self.n_steps += 1
