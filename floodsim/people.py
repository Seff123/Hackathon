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
