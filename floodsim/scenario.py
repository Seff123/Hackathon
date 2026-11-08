"""Seed -> validated ScenarioSpec -> World (cached)."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import Config, ScenarioBounds, ScenarioSpec, sample_raw_spec, FLOOD_TYPES
from .grid import world_to_grid
from .hydrology import fraction_draining_to
from .people import Placement, place_people
from .rng import stream
from .terrain import Terrain, build_terrain, pick_base

MAX_ATTEMPTS = 50
_TERRAIN_CACHE: "OrderedDict[str, tuple[Terrain, Placement]]" = OrderedDict()


class ScenarioRejected(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Hydrograph and storm (shared with flood.py)
# --------------------------------------------------------------------------


def hydrograph(spec: ScenarioSpec, t_s):
    """River inflow (m^3/s) at seconds since flood onset."""
    t = np.asarray(t_s, float)
    q_peak = spec.peak_inflow_m3s
    q_base = spec.base_flow_frac * q_peak
    if not spec.has_river_peak:
        return np.full_like(t, q_base)
    tp = spec.time_to_peak_h * 3600.0
    r = np.maximum(t, 0.0) / tp
    shape = (r * np.exp(1.0 - r)) ** spec.hydro_shape_k
    return q_base + (q_peak - q_base) * shape


def solver_window(spec: ScenarioSpec, cfg: Config):
    """(t_start_with_spinup, t_episode_start, t_end) in seconds since onset."""
    t_ep = spec.start_offset_h * 3600.0
    t0 = min(0.0, t_ep) - cfg.world.spinup_s
    t1 = t_ep + spec.episode_s + cfg.world.post_episode_s
    return t0, t_ep, t1


# --------------------------------------------------------------------------
# Cheap HAND-based flood estimate used only by the rejection checks
# --------------------------------------------------------------------------


def cheap_flood_extent(spec: ScenarioSpec, terr: Terrain, cfg: Config) -> np.ndarray:
    _, t_ep, t1 = solver_window(spec, cfg)
    ts = np.linspace(t_ep, t1, 400)
    q = float(hydrograph(spec, ts).max())
    wc = cfg.world
    area = wc.domain_m**2
    if spec.has_rain:
        rain = spec.rain_peak_mmh / 1000.0 / 3600.0
        footprint = min(2 * math.pi * (spec.storm_radius_km * 1000.0) ** 2, area)
        c_mean = 0.55
        q += rain * c_mean * footprint * min(1.0, spec.storm_duration_h / 2.0) * 0.5
    S = max(terr.river_slope, 1e-4)
    c = terr.cell_size
    on_main = terr.river_cells.ravel()[terr.drain_id.ravel()].reshape(terr.hand.shape) & terr.land
    hand_main = terr.hand[on_main].astype(float)
    Lr = max(terr.river_length_m, 1000.0)
    w_ch = spec.channel_width_m

    def q_of(h):
        k_ch = h ** (5 / 3) * math.sqrt(S) / spec.n_channel
        d = np.maximum(h - hand_main, 0.0)
        k_fp = (c * c / Lr) * np.sum(d ** (5 / 3)) * math.sqrt(S) / spec.n_floodplain
        return w_ch * k_ch + k_fp

    lo, hi = 0.0, 40.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if q_of(mid) < q:
            lo = mid
        else:
            hi = mid
    h = 0.5 * (lo + hi)
    wet = on_main & (terr.hand < h)
    if spec.has_rain:
        wet |= terr.land & (terr.depression_depth > 0.1)
    return wet | terr.river_cells


def rejection_checks(spec: ScenarioSpec, terr: Terrain, people: Placement, cfg: Config) -> list[str]:
    fails = []
    land = terr.land
    wet = cheap_flood_extent(spec, terr, cfg)
    frac = float((wet & land).sum() / max(land.sum(), 1))
    lo, hi = cfg.world.cheap_extent_range
    if not lo <= frac <= hi:
        fails.append(f"extent {frac:.3f}")
    d = ndimage.distance_transform_edt(~(wet & land)) * terr.cell_size
    iy, ix = world_to_grid(people.x, people.y, terr.cell_size, terr.n)
    near = float(np.mean(d[iy, ix] <= 1000.0))
    if near < 0.6:
        fails.append(f"people near flood {near:.2f}")
    base = pick_base(terr, cfg.world, excluded=ndimage.binary_dilation(wet, iterations=2))
    terr.set_base(base, cfg.world.base_clear_radius_m)
    by, bx = world_to_grid(base[0], base[1], terr.cell_size, terr.n)
    if wet[by, bx] or not terr.road_mask[by, bx]:
        fails.append("base flooded or off-road")
    if float(terr.z.max() - terr.z[by, bx]) > 300.0:
        fails.append("terrain > 300 m above base")
    z0 = cfg.weather.z0_m
    u300 = spec.wind10_ms * math.log(300 / z0) / math.log(10 / z0)
    if u300 > 15.0:
        fails.append("wind at 300 m > 15 m/s")
    if fraction_draining_to(terr, land, spec.outlet_edge) < 0.8:
        fails.append("drainage to outlet < 80%")
    return fails


def solver_check(spec: ScenarioSpec, cfg: Config) -> list[str]:
    """Run the flood solver and check its peak extent; the result is kept for build_world."""
    from . import world as W

    key = spec.spec_hash + cfg.world_key()
    if key in W._MEM:
        w = W._MEM[key]
    else:
        from pathlib import Path

        cached = W._cache_path(spec, "cache")
        w = W._load(spec, cfg, cached) if cached.exists() else None
        if w is None:
            w = W.flood_world(spec, cfg)
            W._FLOOD[key] = w
            while len(W._FLOOD) > 3:
                W._FLOOD.pop(next(iter(W._FLOOD)))
    ext = w.stats.get("peak_extent", w.peak_extent())
    lo, hi = cfg.world.extent_range
    return [] if lo <= ext <= hi else [f"solver extent {ext:.3f}"]


def sample_spec(seed: int, bounds: ScenarioBounds | None = None, cfg: Config | None = None,
                return_log: bool = False, flood_check: bool = True):
    """Sample a spec for `seed` that passes every rejection check.

    The plan's cheap HAND-based checks run first; then (unless disabled) the
    flood solver's peak extent must lie in extent_range. The solver result is
    reused by build_world, so the check costs nothing extra later.
    """
    cfg = cfg or Config()
    bounds = bounds or cfg.bounds
    log = []
    for attempt in range(MAX_ATTEMPTS):
        spec = sample_raw_spec(seed, bounds, attempt)
        terr = build_terrain(spec, cfg.world, stream(seed, "terrain", attempt))
        people = place_people(spec, terr, cfg.people, stream(seed, "people", attempt))
        fails = rejection_checks(spec, terr, people, cfg)
        if not fails:
            _remember(spec, cfg, terr, people)
            if flood_check and cfg.world.solver_extent_check:
                fails = solver_check(spec, cfg)
        log.append((attempt, fails))
        if not fails:
            return (spec, log) if return_log else spec
    raise ScenarioRejected(f"seed {seed}: all {MAX_ATTEMPTS} attempts failed the rejection checks")


def spec_for(seed: int, attempt: int, bounds: ScenarioBounds | None = None) -> ScenarioSpec:
    """Rebuild the spec for a recorded (seed, attempt) pair, e.g. from seeds_<split>.json."""
    return sample_raw_spec(seed, bounds or ScenarioBounds(), attempt)


def _key(spec: ScenarioSpec, cfg: Config) -> str:
    return spec.spec_hash + cfg.world_key()


def _remember(spec, cfg, terr, people):
    _TERRAIN_CACHE[_key(spec, cfg)] = (terr, people)
    while len(_TERRAIN_CACHE) > 4:
        _TERRAIN_CACHE.popitem(last=False)


def terrain_for(spec: ScenarioSpec, cfg: Config) -> tuple[Terrain, Placement]:
    """Terrain (with base placed by the checks) and initial people for a spec."""
    k = _key(spec, cfg)
    if k in _TERRAIN_CACHE:
        return _TERRAIN_CACHE[k]
    terr = build_terrain(spec, cfg.world, stream(spec.seed, "terrain", spec.attempt))
    people = place_people(spec, terr, cfg.people, stream(spec.seed, "people", spec.attempt))
    rejection_checks(spec, terr, people, cfg)  # places the base exactly as during sampling
    _remember(spec, cfg, terr, people)
    return terr, people


def sample_seeds(n: int, strata: tuple[str, ...] = ("flood_type", "n_drones"), start: int = 0,
                 bounds: ScenarioBounds | None = None) -> list[int]:
    """Seeds balanced across flood types and swarm sizes (uses attempt-0 specs)."""
    bounds = bounds or ScenarioBounds()
    bins: dict[tuple, list[int]] = {}
    seed = start
    while seed < start + 50 * max(n, 1):
        s = sample_raw_spec(seed, bounds, 0)
        bins.setdefault(tuple(getattr(s, k) for k in strata), []).append(seed)
        seed += 1
        if sum(len(v) for v in bins.values()) >= 6 * n and len(bins) >= 2:
            break
    keys = sorted(bins, key=str)
    out: list[int] = []
    i = 0
    while len(out) < n:
        progressed = False
        for k in keys:
            if i < len(bins[k]):
                out.append(bins[k][i])
                progressed = True
                if len(out) == n:
                    break
        if not progressed:
            break
        i += 1
    return sorted(out)
