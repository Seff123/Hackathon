"""Scenario sampling with rejection checks, and world building (cached)."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import rng as rngmod
from .config import Config, ScenarioBounds, ScenarioSpec, TeamInputs, WorldConfig, load_config
from .people import PeopleState, place_people
from .terrain import Terrain, generate_terrain

FLOOD_TYPES = ("fluvial", "fluvial+pluvial", "pluvial", "breach")


class ScenarioRejected(RuntimeError):
    pass


def _choice(rng, weights: dict):
    keys = list(weights)
    p = np.array([weights[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def strata_of(seed: int, bounds: ScenarioBounds | None = None) -> tuple[str, int]:
    """Flood type and swarm size depend on the seed only (not the rejection attempt) so strata are stable."""
    b = bounds or ScenarioBounds()
    r = rngmod.child(seed, 7919)
    return _choice(r, b.flood_type), int(b.n_drones.sample(r))


def _sample_raw(seed: int, attempt: int, b: ScenarioBounds, team: TeamInputs, overrides: dict) -> ScenarioSpec:
    r = rngmod.stream(seed, attempt, "spec")
    flood_type, n_drones = strata_of(seed, b)
    n_towns = int(b.n_towns.sample(r))
    cloud = b.cloud_cover.sample(r)
    wstar = b.wstar_base_ms.sample(r) * (1.0 - cloud) * b.wstar_factor
    rain_scale = 0.7 + 0.6 * (cloud - b.cloud_cover.lo) / (b.cloud_cover.hi - b.cloud_cover.lo)
    peak_rain = float(np.clip(b.peak_rain_mmh.sample(r) * rain_scale, b.peak_rain_mmh.lo, b.peak_rain_mmh.hi))
    alpha = b.category_concentration * np.asarray(b.category_mean)
    levee = b.levee_height_m.sample(r)
    if flood_type == "breach":
        levee = float(r.uniform(1.0, b.levee_height_m.hi))
    d = dict(
        seed=int(seed), attempt=int(attempt),
        relief_m=b.relief_m.sample(r), spectral_beta=b.spectral_beta.sample(r),
        regional_slope=b.regional_slope.sample(r), outlet_edge=_choice(r, b.outlet_edge),
        valley_width_m=b.valley_width_m.sample(r), valley_depth_m=b.valley_depth_m.sample(r),
        sinuosity=b.sinuosity.sample(r), channel_width_m=b.channel_width_m.sample(r),
        channel_depth_m=b.channel_depth_m.sample(r), n_towns=n_towns,
        town_radius_m=[b.town_radius_m.sample(r) for _ in range(n_towns)], levee_height_m=levee,
        forest_fraction=b.forest_fraction.sample(r), floodplain_noise_m=b.floodplain_noise_m.sample(r),
        river_start_frac=float(r.uniform(-0.3, 0.3)), river_end_frac=float(r.uniform(-0.3, 0.3)),
        flood_type=flood_type, peak_inflow_m3s=b.peak_inflow_m3s.sample(r),
        base_flow_frac=b.base_flow_frac.sample(r), time_to_peak_h=b.time_to_peak_h.sample(r),
        hydrograph_k=b.hydrograph_k.sample(r), start_offset_h=b.start_offset_h.sample(r),
        peak_rain_mmh=peak_rain, climate_factor=b.climate_factor.sample(r),
        storm_duration_h=b.storm_duration_h.sample(r), storm_radius_km=b.storm_radius_km.sample(r),
        storm_track_deg=b.storm_track_dev_deg.sample(r),
        manning_channel=b.manning_channel.sample(r), manning_floodplain=b.manning_floodplain.sample(r),
        manning_urban=b.manning_urban.sample(r), manning_forest=b.manning_forest.sample(r),
        breach_time_h=b.breach_time_h.sample(r), breach_width_m=b.breach_width_m.sample(r),
        breach_head_m=b.breach_head_m.sample(r), fc_sigma0_m=b.fc_sigma0_m.sample(r),
        fc_growth_mph=b.fc_growth_mph.sample(r), fc_bias_m=b.fc_bias_m.sample(r),
        wind10_ms=b.wind10_ms.sample(r), wind_from_deg=b.wind_from_deg.sample(r), cloud_cover=cloud,
        wstar_ms=float(wstar), zi_m=b.zi_m.sample(r), thermal_density_km2=b.thermal_density_km2.sample(r),
        start_hour=b.start_hour.sample(r), ridge_k=b.ridge_k.sample(r),
        n_targets=int(b.n_targets.sample(r)), category_weights=[float(v) for v in r.dirichlet(alpha)],
        immobile_fraction=b.immobile_fraction.sample(r), walk_speed_ms=b.walk_speed_ms.sample(r),
        drift_fraction=b.drift_fraction.sample(r), n_drones=n_drones, lambda_fa=0.002,
        mass_kg=team.mass_kg, battery_life_min=team.battery_life_min, battery_wh=team.battery_wh,
        charge_time_min=team.charge_time_min, signal_radius_km=team.signal_radius_km,
        camera_range_m=team.camera_range_m,
    )
    d.update(overrides)
    return ScenarioSpec(**d)


# ----------------------------------------------------------------------------- cheap flood estimate
def hydrograph(spec: ScenarioSpec, t_flood_s):
    """River inflow (m3/s) at flood time t (s since onset). Pluvial-only scenarios keep base flow."""
    t = np.asarray(t_flood_s, dtype=float)
    qb = spec.base_flow_frac * spec.peak_inflow_m3s
    if not spec.has_fluvial:
        return np.full_like(t, qb)
    tp = spec.time_to_peak_h * 3600.0
    tt = np.maximum(t, 0.0) / tp
    shape = (tt * np.exp(1.0 - tt)) ** spec.hydrograph_k
    return qb + (spec.peak_inflow_m3s - qb) * shape


def channel_slope(spec: ScenarioSpec) -> float:
    return max(spec.regional_slope / spec.sinuosity, 3e-4)


def stage_for_flow(spec: ScenarioSpec, Q: float, wc: WorldConfig | None = None) -> float:
    """Compound-channel normal depth (above bed) for flow Q (bisection). Conveyance factors are calibrated
    against the full solver so the estimate is conservative (see ASSUMPTIONS.md)."""
    wc = wc or WorldConfig()
    S = math.sqrt(channel_slope(spec))
    W, dch, nch, nfp = spec.channel_width_m, spec.channel_depth_m, spec.manning_channel, spec.manning_floodplain
    Wv = spec.valley_width_m

    def q(h):
        return (wc.est_channel_factor * W / nch * h ** (5 / 3) * S +
                wc.est_floodplain_factor * Wv / nfp * max(h - dch, 0.0) ** (5 / 3) * S)

    lo, hi = 0.0, 40.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if q(mid) < Q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def storm_rain_total(spec: ScenarioSpec, terrain: Terrain, cfg) -> np.ndarray:
    """Total storm rain depth (m) on the 50 m grid."""
    from .flood import make_storm
    if not spec.has_pluvial:
        return np.zeros_like(terrain.z, dtype=np.float64)
    storm = make_storm(spec, cfg, rngmod.stream(spec.seed, spec.attempt, "flood"))
    c = terrain.cell_size
    xs = (np.arange(terrain.n) + 0.5) * c
    X, Y = np.meshgrid(xs, xs)
    return storm.accumulated(X, Y, storm.t_on + storm.duration)


def estimate_extent(spec: ScenarioSpec, terrain: Terrain, wc: WorldConfig, cfg=None) -> np.ndarray:
    """Cheap HAND-based peak flood extent over the episode window (bool grid): river stage against
    height above the river, plus pluvial ponding by filling depressions with routed storm runoff."""
    from . import hydrology
    from .terrain import FOREST, URBAN
    cfg = cfg or load_config()
    t0 = spec.start_offset_h * 3600.0
    ts = np.linspace(t0, t0 + wc.episode_s + wc.post_episode_s, 200)
    Q = float(hydrograph(spec, ts).max())
    wet = np.zeros_like(terrain.river_mask)
    c = terrain.cell_size
    if spec.has_pluvial:
        from .flood import coarse_grid, downscale_depth, make_storm
        cg = coarse_grid(spec, terrain, cfg)
        dx = cg.cell
        storm = make_storm(spec, cfg, rngmod.stream(spec.seed, spec.attempt, "flood"))
        xs = (np.arange(cg.z.shape[0]) + 0.5) * dx
        X, Y = np.meshgrid(xs, xs)
        runoff = storm.accumulated(X, Y, storm.t_on + storm.duration) * cg.runoff * dx * dx
        zf = hydrology.priority_flood(cg.z)
        down = hydrology.d8_directions(zf)
        pond = hydrology.fill_and_spill(cg.z, zf, runoff, down, dx * dx)
        wet |= downscale_depth(pond.astype(np.float32), cg.z.astype(np.float32), terrain.z, terrain.river_mask) > cfg.flood.wet_depth
        Q += float(runoff.sum()) / (spec.storm_duration_h * 3600.0 + 3600.0)
    h = stage_for_flow(spec, Q, wc)
    wet |= (terrain.river_drain_id >= 0) & (terrain.hand_river < h)
    return wet & ~terrain.river_mask


def rejection_checks(spec: ScenarioSpec, terrain: Terrain, people: dict, wc: WorldConfig, cfg=None) -> list[str]:
    from .presets import knob
    fails = []
    land = ~terrain.river_mask
    wet = estimate_extent(spec, terrain, wc, cfg)
    frac = wet.sum() / max(land.sum(), 1)
    ext = knob(spec, "extent_frac", wc.extent_frac)
    if not (ext[0] * wc.extent_check_margin <= frac <= ext[1] / wc.extent_check_margin):
        fails.append(f"extent {frac:.3f}")
    c = terrain.cell_size
    if wet.any():
        dist = distance_transform_edt(~wet, sampling=c)
        iy = np.clip((people["y"] // c).astype(int), 0, terrain.n - 1)
        ix = np.clip((people["x"] // c).astype(int), 0, terrain.n - 1)
        near = (dist[iy, ix] <= wc.targets_near_flood_m).mean()
    else:
        near = 0.0
    if near < knob(spec, "targets_near_flood_frac", wc.targets_near_flood_frac):
        fails.append(f"targets_near {near:.2f}")
    bx, by = terrain.base_xy
    biy, bix = int(by // c), int(bx // c)
    if wet[biy, bix] or not terrain.road_mask[biy, bix]:
        fails.append("base")
    zb = terrain.z[biy, bix]
    if terrain.z.max() - zb > wc.max_terrain_above_base_m:
        fails.append(f"relief_above_base {terrain.z.max() - zb:.0f}")
    u300 = spec.wind10_ms * math.log(300 / 0.1) / math.log(10 / 0.1)
    if u300 > wc.max_wind_300_ms:
        fails.append("wind")
    if terrain.drain_outlet_frac < wc.min_drain_to_outlet:
        fails.append(f"drainage {terrain.drain_outlet_frac:.2f}")
    k_in = knob(spec, "min_targets_in_flood", 0)
    if k_in:
        iy = np.clip((people["y"] // c).astype(int), 0, terrain.n - 1)
        ix = np.clip((people["x"] // c).astype(int), 0, terrain.n - 1)
        if int(wet[iy, ix].sum()) < k_in:
            fails.append("targets_in_flood")
    if len(terrain.towns) < knob(spec, "min_towns", 0):
        fails.append("towns")
    return fails


def sample_spec(seed: int, bounds: ScenarioBounds | None = None, team: TeamInputs | None = None,
                cfg: Config | None = None, attempt: int | None = None, preset: str | None = None,
                **overrides) -> ScenarioSpec:
    """Sample a scenario spec, resampling with sub-seeds (seed, 1, 2, ...) until every check passes.
    `preset` names a scenario preset (floodsim/presets.py) that narrows the bounds and sets generator knobs."""
    cfg = cfg or load_config()
    b = bounds or cfg.bounds
    if preset:
        from .presets import preset_bounds, preset_tag
        b = preset_bounds(preset, b)
        overrides = {"preset": preset_tag(preset), **overrides}
    team = team or cfg.team
    attempts = [attempt] if attempt is not None else range(cfg.world.max_attempts)
    last = []
    for a in attempts:
        spec = _sample_raw(seed, a, b, team, overrides)
        streams = rngmod.streams(seed, a)
        terrain = generate_terrain(spec, cfg.world, streams["terrain"])
        people = place_people(spec, terrain, streams["people"], cfg.people)
        last = rejection_checks(spec, terrain, people, cfg.world, cfg)
        if not last:
            return spec
    raise ScenarioRejected(f"seed {seed}: no attempt passed the rejection checks (last: {last})")


def sample_seeds(n: int, start: int = 0, bounds: ScenarioBounds | None = None) -> list[int]:
    """Seeds balanced across flood types and swarm sizes (round-robin over the strata)."""
    b = bounds or ScenarioBounds()
    types = list(b.flood_type)
    sizes = list(range(int(b.n_drones.lo), int(b.n_drones.hi) + 1))
    buckets: dict = {}
    seed = start
    out = []
    want = [(ft, sz) for ft in types for sz in sizes]
    k = 0
    while len(out) < n and seed < start + 200 * n + 1000:
        st = strata_of(seed, b)
        buckets.setdefault(st, []).append(seed)
        seed += 1
        while len(out) < n:
            target = want[k % len(want)]
            if buckets.get(target):
                out.append(buckets[target].pop(0))
                k += 1
            else:
                break
    return out


# ----------------------------------------------------------------------------- world
@dataclass
class World:
    spec: ScenarioSpec
    cfg: Config
    terrain: Terrain
    flood: object          # FloodTimeline (truth)
    forecast: object       # FloodForecast (agent-visible)
    weather: object        # Weather (truth)
    people: PeopleState    # truth
    build_info: dict = field(default_factory=dict)

    @property
    def episode_s(self) -> float:
        return self.cfg.world.episode_s


def build_world(spec: ScenarioSpec, cache_dir: str | Path | None = "cache/", cfg: Config | None = None,
                use_cache: bool = True) -> World:
    from .flood import build_flood
    from .forecast import build_forecast
    from .weather import Weather

    cfg = cfg or load_config()
    t0 = time.perf_counter()
    streams = rngmod.streams(spec.seed, spec.attempt)
    terrain = generate_terrain(spec, cfg.world, streams["terrain"])
    placement = place_people(spec, terrain, streams["people"], cfg.people)
    flood, cached = build_flood(spec, terrain, cfg, cache_dir=cache_dir if use_cache else None)
    # re-pick the base if the full solver ever wets it
    _maybe_move_base(terrain, flood, cfg)
    forecast = build_forecast(spec, terrain, flood, cfg, streams["forecast"], cache_dir=cache_dir if use_cache else None)
    weather = Weather(spec, terrain, cfg, streams["weather"])
    people = PeopleState.create(placement, spec, cfg.people)
    info = {"build_s": time.perf_counter() - t0, "flood_cached": cached}
    return World(spec=spec, cfg=cfg, terrain=terrain, flood=flood, forecast=forecast, weather=weather,
                 people=people, build_info=info)


def _maybe_move_base(terrain: Terrain, flood, cfg: Config) -> None:
    c = terrain.cell_size
    bx, by = terrain.base_xy
    maxd = flood.max_depth50()
    biy, bix = int(by // c), int(bx // c)
    if maxd[biy, bix] <= cfg.flood.wet_depth:
        return
    xs = (np.arange(terrain.n) + 0.5) * c
    X, Y = np.meshgrid(xs, xs)
    largest = max(terrain.towns, key=lambda t: t["r"])
    dlt = np.hypot(X - largest["x"], Y - largest["y"]) - largest["r"]
    ok = terrain.road_mask & (maxd <= cfg.flood.wet_depth) & ~terrain.river_mask
    near = ok & (dlt <= cfg.world.base_town_dist_m)
    cand = near if near.any() else ok
    if not cand.any():
        cand = (maxd <= cfg.flood.wet_depth) & ~terrain.river_mask
    idx = int(np.argmax(np.where(cand, terrain.z, -np.inf)))
    terrain.base_xy = (float(xs[idx % terrain.n]), float(xs[idx // terrain.n]))


def write_seed_split(path: str | Path, seeds: list[int], cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    rec = {}
    for s in seeds:
        spec = sample_spec(s, cfg=cfg)
        rec[str(s)] = {"attempt": spec.attempt, "spec_hash": spec.spec_hash}
    Path(path).write_text(json.dumps(rec, indent=1))
    return rec


def spec_from_split(seed: int, split: dict, cfg: Config | None = None) -> ScenarioSpec:
    cfg = cfg or load_config()
    a = split[str(seed)]["attempt"]
    return sample_spec(seed, cfg=cfg, attempt=a)
