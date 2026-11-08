"""World: everything built from a spec (truth and agent-visible products), cached."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from .config import Config, ScenarioSpec
from .rng import stream

CACHE_VERSION = 2
_MEM: dict[str, "World"] = {}
_FLOOD: dict[str, "World"] = {}  # flood-only worlds built by the sampler's solver check


@dataclass
class World:
    spec: ScenarioSpec
    cfg: Config
    terrain: object
    flood: object  # FloodTimeline (truth)
    storm: object
    people0: object  # initial Placement (truth)
    forecast: object | None = None
    weather: object | None = None
    build_s: float = 0.0
    from_cache: bool = False
    stats: dict = field(default_factory=dict)

    @property
    def episode_s(self) -> float:
        return self.cfg.sim.episode_s or self.spec.episode_s

    def peak_extent(self) -> float:
        terr = self.terrain
        md = self.flood.max_depth(0.0, self.flood.t_end)
        land = terr.land
        return float(((md > self.cfg.world.wet_depth_m) & land).sum() / max(land.sum(), 1))


def _cache_path(spec: ScenarioSpec, cache_dir) -> Path:
    return Path(cache_dir) / f"{spec.spec_hash}.npz"


def _save(world: World, path: Path):
    from .flood import FloodTimeline

    fl: FloodTimeline = world.flood
    arrays = {
        "depth_snap": fl.depth_snap, "u_snap": fl.u_snap, "v_snap": fl.v_snap,
        "z_coarse": fl.z_coarse.astype(np.float32),
    }
    meta = {
        "version": CACHE_VERSION, "world_key": world.cfg.world_key(), "spec": world.spec.to_dict(),
        "dt_depth": fl.dt_depth, "dt_vel": fl.dt_vel, "cell": fl.cell, "fine_factor": fl.fine_factor,
        "mass_residual": fl.mass_residual, "breach_cell": fl.breach_cell, "breach_t": fl.breach_t,
        "base_xy": list(world.terrain.base_xy), "stats": world.stats,
    }
    if world.forecast is not None:
        arrays.update({f"fc_{k}": v for k, v in world.forecast.to_arrays().items()})
    if world.weather is not None:
        arrays.update({f"wx_{k}": v for k, v in world.weather.to_arrays().items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
    tmp.replace(path)


def _load(spec: ScenarioSpec, cfg: Config, path: Path) -> World | None:
    from .flood import FloodTimeline
    from .scenario import terrain_for

    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("version") != CACHE_VERSION or meta.get("world_key") != cfg.world_key():
                return None
            arrs = {k: z[k] for k in z.files if k != "meta"}
    except (OSError, ValueError, KeyError):
        return None
    terr, people = terrain_for(spec, cfg)
    terr.set_base(tuple(meta["base_xy"]), cfg.world.base_clear_radius_m)
    fl = FloodTimeline(
        depth_snap=arrs["depth_snap"], u_snap=arrs["u_snap"], v_snap=arrs["v_snap"],
        dt_depth=meta["dt_depth"], dt_vel=meta["dt_vel"], cell=meta["cell"],
        z_coarse=arrs["z_coarse"], z_fine=terr.z.astype(np.float32), fine_factor=meta["fine_factor"],
        mass_residual=meta["mass_residual"],
        breach_cell=tuple(meta["breach_cell"]) if meta["breach_cell"] else None, breach_t=meta["breach_t"],
    )
    from .flood import make_storm

    storm = make_storm(spec, terr, cfg, stream(spec.seed, "flood", spec.attempt))
    w = World(spec=spec, cfg=cfg, terrain=terr, flood=fl, storm=storm, people0=people,
              from_cache=True, stats=meta.get("stats", {}))
    fc = {k[3:]: v for k, v in arrs.items() if k.startswith("fc_")}
    wx = {k[3:]: v for k, v in arrs.items() if k.startswith("wx_")}
    _attach_products(w, fc or None, wx or None)
    if (w.forecast is not None and not fc) or (w.weather is not None and not wx):
        _save(w, path)  # cache predates a product module: add the product
    return w


def _attach_products(world: World, fc_arrays=None, wx_arrays=None):
    """Build (or restore) the forecast and weather once those modules exist."""
    try:
        from .forecast import FloodForecast, build_forecast
    except ImportError:
        FloodForecast = None
    if FloodForecast is not None:
        world.forecast = (FloodForecast.from_arrays(fc_arrays, world.terrain, world.cfg)
                          if fc_arrays else build_forecast(world))
    try:
        from .weather import Weather, build_weather
    except ImportError:
        Weather = None
    if Weather is not None:
        world.weather = (Weather.from_arrays(wx_arrays, world.terrain, world.spec, world.cfg, storm=world.storm)
                         if wx_arrays else build_weather(world.spec, world.terrain, world.cfg,
                                                         stream(world.spec.seed, "weather", world.spec.attempt),
                                                         storm=world.storm))


def flood_world(spec: ScenarioSpec, cfg: Config) -> World:
    """Terrain + flood truth for a spec; re-picks the base if the flood wets it."""
    from .flood import run_flood
    from .scenario import terrain_for
    from .terrain import pick_base

    t0 = time.perf_counter()
    terr, people = terrain_for(spec, cfg)
    fl, storm = run_flood(spec, terr, cfg, stream(spec.seed, "flood", spec.attempt))
    wet = fl.max_depth(0.0, fl.t_end) > 0.05
    bx, by = terr.base_xy
    iy, ix = int(by // terr.cell_size), int(bx // terr.cell_size)
    moved = False
    if wet[iy, ix]:
        base = pick_base(terr, cfg.world, excluded=ndimage.binary_dilation(wet, iterations=2))
        terr.set_base(base, cfg.world.base_clear_radius_m)
        moved = True
    w = World(spec=spec, cfg=cfg, terrain=terr, flood=fl, storm=storm, people0=people)
    w.stats = {"peak_extent": w.peak_extent(), "mass_residual": fl.mass_residual, "base_moved": moved,
               "solver_s": time.perf_counter() - t0}
    return w


def build_world(spec: ScenarioSpec, cfg: Config | None = None, cache_dir="cache", use_cache=True) -> World:
    cfg = cfg or Config()
    key = spec.spec_hash + cfg.world_key()
    if use_cache and key in _MEM:
        return _MEM[key]
    t0 = time.perf_counter()
    path = _cache_path(spec, cache_dir) if cache_dir else None
    w = _load(spec, cfg, path) if (use_cache and path is not None and path.exists()) else None
    if w is None:
        w = _FLOOD.pop(key, None) or flood_world(spec, cfg)
        _attach_products(w)
        if path is not None:
            _save(w, path)
    w.build_s = time.perf_counter() - t0
    _MEM[key] = w
    while len(_MEM) > 3:
        _MEM.pop(next(iter(_MEM)))
    return w
