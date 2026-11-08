"""CLI command implementations."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .cli import parse_seeds
from .config import Config, load_config


def make_config(args) -> Config:
    path = args.config
    if path is None and Path("configs/default.toml").exists():
        path = "configs/default.toml"
    team = {
        "mass_kg": args.mass_kg, "battery_life_min": args.battery_life_min,
        "battery_wh": args.battery_wh, "charge_time_min": args.charge_time_min,
        "signal_radius_km": args.signal_radius_km, "camera_range_m": args.camera_range_m,
    }
    return load_config(path, team=team)


def cmd_scenario(args, cfg: Config) -> int:
    from .scenario import sample_spec, terrain_for

    spec = sample_spec(args.seed, cfg=cfg)
    print(json.dumps(spec.to_dict(), indent=1))
    print(f"spec_hash {spec.spec_hash}")
    world = None
    try:
        from .world import build_world

        world = build_world(spec, cfg, cache_dir=args.cache_dir)
    except ImportError:
        pass
    if args.plot:
        from . import viz

        out = Path(args.plot)
        out.mkdir(parents=True, exist_ok=True)
        terr, _ = terrain_for(spec, cfg)
        p = viz.plot_terrain(terr, out / f"scenario_{args.seed}_terrain.png",
                             f"seed {args.seed}: {spec.flood_type}, relief {spec.relief_m:.0f} m")
        print(f"wrote {p}")
        if world is not None:
            for p in viz.scenario_plots(world, out, args.seed):
                print(f"wrote {p}")
    return 0


def _build_one(args):
    seed, cfg, cache_dir = args
    from .scenario import sample_spec
    from .world import build_world

    spec = sample_spec(seed, cfg=cfg)
    w = build_world(spec, cfg, cache_dir=cache_dir)
    return seed, spec.attempt, spec.spec_hash, w.stats


def cmd_build_cache(args, cfg: Config) -> int:
    import time
    from concurrent.futures import ProcessPoolExecutor

    seeds = parse_seeds(args.seeds)
    workers = args.workers or os.cpu_count() or 1
    t0 = time.perf_counter()
    out = {}
    jobs = [(s, cfg, args.cache_dir) for s in seeds]
    if workers <= 1:
        results = map(_build_one, jobs)
    else:
        ex = ProcessPoolExecutor(workers)
        results = ex.map(_build_one, jobs)
    for i, (seed, attempt, h, stats) in enumerate(results):
        out[str(seed)] = {"attempt": attempt, "spec_hash": h, "peak_extent": round(stats.get("peak_extent", 0), 4)}
        print(f"[{i + 1}/{len(seeds)}] seed {seed} attempt {attempt} extent {out[str(seed)]['peak_extent']:.3f}", flush=True)
    split = args.split or (args.seeds[0] if len(args.seeds) == 1 else "custom")
    p = Path(args.cache_dir) / f"seeds_{split}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p} ({time.perf_counter() - t0:.1f} s)")
    return 0


COMMANDS = {"scenario": cmd_scenario, "build-cache": cmd_build_cache}


def run(args) -> int:
    cfg = make_config(args)
    fn = COMMANDS.get(args.cmd)
    if fn is None:
        raise SystemExit(f"command {args.cmd!r} is not built yet")
    return fn(args, cfg)
