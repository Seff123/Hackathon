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

    from . import scenario as S
    from . import world as W

    spec = sample_spec(seed, cfg=cfg)
    w = build_world(spec, cfg, cache_dir=cache_dir)
    stats = dict(w.stats)
    W._MEM.clear()  # keep worker memory flat (8 GB machines)
    W._FLOOD.clear()
    S._TERRAIN_CACHE.clear()
    return seed, spec.attempt, spec.spec_hash, stats


def cmd_build_cache(args, cfg: Config) -> int:
    import time
    from concurrent.futures import ProcessPoolExecutor

    seeds = parse_seeds(args.seeds)
    workers = args.workers or min(os.cpu_count() or 1, 4)
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


def cmd_demo(args, cfg: Config) -> int:
    import time
    from dataclasses import replace

    from . import viz
    from .controllers import ScriptedSweep
    from .scenario import sample_spec
    from .sim import Sim
    from .world import build_world

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    spec = sample_spec(args.seed, cfg=cfg)
    world = build_world(spec, cfg, cache_dir=args.cache_dir)
    print(f"world for seed {args.seed} ({spec.flood_type}, peak extent {world.peak_extent():.1%}) "
          f"in {time.perf_counter() - t0:.1f} s")
    sc = replace(cfg.sim, fast=False, record_grids=True)
    res = Sim(world, ScriptedSweep(cfg.controller), sc, n_drones=args.drones, cfg=cfg).run()
    print(f"episode: {res.n_drones} drones in {res.wall_s:.1f} s")
    report = viz.episode_report(res)
    print(report)
    tag = f"seed{args.seed}"
    (out / f"report_{tag}.txt").write_text(report)
    (out / f"metrics_{tag}.json").write_text(json.dumps(res.metrics, indent=1))
    paths = [
        viz.plot_world(world, 0.0, out / f"world_{tag}.png"),
        viz.plot_coverage(res, world, None, out / f"coverage_{tag}.png"),
        viz.plot_wind_slice(world, 100.0, 0.0, out / f"wind_{tag}.png"),
        viz.plot3d_html(res, world, out / f"episode3d_{tag}.html"),
        viz.animate_episode(res, world, out / f"episode_{tag}.gif"),
    ]
    for p in paths:
        print(f"wrote {p}")
    return 0


def cmd_evaluate(args, cfg: Config) -> int:
    from .evaluate import evaluate

    seeds = parse_seeds(args.seeds)
    out, summary = evaluate(seeds, cfg, controller=args.controller, n_drones=args.drones, out_dir=args.out,
                            workers=args.workers, cache_dir=args.cache_dir)
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}/episodes.csv, scenarios.csv, summary.json")
    return 0


COMMANDS = {"scenario": cmd_scenario, "build-cache": cmd_build_cache, "demo": cmd_demo, "evaluate": cmd_evaluate}


def run(args) -> int:
    cfg = make_config(args)
    fn = COMMANDS.get(args.cmd)
    if fn is None:
        raise SystemExit(f"command {args.cmd!r} is not built yet")
    return fn(args, cfg)
