"""Command-line interface: python -m floodsim <command>."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _team_args(p):
    g = p.add_argument_group("team inputs")
    g.add_argument("--mass-kg", type=float)
    g.add_argument("--battery-life-min", type=float)
    g.add_argument("--battery-wh", type=float)
    g.add_argument("--charge-time-min", type=float)
    g.add_argument("--signal-radius-km", type=float)
    g.add_argument("--camera-range-m", type=float)
    p.add_argument("--config", help="TOML config (default configs/default.toml)")
    p.add_argument("--cache", default="cache/", help="world cache directory")


def _team(args) -> dict:
    return {k: getattr(args, k, None) for k in ("mass_kg", "battery_life_min", "battery_wh", "charge_time_min",
                                                 "signal_radius_km", "camera_range_m")}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="floodsim", description="FloodSim glider SAR environment")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("scenario", help="sample one scenario, print it and optionally plot it")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--plot", help="output directory for PNGs")
    _team_args(s)

    s = sub.add_parser("build-cache", help="build and cache worlds for seed ranges")
    s.add_argument("--seeds", nargs="+", default=["0-199"])
    s.add_argument("--split", default=None, help="name for seeds_<split>.json (default: from the seed range)")
    s.add_argument("--workers", type=int, default=None)
    _team_args(s)

    s = sub.add_parser("demo", help="one recorded episode: GIF, 3D HTML, plots and report")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--drones", type=int, default=None)
    s.add_argument("--controller", default="scripted")
    s.add_argument("--out", default="out/")
    _team_args(s)

    s = sub.add_parser("evaluate", help="run a controller over many seeds and write metrics")
    s.add_argument("--controller", default="scripted")
    s.add_argument("--seeds", nargs="+", default=["0-49"])
    s.add_argument("--drones", type=int, default=None)
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--out", default="results/")
    _team_args(s)

    try:
        from .planner.experiments import add_cli
        add_cli(sub, _team_args)
    except ImportError:
        pass
    try:
        from .vizpage.cli import add_cli as add_viz
        add_viz(sub, _team_args)
    except ImportError:
        pass
    return p


def _worker_build(args):
    seed, config, team, cache = args
    from .runner import make_config
    from .scenario import build_world, sample_spec
    cfg = make_config(config, team)
    t0 = time.perf_counter()
    spec = sample_spec(seed, cfg=cfg)
    w = build_world(spec, cache_dir=cache, cfg=cfg)
    return seed, spec.attempt, spec.spec_hash, round(time.perf_counter() - t0, 2), w.build_info.get("flood_cached")


def cmd_scenario(args):
    from .runner import make_config
    from .scenario import build_world, sample_spec
    cfg = make_config(args.config, _team(args))
    spec = sample_spec(args.seed, cfg=cfg)
    print(json.dumps(spec.to_dict(), indent=1, default=float))
    print("spec_hash", spec.spec_hash)
    if args.plot:
        from . import viz
        out = Path(args.plot)
        out.mkdir(parents=True, exist_ok=True)
        w = build_world(spec, cache_dir=args.cache, cfg=cfg)
        viz.plot_world(w, 0.0, out / f"world_{args.seed}_t0.png")
        viz.plot_world(w, cfg.world.episode_s / 2, out / f"world_{args.seed}_mid.png")
        viz.plot_wind_slice(w, 150.0, 0.0, out / f"wind_{args.seed}.png")
        print("brier", w.forecast.brier, "mass residual", w.flood.mass.get("residual_frac"))
        print("wrote", out)
    return 0


def cmd_build_cache(args):
    from concurrent.futures import ProcessPoolExecutor
    from .runner import parse_seeds
    seeds = parse_seeds(args.seeds)
    team = _team(args)
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    rec = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (seed, att, h, dt, cached) in enumerate(ex.map(_worker_build, [(s, args.config, team, args.cache) for s in seeds])):
            rec[str(seed)] = {"attempt": att, "spec_hash": h}
            if (k + 1) % max(1, len(seeds) // 10) == 0:
                print(f"  {k + 1}/{len(seeds)} worlds, {time.perf_counter() - t0:.0f} s", flush=True)
    split = args.split or "_".join(a.replace(",", "_") for a in args.seeds)
    Path(args.cache).mkdir(parents=True, exist_ok=True)
    path = Path(args.cache) / f"seeds_{split}.json"
    path.write_text(json.dumps(rec, indent=1))
    print(f"built {len(seeds)} worlds in {time.perf_counter() - t0:.0f} s; wrote {path}")
    return 0


def cmd_demo(args):
    from . import viz
    from .metrics import episode_report
    from .runner import run_episode
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    res, world = run_episode(args.seed, args.controller, args.drones, args.config, _team(args), args.cache,
                             fast=False, record=True)
    print(episode_report(res))
    tag = f"{args.seed}_{res.controller}"
    viz.plot_world(world, 0.0, out / f"world_{tag}.png")
    seen = res.recording["frames"][-1]["seen"].reshape(world.terrain.n, world.terrain.n) if res.recording else None
    viz.plot_coverage(res, world, None, out / f"coverage_{tag}.png", seen=seen)
    viz.plot_wind_slice(world, 150.0, 0.0, out / f"wind_{tag}.png")
    gif = viz.animate_episode(res, world, out / f"episode_{tag}.gif")
    html = viz.plot3d_html(res, world, out / f"flight3d_{tag}.html")
    (out / f"report_{tag}.txt").write_text(episode_report(res) + "\n")
    m = {k: v for k, v in res.metrics.items()}
    (out / f"metrics_{tag}.json").write_text(json.dumps(m, indent=1, default=float))
    print(f"wrote {gif} and {html} ({time.perf_counter() - t0:.0f} s)")
    return 0


def cmd_evaluate(args):
    from .runner import parse_seeds, run_batch, scenario_statistics, write_rows
    seeds = parse_seeds(args.seeds)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) / f"eval_{args.controller}_{ts}"
    rows = run_batch(seeds, args.controller, args.drones, args.config, _team(args), args.cache, args.workers)
    write_rows(rows, out / "episodes.csv")
    stats = scenario_statistics(rows)
    (out / "scenario_stats.json").write_text(json.dumps(stats, indent=1, default=float))
    ok = [r for r in rows if "error" not in r]
    import numpy as np
    sar = np.array([r["survival_at_risk"] for r in ok])
    print(f"{len(ok)} episodes ({len(rows) - len(ok)} errors). survival_at_risk {sar.mean():.3f} ± {sar.std() / max(len(sar), 1) ** 0.5:.3f}")
    print(f"extent {stats['peak_extent_frac']['mean']:.3f} (p10 {stats['peak_extent_frac']['p10']:.3f}, p90 {stats['peak_extent_frac']['p90']:.3f}); "
          f"wind {stats['wind10_ms']['mean']:.1f} m/s; thermals {stats['n_thermals']['mean']:.1f}; "
          f"detections true/false per frame {stats['detection_rates']['true_per_frame']:.4f}/{stats['detection_rates']['false_per_frame']:.4f}")
    print("wrote", out)
    return 0


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if not args.cmd:
        p.print_help()
        return 0
    fn = {"scenario": cmd_scenario, "build-cache": cmd_build_cache, "demo": cmd_demo, "evaluate": cmd_evaluate}.get(args.cmd)
    if fn is None and hasattr(args, "func"):
        return args.func(args)
    return fn(args)
