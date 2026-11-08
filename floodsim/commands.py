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


COMMANDS = {"scenario": cmd_scenario}


def run(args) -> int:
    cfg = make_config(args)
    fn = COMMANDS.get(args.cmd)
    if fn is None:
        raise SystemExit(f"command {args.cmd!r} is not built yet")
    return fn(args, cfg)
