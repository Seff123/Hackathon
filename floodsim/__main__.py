"""Command-line interface: scenario, build-cache, demo, evaluate."""
from __future__ import annotations

import argparse
import sys


def parse_seeds(text: list[str] | str) -> list[int]:
    """'0-49' -> [0..49]; several ranges or numbers may be given."""
    items = text if isinstance(text, list) else [text]
    out: list[int] = []
    for it in items:
        for part in str(it).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-")
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
    return out


def _team_args(p):
    p.add_argument("--config", default="configs/default.toml")
    p.add_argument("--mass-kg", type=float)
    p.add_argument("--battery-life-min", type=float)
    p.add_argument("--battery-wh", type=float)
    p.add_argument("--charge-time-min", type=float)
    p.add_argument("--signal-radius-km", type=float)
    p.add_argument("--camera-range-m", type=float)
    p.add_argument("--cache", default="cache")


def _config(args):
    from .config import load_config
    ov = {
        "team.mass_kg": args.mass_kg, "team.battery_life_min": args.battery_life_min,
        "team.battery_wh": args.battery_wh, "team.charge_time_min": args.charge_time_min,
        "team.signal_radius_km": args.signal_radius_km, "team.camera_range_m": args.camera_range_m,
    }
    return load_config(args.config, ov)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="floodsim", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scenario", help="sample one scenario and optionally plot it")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", default=None, help="output directory for plots")
    p.add_argument("--flood-gif", action="store_true", help="also write a flood-only GIF")
    _team_args(p)

    p = sub.add_parser("build-cache", help="build and cache worlds for seeds")
    p.add_argument("--seeds", nargs="+", default=["0-199"])
    p.add_argument("--split", default=None)
    p.add_argument("--workers", type=int, default=0)
    _team_args(p)

    p = sub.add_parser("demo", help="run and animate one episode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drones", type=int, default=None)
    p.add_argument("--out", default="out")
    p.add_argument("--no-gif", action="store_true")
    _team_args(p)

    p = sub.add_parser("evaluate", help="run a controller on many seeds")
    p.add_argument("--controller", default="scripted")
    p.add_argument("--seeds", nargs="+", default=["0-49"])
    p.add_argument("--out", default="results")
    p.add_argument("--workers", type=int, default=0)
    _team_args(p)

    args = ap.parse_args(argv)
    from . import cli
    return getattr(cli, "cmd_" + args.cmd.replace("-", "_"))(args, _config(args))


if __name__ == "__main__":
    sys.exit(main())
