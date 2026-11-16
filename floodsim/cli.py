"""Command line: scenario, build-cache, demo, evaluate."""

from __future__ import annotations

import argparse
import sys


def parse_seeds(text: list[str] | str) -> list[int]:
    """'0-49' or '0-199 10000-10099' or '3' -> list of ints."""
    parts = text.split() if isinstance(text, str) else [p for t in text for p in t.split(",")]
    out: list[int] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return out


def _add_team(p: argparse.ArgumentParser):
    g = p.add_argument_group("team inputs (override configs/default.toml)")
    g.add_argument("--mass-kg", type=float)
    g.add_argument("--battery-life-min", type=float)
    g.add_argument("--battery-wh", type=float)
    g.add_argument("--charge-time-min", type=float)
    g.add_argument("--signal-radius-km", type=float)
    g.add_argument("--camera-range-m", type=float)
    p.add_argument("--config", default=None, help="TOML config (default: configs/default.toml if present)")
    p.add_argument("--cache-dir", default="cache")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="floodsim", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scenario", help="sample and build one scenario, optionally plot it")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", default=None, metavar="DIR")
    _add_team(p)

    p = sub.add_parser("build-cache", help="build and cache worlds for a seed list")
    p.add_argument("--seeds", nargs="+", default=["0-199"])
    p.add_argument("--split", default=None, help="name for seeds_<split>.json")
    p.add_argument("--workers", type=int, default=0, help="0 = cpu count")
    _add_team(p)

    p = sub.add_parser("demo", help="run one episode, write a GIF and a 3D HTML view")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drones", type=int, default=None)
    p.add_argument("--out", default="out")
    _add_team(p)

    p = sub.add_parser("evaluate", help="run a controller over many seeds")
    p.add_argument("--controller", default="scripted")
    p.add_argument("--seeds", nargs="+", default=["0-49"])
    p.add_argument("--drones", type=int, default=None)
    p.add_argument("--out", default="results")
    p.add_argument("--workers", type=int, default=0)
    _add_team(p)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from . import commands

    return int(commands.run(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
