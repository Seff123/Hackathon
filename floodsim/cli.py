"""Command-line interface: python -m floodsim <command>."""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="floodsim", description="FloodSim glider SAR environment")
    p.add_subparsers(dest="cmd")
    return p


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if not args.cmd:
        p.print_help()
    return 0
