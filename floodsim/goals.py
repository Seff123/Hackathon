"""Goals a controller can give a drone. The autopilot turns them into commands."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(kw_only=True)
class Goal:
    allow_soaring: bool = False


@dataclass(kw_only=True)
class Transit(Goal):
    x: float
    y: float
    band: tuple | None = None  # (lo, hi) metres above the surface; None = the drone's own layer


@dataclass(kw_only=True)
class Search(Goal):
    legs: np.ndarray  # (k, 2, 2): k legs of (start xy, end xy), flown in order
    agl: float  # height above the surface
    speed: float  # airspeed (m/s)
    start_leg: int = 0


@dataclass(kw_only=True)
class Verify(Goal):
    x: float
    y: float
    cell: int  # flat 50 m cell index being verified


@dataclass(kw_only=True)
class RTB(Goal):
    pass


@dataclass(kw_only=True)
class Loiter(Goal):
    x: float
    y: float
    agl: float = 120.0


def lawnmower_legs(x0, y0, length, width, spacing, angle_rad):
    """Legs for a rotated box: origin corner (x0, y0), legs along `angle_rad`."""
    ux, uy = np.cos(angle_rad), np.sin(angle_rad)
    px, py = -uy, ux
    n = max(int(np.ceil(width / spacing)), 1)
    legs = []
    for k in range(n):
        off = (k + 0.5) * width / n
        a = (x0 + px * off, y0 + py * off)
        b = (a[0] + ux * length, a[1] + uy * length)
        legs.append((a, b) if k % 2 == 0 else (b, a))
    return np.asarray(legs, float)
