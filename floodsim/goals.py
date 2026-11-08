"""Goals the controller hands to the autopilot."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Goal:
    allow_soaring: bool = False

    @property
    def kind(self) -> str:
        return type(self).__name__


@dataclass
class Transit(Goal):
    xy: tuple = (0.0, 0.0)
    alt_band: tuple | None = None       # (lo, hi) height above the surface; None = drone's layer


@dataclass
class Search(Goal):
    box: tuple | None = None            # (x0, y0, x1, y1)
    agl: float = 84.0                   # height above the surface
    spacing: float | None = None
    speed: float | None = None
    leg_angle_deg: float | None = None  # leg direction relative to the wind direction (90 = crosswind)
    waypoints: list | None = None       # explicit pattern [(x, y), ...] (overrides box)
    start_index: int = 0
    tag: str = ""


@dataclass
class Verify(Goal):
    cell: int = 0                       # flat 50 m cell index
    xy: tuple | None = None


@dataclass
class RTB(Goal):
    pass
