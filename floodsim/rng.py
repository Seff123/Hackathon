"""Named, independent random streams derived from one seed (SeedSequence.spawn).

Changing how one subsystem draws numbers never changes another subsystem's draws.
"""
from __future__ import annotations

import numpy as np

STREAMS = ("spec", "terrain", "hydro", "flood", "weather", "people", "sensor", "forecast", "sim")


def streams(seed: int, attempt: int = 0) -> dict[str, np.random.Generator]:
    ss = np.random.SeedSequence([int(seed), int(attempt)])
    children = ss.spawn(len(STREAMS))
    return {name: np.random.default_rng(c) for name, c in zip(STREAMS, children)}


def stream(seed: int, attempt: int, name: str) -> np.random.Generator:
    return streams(seed, attempt)[name]


def child(rng_seed: int, *keys: int) -> np.random.Generator:
    """Deterministic generator for an arbitrary key path (e.g. episode-level streams)."""
    return np.random.default_rng(np.random.SeedSequence([int(rng_seed), *[int(k) for k in keys]]))
