"""Named, independent random streams derived from a scenario seed.

Each stream gets a fixed integer id, so adding a stream or changing how many
draws one subsystem makes never shifts another subsystem's draws.
"""

from __future__ import annotations

import numpy as np

STREAM_IDS = {
    "spec": 1,
    "terrain": 2,
    "hydro": 3,
    "flood": 4,
    "weather": 5,
    "people": 6,
    "sensor": 7,
    "forecast": 8,
    "sim": 9,
}


def stream(seed: int, name: str, attempt: int = 0, *extra: int) -> np.random.Generator:
    """Return the generator for one named stream of (seed, attempt)."""
    if name not in STREAM_IDS:
        raise KeyError(f"unknown RNG stream {name!r}")
    ss = np.random.SeedSequence([int(seed), int(attempt), STREAM_IDS[name], *map(int, extra)])
    return np.random.default_rng(ss)


def streams(seed: int, attempt: int = 0) -> dict[str, np.random.Generator]:
    """All named streams for one scenario attempt."""
    return {name: stream(seed, name, attempt) for name in STREAM_IDS}
