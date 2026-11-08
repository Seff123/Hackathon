"""Named, independent random streams derived from one seed.

Each subsystem gets its own child stream, so changing how many draws one
subsystem makes never changes another subsystem's draws.
"""
from __future__ import annotations

import numpy as np

STREAMS = ("spec", "terrain", "hydro", "flood", "weather", "people", "sensor", "forecast", "sim")


class Streams:
    def __init__(self, seed: int, attempt: int = 0):
        self.seed = int(seed)
        self.attempt = int(attempt)
        root = np.random.SeedSequence([self.seed, self.attempt])
        children = root.spawn(len(STREAMS))
        self._seq = dict(zip(STREAMS, children))
        self._gens: dict[str, np.random.Generator] = {}

    def __getitem__(self, name: str) -> np.random.Generator:
        if name not in self._gens:
            self._gens[name] = np.random.default_rng(self._seq[name])
        return self._gens[name]

    def fresh(self, name: str, *sub: int) -> np.random.Generator:
        """A new generator for `name`, optionally keyed further (e.g. per drone)."""
        ss = np.random.SeedSequence(entropy=self._seq[name].entropy,
                                    spawn_key=self._seq[name].spawn_key + tuple(int(s) for s in sub) + (7919,))
        return np.random.default_rng(ss)
