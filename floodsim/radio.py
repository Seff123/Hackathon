"""Range-only radio: every message within signal radius arrives at the next tick.

The environment only carries messages; what agents send is up to them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BASE_ID = -1


@dataclass(frozen=True)
class Message:
    sender: int
    t: float
    payload: object
    to: int | None = None  # None = broadcast to everyone in range


class Radio:
    def __init__(self, signal_radius_m: float, perfect: bool = False):
        self.r = float(signal_radius_m)
        self.perfect = perfect
        self.queue: list[Message] = []
        self.delivered = 0
        self.dropped = 0

    def send(self, sender: int, payload, t: float, to: int | None = None):
        self.queue.append(Message(sender, t, payload, to))

    def in_range(self, a_xy, b_xy) -> bool:
        return self.perfect or float(np.hypot(a_xy[0] - b_xy[0], a_xy[1] - b_xy[1])) <= self.r

    def deliver(self, nodes: dict[int, tuple]) -> dict[int, list[Message]]:
        """nodes: id -> (x, y) of every node that can talk now (drones and BASE_ID)."""
        inbox: dict[int, list[Message]] = {k: [] for k in nodes}
        for m in self.queue:
            if m.sender not in nodes:
                self.dropped += 1
                continue
            sxy = nodes[m.sender]
            targets = [m.to] if m.to is not None else [k for k in nodes if k != m.sender]
            got = False
            for k in targets:
                if k in nodes and k != m.sender and self.in_range(sxy, nodes[k]):
                    inbox[k].append(m)
                    got = True
                    self.delivered += 1
            if not got:
                self.dropped += 1
        self.queue.clear()
        return inbox
