"""Range-only radio: messages between drones (and the base) within the signal radius arrive at once."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BASE = -1


@dataclass
class Message:
    t: float
    sender: int
    payload: object
    recipient: int | None = None     # None = broadcast


class Radio:
    def __init__(self, n: int, radius_m: float, base_xy, perfect: bool = False):
        self.n = n
        self.radius = float(radius_m)
        self.base = np.asarray(base_xy, dtype=float)
        self.perfect = perfect
        self.queue: list[Message] = []
        self.inbox: dict[int, list[Message]] = {i: [] for i in range(n)}
        self.inbox[BASE] = []
        self.delivered = 0

    def send(self, t: float, sender: int, payload, recipient: int | None = None) -> None:
        self.queue.append(Message(t, sender, payload, recipient))

    def in_range(self, pos: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
        """Adjacency [n+1, n+1]; index n is the base. Inactive (crashed) drones have no links."""
        pts = np.vstack([pos[:, :2], self.base[None]])
        d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
        adj = (d <= self.radius) | self.perfect
        if active is not None:
            act = np.append(active, True)
            adj &= act[:, None] & act[None, :]
        np.fill_diagonal(adj, False)
        return adj

    def connected_to_base(self, pos: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
        """Multi-hop connectivity of each drone to the base."""
        adj = self.in_range(pos, active)
        n = self.n
        reach = np.zeros(n + 1, dtype=bool)
        reach[n] = True
        frontier = [n]
        while frontier:
            k = frontier.pop()
            for j in np.flatnonzero(adj[k] & ~reach):
                reach[j] = True
                frontier.append(int(j))
        return reach[:n]

    def deliver(self, pos: np.ndarray, t: float, active: np.ndarray | None = None) -> int:
        """Deliver every queued message to all nodes in direct range of the sender."""
        if not self.queue:
            return 0
        adj = self.in_range(pos, active)
        count = 0
        for m in self.queue:
            s = self.n if m.sender == BASE else m.sender
            targets = np.flatnonzero(adj[s])
            for j in targets.tolist():
                rid = BASE if j == self.n else j
                if m.recipient is not None and m.recipient != rid:
                    continue
                self.inbox[rid].append(m)
                count += 1
        self.queue.clear()
        self.delivered += count
        return count

    def take(self, node: int) -> list[Message]:
        msgs = self.inbox[node]
        self.inbox[node] = []
        return msgs
