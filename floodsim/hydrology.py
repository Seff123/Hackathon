"""Depression filling, D8 flow, accumulation, drainage network and HAND."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

# D8 neighbour offsets (dy, dx)
D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def priority_flood(z: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Priority-flood + epsilon (Barnes et al. 2014). All domain edges are outlets.

    Returns z_fill (float64) in which every interior cell has a strictly lower
    neighbour, so D8 descent always reaches an edge.
    """
    ny, nx = z.shape
    W = nx + 2
    zp = np.full((ny + 2, W), -np.inf)
    zp[1:-1, 1:-1] = z
    zf = zp.ravel().tolist()
    closed = np.zeros((ny + 2, W), bool)
    closed[0, :] = closed[-1, :] = closed[:, 0] = closed[:, -1] = True
    heap: list[tuple[float, int]] = []
    edge = np.zeros((ny, nx), bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    ey, ex = np.nonzero(edge)
    for a, b in zip(ey, ex):
        idx = (a + 1) * W + (b + 1)
        closed[a + 1, b + 1] = True
        heap.append((zf[idx], idx))
    heapq.heapify(heap)
    cl = closed.ravel().tolist()
    offs = [dy * W + dx for dy, dx in D8]
    pop, push = heapq.heappop, heapq.heappush
    while heap:
        zc, c = pop(heap)
        for o in offs:
            nb = c + o
            if cl[nb]:
                continue
            cl[nb] = True
            v = zf[nb]
            if v <= zc:
                v = zc + eps
                zf[nb] = v
            push(heap, (v, nb))
    return np.asarray(zf).reshape(ny + 2, W)[1:-1, 1:-1].copy()


def d8_receivers(zf: np.ndarray, cell: float) -> np.ndarray:
    """Flat index of each cell's steepest-descent neighbour; -1 for edge outlets."""
    ny, nx = zf.shape
    big = np.full((ny + 2, nx + 2), np.inf)
    big[1:-1, 1:-1] = zf
    best = np.zeros((ny, nx))
    rec = np.full((ny, nx), -1, np.int64)
    iy, ix = np.mgrid[0:ny, 0:nx]
    for dy, dx in D8:
        nb = big[1 + dy:ny + 1 + dy, 1 + dx:nx + 1 + dx]
        drop = (zf - nb) / (cell * (np.sqrt(2.0) if dy and dx else 1.0))
        better = drop > best
        best = np.where(better, drop, best)
        rec = np.where(better, (iy + dy) * nx + (ix + dx), rec)
    # edge cells without a lower neighbour drain off the map
    return rec


@dataclass
class Hydrology:
    z_fill: np.ndarray
    depression_depth: np.ndarray
    flow_dir: np.ndarray  # receiver flat index, -1 = leaves the domain
    flow_acc: np.ndarray  # upstream cells including itself
    drainage_mask: np.ndarray
    drain_id: np.ndarray  # flat index of the nearest downstream drainage cell
    hand: np.ndarray
    terminal: np.ndarray  # flat index of the edge cell where each cell's flow exits
    order: np.ndarray  # flat indices sorted by z_fill ascending


def compute_hydrology(z: np.ndarray, cell: float, drainage_area_km2: float,
                      force_drainage: np.ndarray | None = None) -> Hydrology:
    ny, nx = z.shape
    zf = priority_flood(z.astype(np.float64))
    rec = d8_receivers(zf, cell).ravel()
    order = np.argsort(zf.ravel(), kind="stable")
    rec_l = rec.tolist()
    # accumulation: highest to lowest
    acc = np.ones(ny * nx, np.float64)
    acc_l = acc.tolist()
    for c in order[::-1].tolist():
        r = rec_l[c]
        if r >= 0:
            acc_l[r] += acc_l[c]
    acc = np.asarray(acc_l).reshape(ny, nx)
    drain = acc * cell * cell >= drainage_area_km2 * 1e6
    if force_drainage is not None:
        drain |= force_drainage
    dflat = drain.ravel().tolist()
    # drain id and terminal: lowest to highest (receivers are processed first)
    did = list(range(ny * nx))
    term = list(range(ny * nx))
    for c in order.tolist():
        r = rec_l[c]
        if r >= 0:
            term[c] = term[r]
            if not dflat[c]:
                did[c] = did[r]
        elif not dflat[c]:
            did[c] = c  # an edge cell outside the network is its own reference
    did = np.asarray(did, np.int64)
    zfl = zf.ravel()
    hand = (zfl - zfl[did]).reshape(ny, nx)
    hand = np.maximum(hand, 0.0)
    return Hydrology(
        z_fill=zf,
        depression_depth=(zf - z).astype(np.float32),
        flow_dir=rec.reshape(ny, nx),
        flow_acc=acc.astype(np.float32),
        drainage_mask=drain,
        drain_id=did.reshape(ny, nx),
        hand=hand.astype(np.float32),
        terminal=np.asarray(term, np.int64).reshape(ny, nx),
        order=order,
    )


def edge_of(flat_idx: np.ndarray, n: int) -> np.ndarray:
    """Which edge ('N','E','S','W' as 0..3, -1 interior) a flat index lies on.

    Corner cells count for the N/S edge first.
    """
    iy, ix = np.divmod(np.asarray(flat_idx), n)
    out = np.full(iy.shape, -1)
    out[ix == n - 1] = 1
    out[ix == 0] = 3
    out[iy == n - 1] = 0
    out[iy == 0] = 2
    return out


def edge_mask(n: int, edge: str) -> np.ndarray:
    m = np.zeros((n, n), bool)
    if edge == "N":
        m[-1, :] = True
    elif edge == "S":
        m[0, :] = True
    elif edge == "E":
        m[:, -1] = True
    elif edge == "W":
        m[:, 0] = True
    return m


def fraction_draining_to(hyd: Hydrology, land: np.ndarray, edge: str) -> float:
    n = hyd.z_fill.shape[0]
    em = edge_mask(n, edge).ravel()
    t = hyd.terminal.ravel()
    ok = em[t].reshape(n, n)
    return float(ok[land].mean()) if land.any() else 1.0
