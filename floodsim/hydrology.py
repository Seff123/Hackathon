"""Priority-flood, D8 flow, accumulation, drainage network and HAND."""
from __future__ import annotations

import heapq

import numpy as np

# D8 neighbour offsets (dy, dx) and distances in cell units
D8 = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
D8_DIST = np.hypot(D8[:, 0], D8[:, 1])


def priority_flood(z: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Barnes (2014) priority-flood with epsilon: every cell gets a strictly descending path to an edge."""
    ny, nx = z.shape
    zf = z.astype(np.float64).copy()
    closed = np.zeros((ny, nx), dtype=bool)
    heap: list[tuple[float, int]] = []
    edge = np.zeros((ny, nx), dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    for idx in np.flatnonzero(edge):
        heap.append((zf.flat[idx], int(idx)))
        closed.flat[idx] = True
    heapq.heapify(heap)
    flat_z = zf.reshape(-1)
    flat_c = closed.reshape(-1)
    offs = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    while heap:
        h, c = heapq.heappop(heap)
        cy, cx = divmod(c, nx)
        for dy, dx in offs:
            yy, xx = cy + dy, cx + dx
            if 0 <= yy < ny and 0 <= xx < nx:
                n = yy * nx + xx
                if not flat_c[n]:
                    flat_c[n] = True
                    if flat_z[n] <= h:
                        flat_z[n] = h + eps
                    heapq.heappush(heap, (flat_z[n], n))
    return zf.astype(np.float32)


def d8_directions(zf: np.ndarray) -> np.ndarray:
    """Flat index of the steepest-descent neighbour; -1 means the cell drains off the grid edge."""
    ny, nx = zf.shape
    pad = np.pad(zf.astype(np.float64), 1, mode="constant", constant_values=np.inf)
    best_slope = np.zeros((ny, nx))
    best = np.full((ny, nx), -1, dtype=np.int64)
    iy, ix = np.mgrid[0:ny, 0:nx]
    for k, (dy, dx) in enumerate(D8):
        nb = pad[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]
        slope = (zf - nb) / D8_DIST[k]
        better = slope > best_slope
        best_slope = np.where(better, slope, best_slope)
        best = np.where(better, (iy + dy) * nx + (ix + dx), best)
    return best.reshape(-1)


def flow_accumulation(zf: np.ndarray, down: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upstream cell count (including self). Returns (acc [ny,nx], order high->low)."""
    order = np.argsort(-zf.reshape(-1), kind="stable")
    acc = np.ones(zf.size, dtype=np.float64)
    dl = down.tolist()
    al = acc.tolist()
    for c in order.tolist():
        d = dl[c]
        if d >= 0:
            al[d] += al[c]
    return np.asarray(al).reshape(zf.shape), order


def drainage_and_hand(zf: np.ndarray, down: np.ndarray, order: np.ndarray, drainage: np.ndarray):
    """Assign every cell its downstream drainage cell; HAND = zf - zf[drain] (>= 0)."""
    n = zf.size
    drain = np.full(n, -1, dtype=np.int64)
    dmask = drainage.reshape(-1)
    dl = down.tolist()
    dr = drain.tolist()
    dm = dmask.tolist()
    for c in order[::-1].tolist():  # low -> high: downstream cells first
        if dm[c]:
            dr[c] = c
        else:
            d = dl[c]
            dr[c] = dr[d] if d >= 0 else -1
    drain = np.asarray(dr, dtype=np.int64)
    zflat = zf.reshape(-1)
    hand = np.where(drain >= 0, zflat - zflat[np.maximum(drain, 0)], np.nan)
    # cells draining off the grid without meeting a drainage cell: height above their own outlet path minimum
    hand = np.where(np.isnan(hand), np.nanmax(hand) if np.isfinite(np.nanmax(hand)) else 0.0, hand)
    return drain.reshape(zf.shape), np.maximum(hand, 0.0).astype(np.float32).reshape(zf.shape)


def outlet_of(down: np.ndarray, order: np.ndarray) -> np.ndarray:
    """For each cell, the flat index of the edge cell where its flow path leaves the grid."""
    out = np.full(down.size, -1, dtype=np.int64)
    dl = down.tolist()
    ol = out.tolist()
    for c in order[::-1].tolist():
        d = dl[c]
        ol[c] = c if d < 0 else ol[d]
    return np.asarray(ol, dtype=np.int64)


def weighted_accumulation(weights: np.ndarray, down: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Sum of `weights` over each cell's upstream area (including itself)."""
    acc = weights.astype(np.float64).reshape(-1).tolist()
    dl = down.tolist()
    for c in order.tolist():
        d = dl[c]
        if d >= 0:
            acc[d] += acc[c]
    return np.asarray(acc).reshape(weights.shape)


def fill_and_spill(z: np.ndarray, z_fill: np.ndarray, runoff_vol: np.ndarray, down: np.ndarray,
                   cell_area: float, min_depth: float = 0.01) -> np.ndarray:
    """Pond depth from routing local runoff volume (m3 per cell) into depressions, upstream ponds first;
    a full pond spills its excess into the next depression downstream. Returns depth grid (m)."""
    from scipy import ndimage
    ny, nx = z.shape
    dep = (z_fill - z) > min_depth
    lab, nlab = ndimage.label(dep, structure=np.ones((3, 3)))
    depth = np.zeros_like(z, dtype=np.float32)
    if nlab == 0:
        return depth
    labf = lab.reshape(-1)
    dl = down.tolist()
    order_up = np.argsort(z_fill.reshape(-1), kind="stable").tolist()   # low -> high
    first = [-1] * labf.size
    lf = labf.tolist()
    for c in order_up:
        if lf[c] > 0:
            first[c] = lf[c]
        else:
            d = dl[c]
            first[c] = first[d] if d >= 0 else -1
    first = np.asarray(first)
    local = np.bincount(np.maximum(first, 0), weights=runoff_vol.reshape(-1) * (first > 0), minlength=nlab + 1)
    # downstream region of each region: follow the outflow of any region cell whose receiver is outside it
    downf = np.asarray(dl)
    region_cells = np.flatnonzero(labf > 0)
    rc_down = downf[region_cells]
    leaves = (rc_down < 0) | (labf[np.maximum(rc_down, 0)] != labf[region_cells])
    next_region = np.zeros(nlab + 1, dtype=np.int64)
    for c, d in zip(region_cells[leaves].tolist(), rc_down[leaves].tolist()):
        r = labf[c]
        next_region[r] = first[d] if d >= 0 else -1
    spill = ndimage.maximum(z_fill, lab, index=np.arange(1, nlab + 1))
    objs = ndimage.find_objects(lab)
    vin = local.astype(float)
    for k in np.argsort(-spill):          # upstream (higher spill level) first
        r = k + 1
        sl = objs[k]
        if sl is None:
            continue
        m = lab[sl] == r
        zz = z[sl][m].astype(np.float64)
        zs = np.sort(zz)
        cum = np.cumsum(zs)
        idx = np.arange(1, len(zs) + 1)
        vol_at = cell_area * (idx * zs - cum)
        cap = cell_area * (len(zs) * float(spill[k]) - cum[-1])
        V = min(vin[r], cap)
        if vin[r] > cap and next_region[r] > 0:
            vin[next_region[r]] += vin[r] - cap
        if V <= 0:
            continue
        j = max(int(np.searchsorted(vol_at, V, side="right") - 1), 0)
        level = min(zs[j] + (V - vol_at[j]) / (cell_area * (j + 1)), float(spill[k]))
        sub = depth[sl]
        sub[m] = np.maximum(level - zz, 0.0).astype(np.float32)
        depth[sl] = sub
    return depth
