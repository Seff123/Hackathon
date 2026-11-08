"""Grid helpers. Grids are indexed [iy, ix] with row 0 at the south edge.

Cell (iy, ix) has its centre at ((ix + 0.5) * cell, (iy + 0.5) * cell).
"""

from __future__ import annotations

import numpy as np


def world_to_grid(x, y, cell: float, n: int | None = None):
    """World metres -> integer (iy, ix) of the containing cell (clipped if n is given)."""
    ix = np.floor(np.asarray(x, float) / cell).astype(np.int64)
    iy = np.floor(np.asarray(y, float) / cell).astype(np.int64)
    if n is not None:
        ix = np.clip(ix, 0, n - 1)
        iy = np.clip(iy, 0, n - 1)
    return iy, ix


def grid_to_world(iy, ix, cell: float):
    """Integer (iy, ix) -> world metres (x, y) of the cell centre."""
    return (np.asarray(ix, float) + 0.5) * cell, (np.asarray(iy, float) + 0.5) * cell


def cell_centres(n: int, cell: float):
    c = (np.arange(n) + 0.5) * cell
    X, Y = np.meshgrid(c, c)
    return X, Y


def bilinear(grid: np.ndarray, x, y, cell: float):
    """Bilinear sample of a cell-centred grid at world (x, y); clamps at edges."""
    ny, nx = grid.shape[-2:]
    fx = np.clip(np.asarray(x, float) / cell - 0.5, 0.0, nx - 1.000001)
    fy = np.clip(np.asarray(y, float) / cell - 0.5, 0.0, ny - 1.000001)
    ix = fx.astype(np.int64)
    iy = fy.astype(np.int64)
    tx = fx - ix
    ty = fy - iy
    g00 = grid[..., iy, ix]
    g01 = grid[..., iy, ix + 1]
    g10 = grid[..., iy + 1, ix]
    g11 = grid[..., iy + 1, ix + 1]
    return (g00 * (1 - tx) + g01 * tx) * (1 - ty) + (g10 * (1 - tx) + g11 * tx) * ty


def nearest(grid: np.ndarray, x, y, cell: float):
    ny, nx = grid.shape[-2:]
    iy, ix = world_to_grid(x, y, cell)
    return grid[..., np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)]


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def block_reduce(a: np.ndarray, f: int, how: str = "mean") -> np.ndarray:
    ny, nx = a.shape
    b = a.reshape(ny // f, f, nx // f, f)
    if how == "mean":
        return b.mean(axis=(1, 3))
    if how == "min":
        return b.min(axis=(1, 3))
    if how == "max":
        return b.max(axis=(1, 3))
    if how == "any":
        return b.any(axis=(1, 3))
    raise ValueError(how)


def upsample_repeat(a: np.ndarray, f: int) -> np.ndarray:
    return np.repeat(np.repeat(a, f, axis=-2), f, axis=-1)


def rasterise_polyline(xs, ys, cell: float, n: int, step: float | None = None):
    """Ordered, 4-connected list of (iy, ix) cells crossed by a polyline."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    step = step or cell / 10.0
    seg = np.hypot(np.diff(xs), np.diff(ys))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        iy, ix = world_to_grid(xs[:1], ys[:1], cell, n)
        return np.stack([iy, ix], 1)
    ss = np.arange(0.0, s[-1] + step, step)
    px = np.interp(ss, s, xs)
    py = np.interp(ss, s, ys)
    iy, ix = world_to_grid(px, py, cell)
    keep = (iy >= 0) & (iy < n) & (ix >= 0) & (ix < n)
    iy, ix = iy[keep], ix[keep]
    if len(iy) == 0:
        return np.zeros((0, 2), np.int64)
    change = np.ones(len(iy), bool)
    change[1:] = (iy[1:] != iy[:-1]) | (ix[1:] != ix[:-1])
    iy, ix = iy[change], ix[change]
    out = [(int(iy[0]), int(ix[0]))]
    for a, b in zip(iy[1:], ix[1:]):
        py0, px0 = out[-1]
        a, b = int(a), int(b)
        if a != py0 and b != px0:  # diagonal step: insert an orthogonal cell
            out.append((py0, b))
        out.append((a, b))
    return np.asarray(out, np.int64)
