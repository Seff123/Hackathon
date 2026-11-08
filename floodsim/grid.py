"""The one helper pair for converting between world metres and grid indices.

Frame: East-North-Up metres, origin at the south-west corner. Grids are indexed [iy, ix] with row 0 in the
south. Cell (iy, ix) covers x in [ix*c, (ix+1)*c), y in [iy*c, (iy+1)*c).
"""
from __future__ import annotations

import numpy as np


def world_to_grid(x, y, cell: float = 50.0, n: int | None = None):
    ix = np.floor(np.asarray(x, dtype=float) / cell).astype(np.int64)
    iy = np.floor(np.asarray(y, dtype=float) / cell).astype(np.int64)
    if n is not None:
        ix = np.clip(ix, 0, n - 1)
        iy = np.clip(iy, 0, n - 1)
    return iy, ix


def grid_to_world(iy, ix, cell: float = 50.0):
    return (np.asarray(ix, dtype=float) + 0.5) * cell, (np.asarray(iy, dtype=float) + 0.5) * cell


def flat_to_world(idx, n: int, cell: float = 50.0):
    iy, ix = np.divmod(np.asarray(idx, dtype=np.int64), n)
    return grid_to_world(iy, ix, cell)


def bilinear(grid: np.ndarray, x, y, cell: float):
    """Bilinear sample of a cell-centred grid at world points (clamped at the edges)."""
    ny, nx = grid.shape
    fx = np.asarray(x, dtype=float) / cell - 0.5
    fy = np.asarray(y, dtype=float) / cell - 0.5
    fx = np.clip(fx, 0.0, nx - 1.000001)
    fy = np.clip(fy, 0.0, ny - 1.000001)
    x0 = fx.astype(np.int64)
    y0 = fy.astype(np.int64)
    ax = fx - x0
    ay = fy - y0
    g00 = grid[y0, x0]
    g01 = grid[y0, x0 + 1]
    g10 = grid[y0 + 1, x0]
    g11 = grid[y0 + 1, x0 + 1]
    return (g00 * (1 - ax) + g01 * ax) * (1 - ay) + (g10 * (1 - ax) + g11 * ax) * ay


def nearest(grid: np.ndarray, x, y, cell: float):
    ny, nx = grid.shape
    iy, ix = world_to_grid(x, y, cell)
    return grid[np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)]


def upsample2(grid: np.ndarray) -> np.ndarray:
    """Upsample a cell-centred grid by 2 with linear interpolation, cell edges aligned (zoom order 1)."""
    from scipy.ndimage import zoom
    return zoom(grid.astype(np.float32), 2, order=1, grid_mode=True, mode="nearest")
