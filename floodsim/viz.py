"""Static plots, GIF animation and plotly 3D views."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LightSource, ListedColormap  # noqa: E402

DRONE_COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6"]
LC_COLORS = {0: (0.25, 0.45, 0.85), 1: (0.80, 0.85, 0.62), 2: (0.25, 0.50, 0.25), 3: (0.65, 0.55, 0.55)}


def _extent(n, cell):
    return (0, n * cell / 1000, 0, n * cell / 1000)


def terrain_rgb(terr):
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(terr.z.astype(float), vert_exag=3, dx=terr.cell_size, dy=terr.cell_size)
    # hillshade expects row 0 at top; our row 0 is south, so flip for shading then back
    hs = ls.hillshade(terr.z[::-1].astype(float), vert_exag=3, dx=terr.cell_size, dy=terr.cell_size)[::-1]
    lc = np.zeros(terr.z.shape + (3,))
    for k, col in LC_COLORS.items():
        lc[terr.landcover == k] = col
    rgb = lc * (0.45 + 0.55 * hs[..., None])
    rgb[terr.road_mask] = (0.15, 0.15, 0.15)
    return np.clip(rgb, 0, 1)


def draw_world(ax, world_or_terrain, depth=None, show_people=False, people=None, t=None):
    terr = getattr(world_or_terrain, "terrain", world_or_terrain)
    ext = _extent(terr.n, terr.cell_size)
    ax.imshow(terrain_rgb(terr), origin="lower", extent=ext, interpolation="nearest")
    if depth is not None:
        d = np.asarray(depth, float)
        cmap = plt.get_cmap("Blues")
        rgba = cmap(np.clip(d / 3.0, 0.15, 1.0))
        rgba[..., 3] = np.clip(d / 0.5, 0, 0.85) * (d > 0.02)
        ax.imshow(rgba, origin="lower", extent=ext, interpolation="nearest")
    ax.contour(terr.levee_mask.astype(float), levels=[0.5], colors="saddlebrown", linewidths=0.8,
               extent=ext, origin="lower")
    for town in terr.towns:
        ax.add_patch(plt.Circle((town.x / 1000, town.y / 1000), town.radius / 1000, fill=False,
                                ec="k", lw=0.6, ls="--"))
    bx, by = terr.base_xy
    ax.plot(bx / 1000, by / 1000, marker="^", ms=10, mec="k", mfc="gold", zorder=10)
    if show_people and people is not None:
        ax.scatter(people.x / 1000, people.y / 1000, s=14, c="red", edgecolors="k", lw=0.4, zorder=11)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    return ax


def plot_terrain(terr, path, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    draw_world(axes[0], terr)
    axes[0].set_title(title or "terrain, land cover, roads, towns, base")
    ext = _extent(terr.n, terr.cell_size)
    im = axes[1].imshow(np.minimum(terr.hand, 30), origin="lower", extent=ext, cmap="viridis_r")
    axes[1].contour(terr.drainage_mask.astype(float), levels=[0.5], colors="w", linewidths=0.4,
                    extent=ext, origin="lower")
    axes[1].set_title("HAND (m, clipped 30) and drainage network")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
