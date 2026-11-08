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


def _save_gif(frames, path, fps=8):
    """frames: list of HxWx3 uint8 arrays. Writes GIF via pillow; MP4 too if ffmpeg exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        Image = None
    if Image is not None:
        ims = [Image.fromarray(f) for f in frames]
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)
    if shutil.which("ffmpeg"):
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            for i, f in enumerate(frames):
                Image.fromarray(f).save(os.path.join(td, f"f{i:05d}.png"))
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                            os.path.join(td, "f%05d.png"), "-pix_fmt", "yuv420p",
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(path.with_suffix(".mp4"))],
                           check=False)
    return path


def _fig_to_array(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def animate_flood(world, path, every_s=600.0, fps=6):
    """Flood-only animation of the truth over the episode and the 2 h after it."""
    terr = world.terrain
    fl = world.flood
    frames = []
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for t in np.arange(0.0, fl.t_end + 1, every_s):
        ax.clear()
        draw_world(ax, terr, depth=fl.depth_grid(t))
        hh, mm = divmod(int((world.spec.start_hour * 3600 + t) // 60), 60)
        ax.set_title(f"seed {world.spec.seed} {world.spec.flood_type}  t+{t/60:.0f} min ({hh % 24:02d}:{mm:02d})")
        fig.tight_layout()
        frames.append(_fig_to_array(fig))
    plt.close(fig)
    return _save_gif(frames, path, fps)


def plot_world(world, t=0.0, path=None, show_people=True, ax=None):
    """Hillshade + land cover + flood depth (blues) + roads, towns, base, true people (debug)."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 7))
    people = None
    if show_people and getattr(world, "people0", None) is not None:
        people = world.people0
    draw_world(ax, world, depth=world.flood.depth_grid(t), show_people=show_people, people=people)
    ax.set_title(f"seed {world.spec.seed} {world.spec.flood_type}  t={t/60:.0f} min")
    if own:
        fig.tight_layout()
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=110)
        plt.close(fig)
    return path


def scenario_plots(world, out, seed):
    out = Path(out)
    paths = []
    fl = world.flood
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    draw_world(axes[0], world, depth=fl.max_depth(0, world.episode_s), show_people=True, people=world.people0)
    axes[0].set_title(f"peak flood depth during the episode (extent {world.peak_extent():.1%}) + people")
    draw_world(axes[1], world, depth=fl.depth_grid(world.episode_s))
    axes[1].set_title("flood at episode end")
    fig.tight_layout()
    p = out / f"scenario_{seed}_flood.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    paths.append(p)
    if hasattr(world, "weather") and world.weather is not None and "plot_wind_slice" in globals():
        paths.append(plot_wind_slice(world, 100.0, 0.0, out / f"scenario_{seed}_wind.png"))
    return paths
