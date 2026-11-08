"""Static plots, GIF animation and plotly 3D view (environment section 16)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

DRONE_COLOURS = ["#e4572e", "#7b5cd6", "#17a398", "#d9a400", "#3f88c5", "#c73e9b", "#5b8c2a", "#8c564b"]
MODE_STYLE = {"search": "-", "transit": "--", "thermal": ":", "verify": "-", "rtb": "-", "ground": "-", "down": "-"}
MODE_WIDTH = {"verify": 2.6}
LC_TINT = {0: (0.93, 0.92, 0.85), 1: (0.55, 0.70, 0.88), 2: (0.85, 0.75, 0.70), 3: (0.62, 0.74, 0.58)}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def hillshade(z: np.ndarray, cell: float, az=315.0, alt=45.0) -> np.ndarray:
    gy, gx = np.gradient(z.astype(float), cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, b = math.radians(az), math.radians(alt)
    hs = np.sin(b) * np.cos(slope) + np.cos(b) * np.sin(slope) * np.cos(a - aspect)
    return np.clip(hs, 0, 1)


def base_image(terrain) -> np.ndarray:
    """RGB terrain: land-cover tint modulated by hillshade. Row 0 = south (use origin='lower')."""
    hs = hillshade(terrain.z, terrain.cell_size)
    rgb = np.zeros(terrain.z.shape + (3,))
    for k, c in LC_TINT.items():
        rgb[terrain.landcover == k] = c
    return np.clip(rgb * (0.55 + 0.45 * hs[..., None]), 0, 1)


def _draw_static(ax, world, show_people=False, t=0.0):
    ter = world.terrain
    L = world.cfg.world.domain_m
    ax.imshow(base_image(ter), origin="lower", extent=(0, L, 0, L))
    for pl in ter.roads:
        p = np.asarray(pl)
        ax.plot(p[:, 0], p[:, 1], color="#555", lw=0.8, alpha=0.8)
    ax.plot(ter.river_xy[::10, 0], ter.river_xy[::10, 1], color="#2a5d9f", lw=1.2)
    for lv in ter.levee_lines:
        p = np.asarray(lv)
        if len(p) > 1:
            ax.plot(p[:, 0], p[:, 1], color="#7a4e2d", lw=2.0, alpha=0.7)
    for tw in ter.towns:
        ax.text(tw["x"], tw["y"] + tw["r"] * 0.6, tw["name"], ha="center", fontsize=7, weight="bold", color="#222")
    bx, by = ter.base_xy
    ax.plot([bx], [by], marker="^", color="k", ms=8)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_flood(ax, world, t, alpha_max=0.8):
    d = world.flood.depth_grid50(t)
    L = world.cfg.world.domain_m
    rgba = np.zeros(d.shape + (4,))
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 0.12, 0.42, 0.78
    rgba[..., 3] = np.where(d > 0.05, np.clip(0.25 + d / 3.0, 0, 1) * alpha_max, 0)
    ax.imshow(rgba, origin="lower", extent=(0, L, 0, L))


def _draw_people(ax, people, found_only=False):
    from .people import STATE_NAMES
    marker = {0: "s", 1: "^", 2: "D", 3: "o", 4: "o", 5: "o"}
    for k in range(people.n):
        if people.found[k]:
            ax.plot(people.x[k], people.y[k], "o", ms=6, mfc="#23845a", mec="#23845a")
            ax.plot(people.x[k], people.y[k], "o", ms=11, mfc="none", mec="#23845a")
        elif not found_only:
            c = "#c2362f" if people.ever_affected[k] else "#666"
            ax.plot(people.x[k], people.y[k], marker[int(people.state[k])], ms=6, mfc="none", mec=c, mew=1.4)


def plot_world(world, t: float = 0.0, path=None, show_people=True):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    _draw_static(ax, world)
    _draw_flood(ax, world, t)
    if show_people:
        _draw_people(ax, world.people)
    s = world.spec
    ax.set_title(f"seed {s.seed}: {s.flood_type}, t = {t / 60:.0f} min, peak inflow {s.peak_inflow_m3s:.0f} m³/s"
                 + ("  (people: truth, debug)" if show_people else ""), fontsize=9)
    if path:
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return fig


def _segments_by_mode(traj, drone):
    from .autopilot import MODE_NAMES
    d = traj[traj[:, 1] == drone]
    if len(d) == 0:
        return []
    segs = []
    start = 0
    for k in range(1, len(d) + 1):
        if k == len(d) or d[k, 8] != d[start, 8]:
            segs.append((MODE_NAMES[int(d[start, 8])], d[max(start - 1, 0):k]))
            start = k
    return segs


def plot_coverage(result, world, t: float | None = None, path=None, seen=None):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    _draw_static(ax, world)
    T = result.trajectory
    if t is not None:
        T = T[T[:, 0] <= t]
    L = world.cfg.world.domain_m
    if seen is not None:
        rgba = np.zeros(seen.shape + (4,))
        rgba[..., 0], rgba[..., 2] = 0.8, 0.6
        rgba[..., 3] = np.where(seen, 0.25, 0)
        ax.imshow(rgba, origin="lower", extent=(0, L, 0, L))
    for i in range(result.n_drones):
        for mode, seg in _segments_by_mode(T, i):
            if mode in ("ground", "down"):
                continue
            ax.plot(seg[:, 2], seg[:, 3], MODE_STYLE.get(mode, "-"), color=DRONE_COLOURS[i % 8] if mode != "rtb" else "#999",
                    lw=MODE_WIDTH.get(mode, 1.1), alpha=0.9)
    c = world.terrain.cell_size
    n = world.terrain.n
    for (tt, i, cell, pid) in result.detections_log:
        if t is not None and tt > t:
            continue
        iy, ix = divmod(cell, n)
        ax.plot((ix + 0.5) * c, (iy + 0.5) * c, "x" if pid < 0 else "+", color="#c2362f" if pid < 0 else "#23845a", ms=5)
    ax.set_title(f"{result.controller}: tracks and detections (+ true, x false)", fontsize=9)
    if path:
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_wind_slice(world, z_agl: float = 150.0, t: float = 0.0, path=None):
    plt = _plt()
    u, v, w = world.weather.wind_grid(z_agl, t)
    L = world.cfg.world.domain_m
    fig, ax = plt.subplots(figsize=(7, 6), dpi=110)
    lim = max(0.5, float(np.abs(w).max()))
    im = ax.imshow(w, origin="lower", extent=(0, L, 0, L), cmap="RdBu_r", vmin=-lim, vmax=lim)
    s = 12
    xs = (np.arange(0, w.shape[0], s) + 0.5) * world.terrain.cell_size
    X, Y = np.meshgrid(xs, xs)
    ax.quiver(X, Y, u[::s, ::s], v[::s, ::s], color="k", alpha=0.6, scale=200)
    for th in world.weather.thermals():
        ax.add_patch(plt.Circle((th.x, th.y), th.r_t, fill=False, ls=":", color="#c2362f"))
    fig.colorbar(im, ax=ax, label="vertical wind (m/s)")
    ax.set_title(f"wind at {z_agl:.0f} m above ground: ridge lift and thermals", fontsize=9)
    ax.set_aspect("equal")
    if path:
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return fig


def animate_episode(result, world, path, every: int = 2, fps: int = 8, max_frames: int = 160) -> str | None:
    """GIF with 2x2 panels per recorded frame: truth flood + people; coverage + tracks; altitude and battery;
    vertical wind at glider height. Needs a run with record_grids=True."""
    plt = _plt()
    rec = result.recording
    if not rec or not rec["frames"]:
        return None
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None
    frames = rec["frames"][::every][:max_frames]
    T = result.trajectory
    L = world.cfg.world.domain_m
    base = base_image(world.terrain)
    u, v, wgrid = world.weather.wind_grid(150.0, 0.0)
    lim = max(0.5, float(np.abs(wgrid).max()))
    n = result.n_drones
    cap = result.energy_log["capacity_wh"]
    imgs = []
    fig = plt.figure(figsize=(10, 8.4), dpi=72)
    for fr in frames:
        fig.clf()
        t = fr["t"]
        ax1, ax2 = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2)
        ax3, ax4 = fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)
        for ax in (ax1, ax2, ax4):
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax1.imshow(base, origin="lower", extent=(0, L, 0, L))
        d = fr["depth"].astype(np.float32)
        rgba = np.zeros(d.shape + (4,)); rgba[..., 2] = 0.8; rgba[..., 1] = 0.4
        rgba[..., 3] = np.where(d > 0.05, np.clip(0.3 + d / 3, 0, 0.85), 0)
        ax1.imshow(rgba, origin="lower", extent=(0, L, 0, L))
        P = fr["people"]
        for row in P:
            x, y, st, found, surv, aff = row
            if found:
                ax1.plot(x, y, "o", color="#23845a", ms=6)
            else:
                ax1.plot(x, y, "o", mfc="none", mec="#c2362f" if aff else "#555", ms=5)
        ax1.set_title(f"truth: flood + people  t = {t / 60:.0f} min", fontsize=9)
        ax2.imshow(base, origin="lower", extent=(0, L, 0, L))
        seen = fr["seen"].reshape(world.terrain.n, world.terrain.n)
        rg = np.zeros(seen.shape + (4,)); rg[..., 0] = 0.85; rg[..., 2] = 0.6; rg[..., 3] = seen * 0.3
        ax2.imshow(rg, origin="lower", extent=(0, L, 0, L))
        Tt = T[(T[:, 0] <= t) & (T[:, 0] > t - 900)]
        for i in range(n):
            di = Tt[Tt[:, 1] == i]
            if len(di):
                ax2.plot(di[:, 2], di[:, 3], color=DRONE_COLOURS[i % 8], lw=1)
            x, y = fr["drones"][i][0], fr["drones"][i][1]
            ax2.plot(x, y, "o", color=DRONE_COLOURS[i % 8], ms=5, mec="k")
        ax2.set_xlim(0, L); ax2.set_ylim(0, L)
        ax2.set_title("coverage + tracks (fleet)", fontsize=9)
        Th = T[T[:, 0] <= t]
        for i in range(n):
            di = Th[Th[:, 1] == i]
            if len(di):
                ax3.plot(di[:, 0] / 60, di[:, 4], color=DRONE_COLOURS[i % 8], lw=1)
        ax3b = ax3.twinx()
        for i in range(n):
            di = Th[Th[:, 1] == i]
            if len(di):
                ax3b.plot(di[:, 0] / 60, di[:, 7] / cap * 100, ":", color=DRONE_COLOURS[i % 8], lw=1)
        ax3.set_xlim(0, world.cfg.world.episode_s / 60)
        ax3b.set_ylim(0, 105)
        ax3.set_xlabel("minutes"); ax3.set_ylabel("altitude (m)"); ax3b.set_ylabel("battery % (dotted)")
        ax4.imshow(wgrid, origin="lower", extent=(0, L, 0, L), cmap="RdBu_r", vmin=-lim, vmax=lim)
        for i in range(n):
            ax4.plot(fr["drones"][i][0], fr["drones"][i][1], "o", color=DRONE_COLOURS[i % 8], ms=5, mec="k")
        ax4.set_xlim(0, L); ax4.set_ylim(0, L)
        ax4.set_title("vertical wind at 150 m (red up, blue down)", fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        imgs.append(Image.fromarray(buf))
    plt.close(fig)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    _maybe_mp4(imgs, Path(path).with_suffix(".mp4"), fps)
    return str(path)


def _maybe_mp4(imgs, path, fps):
    import shutil
    import subprocess
    import tempfile
    if shutil.which("ffmpeg") is None:
        return
    with tempfile.TemporaryDirectory() as td:
        for k, im in enumerate(imgs):
            im.save(f"{td}/f{k:04d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{td}/f%04d.png",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", str(path)], check=False)


def plot3d_html(result, world, path, exaggeration: float = 3.0, water_times=None, radio_links: bool = True):
    """Plotly 3D: terrain, translucent water surface, 3D path per drone (dash by mode), thermal columns,
    radio links. Falls back to a matplotlib 3D PNG when plotly is missing."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return plot3d_png(result, world, Path(path).with_suffix(".png"))
    from .autopilot import MODE_NAMES
    ter = world.terrain
    f = 2
    z = ter.z[::f, ::f].astype(float)
    c = ter.cell_size * f
    xs = (np.arange(z.shape[1]) + 0.5) * c
    z0 = float(z.min())
    ex = exaggeration
    lc = ter.landcover[::f, ::f]
    hs = hillshade(z, c)
    cs = np.choose(np.clip(lc, 0, 3), [0.55, 0.15, 0.85, 0.35]) * 0.7 + 0.3 * hs
    fig = go.Figure()
    fig.add_trace(go.Surface(x=xs, y=xs, z=(z - z0) * ex, surfacecolor=cs, colorscale="Earth", showscale=False,
                             name="terrain", hoverinfo="skip", cmin=0, cmax=1))
    times = water_times or [0.0, world.cfg.world.episode_s / 2, world.cfg.world.episode_s]
    n_water = 0
    for k, t in enumerate(times):
        d = world.flood.depth_grid50(t)[::f, ::f]
        wz = np.where(d > 0.1, (ter.z[::f, ::f] + d - z0) * ex + 0.5, np.nan)
        fig.add_trace(go.Surface(x=xs, y=xs, z=wz, colorscale=[[0, "#1f6fc2"], [1, "#1f6fc2"]], opacity=0.55,
                                 showscale=False, visible=(k == len(times) - 1), name=f"water {t / 60:.0f} min"))
        n_water += 1
    T = result.trajectory
    dash = {"search": "solid", "transit": "dash", "thermal": "dot", "verify": "solid", "rtb": "longdash"}
    for i in range(result.n_drones):
        for mode, seg in _segments_by_mode(T, i):
            if mode in ("ground", "down") or len(seg) < 2:
                continue
            fig.add_trace(go.Scatter3d(x=seg[:, 2], y=seg[:, 3], z=(seg[:, 4] - z0) * ex, mode="lines",
                                       line=dict(color=DRONE_COLOURS[i % 8] if mode != "rtb" else "#888",
                                                 width=6 if mode == "verify" else 3, dash=dash.get(mode, "solid")),
                                       name=f"G{i + 1} {mode}", showlegend=False, hoverinfo="name"))
    for fd in result.finds:
        k = fd["person"]
        px, py = world.people.start_x[k], world.people.start_y[k]
        if result.recording and result.recording["frames"]:
            pass
        pz = (ter.z[int(py // ter.cell_size), int(px // ter.cell_size)] - z0) * ex
        fig.add_trace(go.Scatter3d(x=[px], y=[py], z=[pz + 10], mode="markers", marker=dict(color="#23845a", size=6),
                                   name="confirmed survivor", showlegend=False))
    bx, by = ter.base_xy
    bz = (ter.z[int(by // ter.cell_size), int(bx // ter.cell_size)] - z0) * ex
    fig.add_trace(go.Cone(x=[bx], y=[by], z=[bz + 120], u=[0], v=[0], w=[-100], sizeref=200, showscale=False,
                          colorscale=[[0, "black"], [1, "black"]], name="base"))
    zi = world.weather.zi
    for th in world.weather.thermals():
        a = np.linspace(0, 2 * np.pi, 24)
        h = np.array([0, zi])
        A, H = np.meshgrid(a, h)
        gz = (ter.z[min(int(th.y // ter.cell_size), ter.n - 1), min(int(th.x // ter.cell_size), ter.n - 1)] - z0) * ex
        fig.add_trace(go.Surface(x=th.x + th.r_t * np.cos(A), y=th.y + th.r_t * np.sin(A), z=gz + H * ex,
                                 showscale=False, opacity=0.15, colorscale=[[0, "#e4572e"], [1, "#e4572e"]],
                                 hoverinfo="skip", name="thermal"))
    if radio_links and len(T):
        tl = T[:, 0].max() / 2
        sl = T[np.abs(T[:, 0] - T[np.argmin(np.abs(T[:, 0] - tl)), 0]) < 1e-6]
        R = world.spec.signal_radius_km * 1000
        for a in range(len(sl)):
            for b in range(a + 1, len(sl)):
                if math.hypot(sl[a, 2] - sl[b, 2], sl[a, 3] - sl[b, 3]) <= R:
                    fig.add_trace(go.Scatter3d(x=sl[[a, b], 2], y=sl[[a, b], 3], z=(sl[[a, b], 4] - z0) * ex, mode="lines",
                                               line=dict(color="#999", width=1), showlegend=False, hoverinfo="skip"))
    n_tr = len(fig.data)
    buttons = []
    for k, t in enumerate(times):
        vis = [True] * n_tr
        for j in range(n_water):
            vis[1 + j] = (j == k)
        buttons.append(dict(label=f"water {t / 60:.0f} min", method="update", args=[{"visible": vis}]))
    L = world.cfg.world.domain_m
    cams = {"overview": dict(eye=dict(x=1.4, y=-1.4, z=0.9)), "valley floor": dict(eye=dict(x=0.3, y=-0.8, z=0.25))}
    fig.update_layout(
        title=f"FloodSim seed {world.spec.seed}: {result.controller}, {result.n_drones} gliders (vertical ×{ex:g})",
        scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.35), xaxis=dict(range=[0, L], title="east (m)"),
                   yaxis=dict(range=[0, L], title="north (m)"), zaxis=dict(title="height ×%g" % ex)),
        updatemenus=[dict(type="buttons", buttons=buttons, x=0.0, y=1.0, direction="left"),
                     dict(type="buttons", x=0.0, y=0.92, direction="left",
                          buttons=[dict(label=k, method="relayout", args=[{"scene.camera": v}]) for k, v in cams.items()])],
        margin=dict(l=0, r=0, t=40, b=0), template="plotly_white")
    fig.write_html(str(path), include_plotlyjs="cdn" if False else True, full_html=True)
    return str(path)


def plot3d_png(result, world, path):
    plt = _plt()
    ter = world.terrain
    f = 4
    z = ter.z[::f, ::f]
    xs = (np.arange(z.shape[0]) + 0.5) * ter.cell_size * f
    X, Y = np.meshgrid(xs, xs)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, z * 3, cmap="terrain", alpha=0.8, linewidth=0)
    T = result.trajectory
    for i in range(result.n_drones):
        d = T[T[:, 1] == i]
        ax.plot(d[:, 2], d[:, 3], d[:, 4] * 3, color=DRONE_COLOURS[i % 8], lw=1)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)
