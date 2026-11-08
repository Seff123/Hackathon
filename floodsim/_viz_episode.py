"""Episode visuals (re-exported by viz.py): coverage plot, demo GIF, 3D HTML, report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .viz import DRONE_COLORS, _extent, _save_gif, terrain_rgb

MODE_STYLE = {  # mode name -> (line style, width): solid search, dashed transit, dotted thermal, thick verify
    "SEARCH": ("-", 1.2), "TRANSIT": ("--", 1.0), "LAUNCH": ("--", 1.0), "RTB": ("--", 1.0),
    "LOITER": ("--", 1.0), "THERMAL": (":", 1.2), "VERIFY": ("-", 2.6),
}
STATE_COLORS = ["#ff9900", "#ff3300", "#cc00cc", "#ffff00", "#00cc00", "#00ffff", "#555555"]


def _coverage_rgba(seen, n):
    rgba = np.zeros((n, n, 4))
    m = np.asarray(seen).reshape(n, n) > 0
    rgba[m] = (0.85, 0.0, 0.85, 0.45)
    return rgba


def _flood_rgba(d):
    rgba = plt.get_cmap("Blues")(np.clip(d / 3.0, 0.15, 1.0))
    rgba[..., 3] = np.clip(d / 0.5, 0, 0.85) * (d > 0.02)
    return rgba


def plot_coverage(result, world, t=None, path=None, ax=None):
    """Drone tracks (style by mode), area covered so far, detections (green circle true, black x false)."""
    from .autopilot import MODE_NAMES

    terr = world.terrain
    tr = result.trajectory
    t = tr["t"][-1] if t is None else t
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 7))
    ext = _extent(terr.n, terr.cell_size)
    grey = terrain_rgb(terr).mean(axis=2, keepdims=True).repeat(3, 2)
    ax.imshow(grey, origin="lower", extent=ext)
    snaps = result.snapshots
    if len(snaps.get("t", [])):
        k = int(np.searchsorted(snaps["t"], t, side="right")) - 1
        if k >= 0:
            ax.imshow(_coverage_rgba(snaps["seen"][k], terr.n), origin="lower", extent=ext)
    keep = tr["t"] <= t
    for i in range(result.n_drones):
        x = tr["x"][keep, i] / 1000
        y = tr["y"][keep, i] / 1000
        modes = tr["mode"][keep, i]
        col = DRONE_COLORS[i % len(DRONE_COLORS)]
        start = 0
        for j in range(1, len(x) + 1):
            if j == len(x) or modes[j] != modes[start]:
                ls, lw = MODE_STYLE.get(MODE_NAMES[modes[start]], ("-", 0.0))
                if lw > 0:
                    ax.plot(x[start:j + 1], y[start:j + 1], ls, color=col, lw=lw)
                start = j
        if keep.any():
            ax.plot(x[-1:], y[-1:], "o", color=col, mec="k", ms=6)
    n = terr.n
    det = [d for d in result.detections if d[0] <= t]
    if det:
        cells = np.array([d[2] for d in det])
        tru = np.array([d[3] is not None for d in det])
        cy, cx = np.divmod(cells, n)
        c = terr.cell_size / 1000
        ax.scatter((cx[tru] + 0.5) * c, (cy[tru] + 0.5) * c, s=40, facecolors="none", edgecolors="lime", lw=1.5,
                   zorder=12)
        ax.scatter((cx[~tru] + 0.5) * c, (cy[~tru] + 0.5) * c, s=12, c="k", marker="x", lw=0.8, zorder=12)
    bx, by = terr.base_xy
    ax.plot(bx / 1000, by / 1000, marker="^", ms=10, mec="k", mfc="gold", zorder=13)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_title(f"coverage (magenta) and tracks, t={t / 60:.0f} min")
    if own:
        fig.tight_layout()
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=110)
        plt.close(fig)
    return path


def animate_episode(result, world, path, every_s=30.0, fps=10, dpi=55):
    """The demo clip: 2 x 2 panels per frame (truth flood + people; coverage + tracks;
    altitude and battery over time; vertical wind at glider height)."""
    terr = world.terrain
    fl = world.flood
    tr = result.trajectory
    snaps = result.snapshots
    nd = result.n_drones
    n = terr.n
    ext = _extent(n, terr.cell_size)
    cap = world.cfg.glider.battery_capacity_wh
    agl = tr["z"] - terr.ground_at(tr["x"], tr["y"])
    times = np.arange(0.0, tr["t"][-1] + 1e-6, every_s)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10.5), dpi=dpi)
    a0, a1, a2, a3 = axs.ravel()
    base_rgb = terrain_rgb(terr)
    grey = base_rgb.mean(axis=2, keepdims=True).repeat(3, 2)
    a0.imshow(base_rgb, origin="lower", extent=ext)
    flood_im = a0.imshow(np.zeros((n, n, 4)), origin="lower", extent=ext)
    ppl = a0.scatter(np.zeros(0), np.zeros(0), s=26, edgecolors="k", lw=0.5, zorder=10)
    a0.plot(terr.base_xy[0] / 1000, terr.base_xy[1] / 1000, marker="^", ms=9, mec="k", mfc="gold")
    a0.set_title("truth: flood depth and people (white = found)")
    a1.imshow(grey, origin="lower", extent=ext)
    cov_im = a1.imshow(np.zeros((n, n, 4)), origin="lower", extent=ext)
    tracks = [a1.plot([], [], "-", color=DRONE_COLORS[i % 8], lw=1.0)[0] for i in range(nd)]
    heads = [a1.plot([], [], "o", color=DRONE_COLORS[i % 8], mec="k", ms=6)[0] for i in range(nd)]
    det_t = a1.scatter(np.zeros(0), np.zeros(0), s=40, facecolors="none", edgecolors="lime", lw=1.5, zorder=12)
    det_f = a1.scatter(np.zeros(0), np.zeros(0), s=12, c="k", marker="x", lw=0.8, zorder=12)
    a1.set_title("coverage (magenta), tracks, detections (o true, x false)")
    for a in (a0, a1):
        a.set_xlim(ext[0], ext[1])
        a.set_ylim(ext[2], ext[3])
    hl = [a2.plot([], [], "-", color=DRONE_COLORS[i % 8], lw=1)[0] for i in range(nd)]
    a2.set_xlim(0, tr["t"][-1] / 60)
    a2.set_ylim(0, max(float(np.nanmax(agl)) * 1.1, 100))
    a2.set_xlabel("minutes")
    a2.set_ylabel("height above ground (m), solid")
    a2b = a2.twinx()
    bl = [a2b.plot([], [], "--", color=DRONE_COLORS[i % 8], lw=0.8)[0] for i in range(nd)]
    a2b.set_ylim(0, 1.05)
    a2b.set_ylabel("battery fraction, dashed")
    a2.set_title("altitude and battery")
    u, v, w = world.weather.wind_grid(100.0)
    lim = max(float(np.percentile(np.abs(w), 99.5)), 0.3)
    a3.imshow(w, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
    th = world.weather.thermal_list()
    if len(th):
        a3.scatter(th[:, 0] / 1000, th[:, 1] / 1000, s=10, facecolors="none", edgecolors="orange")
    pos = a3.scatter(np.zeros(0), np.zeros(0), s=30, edgecolors="k")
    a3.set_title("vertical wind at 100 m AGL (red up, blue down) and drones")
    a3.set_xlim(ext[0], ext[1])
    a3.set_ylim(ext[2], ext[3])
    sup = fig.suptitle("")
    cells = np.array([d[2] for d in result.detections], dtype=np.int64)
    dts = np.array([d[0] for d in result.detections], float)
    dtrue = np.array([d[3] is not None for d in result.detections], bool)
    frames = []
    for t in times:
        flood_im.set_data(_flood_rgba(fl.depth_grid(min(t, fl.t_end))))
        k = int(np.searchsorted(snaps["t"], t, side="right")) - 1 if len(snaps.get("t", [])) else -1
        found = 0
        if k >= 0:
            cov_im.set_data(_coverage_rgba(snaps["seen"][k], n))
            st = snaps["pstate"][k]
            fd = snaps["pfound"][k]
            ppl.set_offsets(np.c_[snaps["px"][k] / 1000, snaps["py"][k] / 1000])
            ppl.set_facecolors(["#ffffff" if f else STATE_COLORS[s_] for s_, f in zip(st, fd)])
            found = int(fd.sum())
        keep = tr["t"] <= t
        for i in range(nd):
            tracks[i].set_data(tr["x"][keep, i] / 1000, tr["y"][keep, i] / 1000)
            if keep.any():
                heads[i].set_data(tr["x"][keep, i][-1:] / 1000, tr["y"][keep, i][-1:] / 1000)
            hl[i].set_data(tr["t"][keep] / 60, agl[keep, i])
            bl[i].set_data(tr["t"][keep] / 60, tr["E"][keep, i] / cap)
        if len(cells):
            m = dts <= t
            cy, cx = np.divmod(cells[m], n)
            xy = np.c_[(cx + 0.5) * terr.cell_size / 1000, (cy + 0.5) * terr.cell_size / 1000]
            tm = dtrue[m]
            det_t.set_offsets(xy[tm] if tm.any() else np.zeros((0, 2)))
            det_f.set_offsets(xy[~tm] if (~tm).any() else np.zeros((0, 2)))
        if keep.any():
            last = int(np.nonzero(keep)[0][-1])
            pos.set_offsets(np.c_[tr["x"][last] / 1000, tr["y"][last] / 1000])
            pos.set_facecolors([DRONE_COLORS[i % 8] for i in range(nd)])
        sup.set_text(f"FloodSim seed {world.spec.seed} ({world.spec.flood_type}), {nd} drones, "
                     f"t = {t / 60:.0f} min, found {found}/{len(world.people0.x)}")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return _save_gif(frames, path, fps)


def plot3d_html(result, world, path):
    """Interactive 3D: terrain, translucent peak water surface, drone paths (hue per drone,
    dash by mode), radio links every 10 min, thermal columns as translucent cylinders."""
    from .autopilot import MODE_NAMES

    try:
        import plotly.graph_objects as go
    except ImportError:
        return plot3d_png(result, world, Path(path).with_suffix(".png"))
    terr = world.terrain
    f = 2
    z = terr.z[::f, ::f].astype(float)
    xs = (np.arange(z.shape[1]) * f + 0.5) * terr.cell_size
    ys = (np.arange(z.shape[0]) * f + 0.5) * terr.cell_size
    fig = go.Figure()
    fig.add_trace(go.Surface(x=xs, y=ys, z=z, colorscale="Earth", showscale=False, name="terrain",
                             hoverinfo="skip"))
    d = world.flood.max_depth(0, world.episode_s)[::f, ::f]
    water = np.where(d > 0.05, z + d, np.nan)
    fig.add_trace(go.Surface(x=xs, y=ys, z=water, colorscale=[[0, "#2060c0"], [1, "#2060c0"]],
                             showscale=False, opacity=0.6, name="peak flood", hoverinfo="skip"))
    tr = result.trajectory
    groups = {"search": ("SEARCH",), "transit": ("TRANSIT", "LAUNCH", "RTB", "LOITER"),
              "thermal": ("THERMAL",), "verify": ("VERIFY",)}
    style = {"search": ("solid", 3), "transit": ("dash", 2), "thermal": ("dot", 3), "verify": ("solid", 8)}
    names = np.array(MODE_NAMES)
    for i in range(result.n_drones):
        col = DRONE_COLORS[i % len(DRONE_COLORS)]
        mn = names[tr["mode"][:, i]]
        for g, modes in groups.items():
            m = np.isin(mn, modes)
            if not m.any():
                continue
            dash, w = style[g]
            fig.add_trace(go.Scatter3d(
                x=np.where(m, tr["x"][:, i], np.nan), y=np.where(m, tr["y"][:, i], np.nan),
                z=np.where(m, tr["z"][:, i], np.nan), mode="lines", connectgaps=False,
                line=dict(color=col, width=w, dash=dash), name=f"drone {i} {g}", legendgroup=f"d{i}"))
    R = world.cfg.team.signal_radius_km * 1000
    lx, ly, lz = [], [], []
    if len(tr["t"]) > 1:
        step = max(int(round(600 / max(tr["t"][1] - tr["t"][0], 1))), 1)
        for k in range(0, len(tr["t"]), step):
            for a in range(result.n_drones):
                for b in range(a + 1, result.n_drones):
                    if np.hypot(tr["x"][k, a] - tr["x"][k, b], tr["y"][k, a] - tr["y"][k, b]) <= R:
                        lx += [tr["x"][k, a], tr["x"][k, b], None]
                        ly += [tr["y"][k, a], tr["y"][k, b], None]
                        lz += [tr["z"][k, a], tr["z"][k, b], None]
    if lx:
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", name="radio links (every 10 min)",
                                   line=dict(color="rgba(60,60,60,0.25)", width=1)))
    zi = world.spec.zi_m
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    for (tx, ty, wc, rt) in world.weather.thermal_list():
        g0 = float(terr.ground_at(np.array([tx]), np.array([ty]))[0])
        cx, cy = tx + rt * np.cos(ang), ty + rt * np.sin(ang)
        I, J, K = [], [], []
        for a in range(16):
            b = (a + 1) % 16
            I += [a, b]
            J += [b, b + 16]
            K += [a + 16, a + 16]
        fig.add_trace(go.Mesh3d(x=np.r_[cx, cx], y=np.r_[cy, cy], z=np.r_[np.full(16, g0), np.full(16, g0 + zi)],
                                i=I, j=J, k=K, color="orange", opacity=0.12, name=f"thermal {wc:.1f} m/s",
                                showlegend=False, hoverinfo="name"))
    bx, by = terr.base_xy
    fig.add_trace(go.Scatter3d(x=[bx], y=[by], z=[float(terr.ground_at(np.array([bx]), np.array([by]))[0]) + 20],
                               mode="markers", marker=dict(size=6, color="gold", symbol="diamond"), name="base"))
    zr = float(np.nanmax(tr["z"])) - float(z.min())
    fig.update_layout(
        title=f"FloodSim seed {world.spec.seed}: {world.spec.flood_type}, {result.n_drones} drones",
        font=dict(size=10),
        title_font=dict(size=13),
        scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=max(zr / terr.domain_m * 3, 0.15)),
                   xaxis=dict(title=dict(text="east (m)", font=dict(size=10)), tickfont=dict(size=8)),
                   yaxis=dict(title=dict(text="north (m)", font=dict(size=10)), tickfont=dict(size=8)),
                   zaxis=dict(title=dict(text="z (m)", font=dict(size=10)), tickfont=dict(size=8))),
        legend=dict(itemsizing="constant", font=dict(size=9)), margin=dict(l=0, r=0, t=34, b=0))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=True)
    return path


_SMALL = {"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6,
          "ytick.labelsize": 6, "legend.fontsize": 6, "figure.titlesize": 9}


def animate_paths(result, world, path, every_s=30.0, fps=12, dpi=90, tail_s=300.0):
    """Drone paths over the map: full track faint, last 5 min bright, flood underneath,
    found people as white stars."""
    terr, fl, tr = world.terrain, world.flood, result.trajectory
    snaps = result.snapshots
    nd, n = result.n_drones, terr.n
    ext = _extent(n, terr.cell_size)
    times = np.arange(0.0, tr["t"][-1] + 1e-6, every_s)
    finds = sorted(result.finds, key=lambda f: f[1])
    with plt.rc_context(_SMALL):
        fig, ax = plt.subplots(figsize=(5.2, 5.4), dpi=dpi)
        ax.imshow(0.45 + 0.55 * terrain_rgb(terr), origin="lower", extent=ext)  # lightened map
        flood_im = ax.imshow(np.zeros((n, n, 4)), origin="lower", extent=ext)
        bx, by = terr.base_xy
        ax.plot(bx / 1000, by / 1000, marker="^", ms=8, mec="k", mfc="gold", zorder=15)
        full, tail, head = [], [], []
        for i in range(nd):
            c = DRONE_COLORS[i % len(DRONE_COLORS)]
            full.append(ax.plot([], [], "-", color=c, lw=0.8, alpha=0.7)[0])
            tail.append(ax.plot([], [], "-", color=c, lw=2.2)[0])
            head.append(ax.plot([], [], "o", color=c, mec="k", mew=0.6, ms=5, zorder=14,
                                label=f"drone {i}")[0])
        found_sc = ax.scatter(np.zeros(0), np.zeros(0), marker="*", s=70, c="white", edgecolors="k",
                              lw=0.6, zorder=16, label="person found")
        ax.legend(loc="upper right", framealpha=0.85, ncol=2, handlelength=1.0)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("east (km)")
        ax.set_ylabel("north (km)")
        title = ax.set_title("Drone paths, placeholder title for layout")
        fig.tight_layout()
        frames = []
        for t in times:
            flood_im.set_data(_flood_rgba(fl.depth_grid(min(t, fl.t_end))))
            keep = tr["t"] <= t
            recent = keep & (tr["t"] >= t - tail_s)
            for i in range(nd):
                full[i].set_data(tr["x"][keep, i] / 1000, tr["y"][keep, i] / 1000)
                tail[i].set_data(tr["x"][recent, i] / 1000, tr["y"][recent, i] / 1000)
                if keep.any():
                    head[i].set_data(tr["x"][keep, i][-1:] / 1000, tr["y"][keep, i][-1:] / 1000)
            done = [f for f in finds if f[1] <= t]
            if done and len(snaps.get("t", [])):
                pts = []
                for p, tf, _, _ in done:
                    k = max(int(np.searchsorted(snaps["t"], tf, side="right")) - 1, 0)
                    pts.append((snaps["px"][k][p] / 1000, snaps["py"][k][p] / 1000))
                found_sc.set_offsets(np.asarray(pts))
            hh, mm = divmod(int((world.spec.start_hour * 3600 + t) // 60), 60)
            title.set_text(f"Drone paths, seed {world.spec.seed} ({world.spec.flood_type}), "
                           f"t+{t / 60:.0f} min ({hh % 24:02d}:{mm:02d}), found {len(done)}")
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    return _save_gif(frames, path, fps)


def animate_battery_altitude(result, world, path, every_s=30.0, fps=12, dpi=90):
    """Height above ground (top) and battery charge (bottom) for every drone, drawn over time."""
    from .autopilot import MODE_NAMES

    terr, tr = world.terrain, result.trajectory
    nd = result.n_drones
    cap = world.cfg.glider.battery_capacity_wh
    reserve = world.cfg.glider.reserve_frac * 100
    t_min = tr["t"] / 60
    agl = tr["z"] - terr.ground_at(tr["x"], tr["y"])
    on_ground = np.isin(tr["mode"], [MODE_NAMES.index("GROUND"), MODE_NAMES.index("DOWN")])
    agl = np.where(on_ground, np.nan, agl)
    batt = tr["E"] / cap * 100
    times = np.arange(0.0, tr["t"][-1] + 1e-6, every_s)
    with plt.rc_context(_SMALL):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.0, 4.6), dpi=dpi, sharex=True)
        alt_l, alt_p, bat_l, bat_p = [], [], [], []
        for i in range(nd):
            c = DRONE_COLORS[i % len(DRONE_COLORS)]
            alt_l.append(a1.plot([], [], "-", color=c, lw=0.9, label=f"drone {i}")[0])
            alt_p.append(a1.plot([], [], "o", color=c, mec="k", mew=0.5, ms=4)[0])
            bat_l.append(a2.plot([], [], "-", color=c, lw=0.9)[0])
            bat_p.append(a2.plot([], [], "o", color=c, mec="k", mew=0.5, ms=4)[0])
        a1.set_xlim(0, t_min[-1])
        a1.set_ylim(0, max(float(np.nanmax(agl)) * 1.05, 100))
        a1.set_ylabel("height above ground (m)")
        a1.axhline(world.cfg.team.ceiling_agl_m, color="0.5", lw=0.6, ls=":")
        a1.text(1, world.cfg.team.ceiling_agl_m, " ceiling", va="bottom", color="0.4")
        a1.legend(loc="upper right", ncol=min(nd, 4), framealpha=0.85, handlelength=1.2)
        a2.set_ylim(0, 102)
        a2.set_ylabel("battery (%)")
        a2.set_xlabel("minutes since episode start")
        a2.axhline(reserve, color="crimson", lw=0.6, ls="--")
        a2.text(1, reserve, f" {reserve:.0f}% landing reserve", va="bottom", color="crimson")
        for a in (a1, a2):
            a.grid(alpha=0.25, lw=0.4)
        title = a1.set_title("Altitude and battery, placeholder title for layout")
        fig.tight_layout()
        frames = []
        for t in times:
            keep = tr["t"] <= t
            for i in range(nd):
                alt_l[i].set_data(t_min[keep], agl[keep, i])
                bat_l[i].set_data(t_min[keep], batt[keep, i])
                if keep.any():
                    k = int(np.nonzero(keep)[0][-1])
                    alt_p[i].set_data(t_min[k:k + 1], agl[k:k + 1, i])
                    bat_p[i].set_data(t_min[k:k + 1], batt[k:k + 1, i])
            title.set_text(f"Altitude and battery, {nd} drones, t+{t / 60:.0f} min")
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    return _save_gif(frames, path, fps)


def plot3d_png(result, world, path):
    """matplotlib fallback when plotly is missing."""
    terr = world.terrain
    f = 4
    z = terr.z[::f, ::f]
    c = (np.arange(z.shape[0]) * f + 0.5) * terr.cell_size
    X, Y = np.meshgrid(c, c)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, z, cmap="terrain", linewidth=0, alpha=0.8)
    tr = result.trajectory
    for i in range(result.n_drones):
        ax.plot(tr["x"][:, i], tr["y"][:, i], tr["z"][:, i], color=DRONE_COLORS[i % 8], lw=1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def episode_report(result) -> str:
    """Metrics table and safety events for one episode."""
    m = result.metrics
    lines = [f"Episode: seed {result.seed}, controller {result.controller}, {result.n_drones} drones", "-" * 60]
    for k, v in m.items():
        lines.append(f"{k:28s} {v:.4g}" if isinstance(v, float) else f"{k:28s} {v}")
    lines.append("-" * 60)
    lines.append("mode time (drone-seconds): " + ", ".join(f"{k} {v:.0f}" for k, v in result.mode_time.items() if v))
    ev = [e for e in result.safety_events if e[1] != "landed"]
    lines.append(f"safety events ({len(ev)}):")
    for t, kind, who, info in ev[:30]:
        lines.append(f"  t={t:7.0f}s {kind:20s} {who} {info}")
    lines.append(f"finds ({len(result.finds)}): " +
                 ", ".join(f"person {p} at {t / 60:.0f} min" for p, t, _, _ in result.finds))
    return "\n".join(lines)
