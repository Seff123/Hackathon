"""V5: 3D view (plotly, offline HTML): terrain, water at 6 times, drone paths, finds, base, thermals,
optional lives-at-stake drape, camera presets."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..viz import DRONE_COLOURS, _segments_by_mode, hillshade


def demo3d(result, world, path, exaggeration: float = 3.0, n_times: int = 6, drape=None):
    try:
        import plotly.graph_objects as go
    except ImportError:
        from ..viz import plot3d_png
        return plot3d_png(result, world, Path(path).with_suffix(".png"))
    ter = world.terrain
    f = 2
    z = ter.z.reshape(ter.n // f, f, ter.n // f, f).mean(axis=(1, 3))
    c = ter.cell_size * f
    xs = (np.arange(z.shape[1]) + 0.5) * c
    z0 = float(z.min())
    ex = exaggeration
    lc = ter.landcover[::f, ::f]
    hs = hillshade(z, c)
    colour = np.choose(np.clip(lc, 0, 3), [0.55, 0.12, 0.85, 0.32]) * 0.7 + 0.3 * hs
    fig = go.Figure()
    fig.add_trace(go.Surface(x=xs, y=xs, z=(z - z0) * ex, surfacecolor=colour, colorscale="Earth", cmin=0, cmax=1,
                             showscale=False, name="terrain (land cover)", hoverinfo="skip"))
    n_base = 1
    if drape is not None:
        fig.add_trace(go.Surface(x=xs, y=xs, z=(z - z0) * ex + 1.0, surfacecolor=drape, colorscale="YlOrBr",
                                 showscale=False, opacity=1.0, visible=False, name="lives at stake (drape)", hoverinfo="skip"))
        n_base = 2
    ep = world.cfg.world.episode_s
    times = np.linspace(0, ep, n_times)
    for k, t in enumerate(times):
        d = world.flood.depth_grid(t)
        wz = np.where(d > 0.1, (z + d - z0) * ex + 0.6, np.nan)
        fig.add_trace(go.Surface(x=xs, y=xs, z=wz, colorscale=[[0, "#1f6fc2"], [1, "#1f6fc2"]], opacity=0.6, showscale=False,
                                 visible=(k == n_times - 1), name=f"water (truth) {t / 60:.0f} min", hoverinfo="name"))
    T = result.trajectory
    dash = {"search": "solid", "transit": "dash", "thermal": "dot", "verify": "solid", "rtb": "longdash"}
    for i in range(result.n_drones):
        for mode, seg in _segments_by_mode(T, i):
            if mode in ("ground", "down") or len(seg) < 2:
                continue
            seg = seg[::2] if len(seg) > 400 else seg
            fig.add_trace(go.Scatter3d(x=seg[:, 2], y=seg[:, 3], z=(seg[:, 4] - z0) * ex, mode="lines",
                                       line=dict(color=DRONE_COLOURS[i % 8] if mode != "rtb" else "#888",
                                                 width=6 if mode == "verify" else 3, dash=dash.get(mode, "solid")),
                                       name=f"G{i + 1} {mode}", showlegend=False, hoverinfo="name"))
    if result.recording and result.recording["frames"]:
        P = result.recording["frames"][-1]["people"]
        for fd in result.finds:
            k = fd["person"]
            px, py = float(P[k, 0]), float(P[k, 1])
            pz = (ter.z[min(int(py // ter.cell_size), ter.n - 1), min(int(px // ter.cell_size), ter.n - 1)] - z0) * ex
            fig.add_trace(go.Scatter3d(x=[px], y=[py], z=[pz + 15], mode="markers", marker=dict(color="#23845a", size=6),
                                       name="confirmed survivor", showlegend=False))
    bx, by = ter.base_xy
    bz = (ter.z[int(by // ter.cell_size), int(bx // ter.cell_size)] - z0) * ex
    fig.add_trace(go.Cone(x=[bx], y=[by], z=[bz + 150], u=[0], v=[0], w=[-120], sizeref=250, showscale=False,
                          colorscale=[[0, "black"], [1, "black"]], name="base"))
    zi = world.weather.zi
    for th in world.weather.thermals():
        a = np.linspace(0, 2 * np.pi, 20)
        A, H = np.meshgrid(a, np.array([0.0, zi]))
        gz = (ter.z[min(int(th.y // ter.cell_size), ter.n - 1), min(int(th.x // ter.cell_size), ter.n - 1)] - z0) * ex
        fig.add_trace(go.Surface(x=th.x + th.r_t * np.cos(A), y=th.y + th.r_t * np.sin(A), z=gz + H * ex, showscale=False,
                                 opacity=0.13, colorscale=[[0, "#e4572e"], [1, "#e4572e"]], hoverinfo="skip", name="thermal"))
    n_tr = len(fig.data)
    buttons = []
    for k, t in enumerate(times):
        vis = [True] * n_tr
        for j in range(n_times):
            vis[n_base + j] = (j == k)
        if drape is not None:
            vis[1] = False
        buttons.append(dict(label=f"{t / 60:.0f} min", method="update", args=[{"visible": vis}]))
    if drape is not None:
        vis = [True] * n_tr
        for j in range(n_times):
            vis[n_base + j] = (j == n_times - 1)
        buttons.append(dict(label="drape: lives at stake", method="update", args=[{"visible": vis}]))
    L = world.cfg.world.domain_m
    g1 = T[T[:, 1] == 0]
    if len(g1):
        mx, my = g1[:, 2].mean() / L - 0.5, g1[:, 3].mean() / L - 0.5
    else:
        mx = my = 0.0
    cams = {"overview": dict(eye=dict(x=1.35, y=-1.35, z=0.85)),
            "follow G1": dict(eye=dict(x=mx * 1.2 + 0.25, y=my * 1.2 - 0.45, z=0.25), center=dict(x=mx, y=my, z=-0.1)),
            "valley floor": dict(eye=dict(x=0.25, y=-0.7, z=0.12), center=dict(x=0, y=0, z=-0.15))}
    fig.update_layout(
        title=f"FloodSim seed {world.spec.seed}: {result.controller}, {result.n_drones} gliders (water = truth; vertical ×{ex:g})",
        scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.35), xaxis=dict(range=[0, L], title="east (m)"),
                   yaxis=dict(range=[0, L], title="north (m)"), zaxis=dict(title=f"height ×{ex:g}")),
        updatemenus=[dict(type="buttons", buttons=buttons, x=0.0, y=1.0, xanchor="left", direction="left"),
                     dict(type="buttons", x=0.0, y=0.93, xanchor="left", direction="left",
                          buttons=[dict(label=k, method="relayout", args=[{"scene.camera": v}]) for k, v in cams.items()])],
        margin=dict(l=0, r=0, t=40, b=0), template="plotly_white")
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    return str(path)
