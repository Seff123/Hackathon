"""Ground-truth flood: vectorised local-inertial shallow-water solver (Bates et al. 2010) on a 100 m grid.

All times are episode time in seconds (0 = episode start). Flood onset is at t_on = -start_offset.
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config, ScenarioSpec
from .grid import bilinear, upsample2
from .people import hazard_rating
from .scenario import channel_slope, hydrograph
from .terrain import FOREST, OPEN, URBAN, Terrain, rasterize4

FLOOD_VERSION = "v3"


# ----------------------------------------------------------------------------- coarse grid
@dataclass
class CoarseGrid:
    z: np.ndarray
    n_cell: np.ndarray
    river: np.ndarray
    river_cells: np.ndarray       # [M, 2] (iy, ix) from inflow to outlet
    levee: np.ndarray
    runoff: np.ndarray
    outlet: np.ndarray            # bool mask of outlet-edge cells
    outlet_slope: np.ndarray
    cell: float
    bank: np.ndarray              # z without levees


def _block(a: np.ndarray, f: int, fn):
    ny, nx = a.shape
    return fn(a.reshape(ny // f, f, nx // f, f), axis=(1, 3))


def coarse_grid(spec: ScenarioSpec, terrain: Terrain, cfg: Config) -> CoarseGrid:
    f = int(round(cfg.world.flood_cell_m / terrain.cell_size))
    dx = cfg.world.flood_cell_m
    nf = cfg.world.nf
    fine_land = np.where(terrain.river_mask, np.nan, terrain.z_bare.astype(np.float64))
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bank = _block(fine_land, f, np.nanmean)
    bank = np.where(np.isnan(bank), _block(terrain.z_bare.astype(np.float64), f, np.max), bank)
    z = bank.copy()
    # river path at 100 m, 4-connected, bed from the centreline
    cells = rasterize4(terrain.river_xy, dx, nf)
    river = np.zeros((nf, nf), dtype=bool)
    river[cells[:, 0], cells[:, 1]] = True
    bed = terrain.river_bed[cells[:, 2]]
    # enforce a monotone bed so the coarse channel never pools against itself
    bed = np.minimum.accumulate(bed)
    z[cells[:, 0], cells[:, 1]] = bed
    # levees: coarse cells holding fine levee cells (not river)
    lev = _block(terrain.levee_mask.astype(np.int8), f, np.max).astype(bool) & ~river
    if spec.levee_height_m > 0.05:
        z[lev] = bank[lev] + spec.levee_height_m
    lc = terrain.landcover
    frac = {k: _block((lc == k).astype(np.float64), f, np.mean) for k in (OPEN, URBAN, FOREST)}
    n_cell = (frac[OPEN] * spec.manning_floodplain + frac[URBAN] * spec.manning_urban +
              frac[FOREST] * spec.manning_forest)
    tot = frac[OPEN] + frac[URBAN] + frac[FOREST]
    n_cell = np.where(tot > 0, n_cell / np.maximum(tot, 1e-9), spec.manning_floodplain)
    n_eff = spec.manning_channel * dx / spec.channel_width_m
    n_cell = np.where(river, n_eff, n_cell)
    fp = cfg.flood
    runoff = (frac[OPEN] * fp.runoff_open + frac[URBAN] * fp.runoff_urban + frac[FOREST] * fp.runoff_forest)
    runoff = np.where(tot > 0, runoff / np.maximum(tot, 1e-9), 0.0)
    runoff = np.where(river, 1.0, runoff)
    from .terrain import edge_mask
    outlet = edge_mask(nf, spec.outlet_edge)
    outlet_slope = np.where(river, channel_slope(spec), max(spec.regional_slope, 1e-3))
    return CoarseGrid(z=z, n_cell=n_cell, river=river, river_cells=cells[:, :2], levee=lev, runoff=runoff,
                      outlet=outlet, outlet_slope=outlet_slope, cell=dx, bank=bank)


# ----------------------------------------------------------------------------- solver
@dataclass
class SolverState:
    h: np.ndarray
    qx: np.ndarray   # [ny, nx+1]
    qy: np.ndarray   # [ny+1, nx]
    t: float
    vol_in: float = 0.0
    vol_out: float = 0.0
    vol_clip: float = 0.0


class LocalInertialSolver:
    def __init__(self, z, n_cell, dx, cfg_flood, outlet=None, outlet_slope=None, outlet_edge="S"):
        self.z = z.astype(np.float64)
        self.dx = float(dx)
        self.fp = cfg_flood
        n = n_cell.astype(np.float64)
        self.n2x = (0.5 * (n[:, :-1] + n[:, 1:])) ** 2
        self.n2y = (0.5 * (n[:-1, :] + n[1:, :])) ** 2
        self.n_cell = n
        self.outlet = outlet if outlet is not None else np.zeros_like(z, dtype=bool)
        self.outlet_slope = outlet_slope if outlet_slope is not None else np.full_like(self.z, 1e-3)
        self.outlet_edge = outlet_edge
        self.zmax_x = np.maximum(self.z[:, :-1], self.z[:, 1:])
        self.zmax_y = np.maximum(self.z[:-1, :], self.z[1:, :])

    def set_z(self, z):
        self.z = z.astype(np.float64)
        self.zmax_x = np.maximum(self.z[:, :-1], self.z[:, 1:])
        self.zmax_y = np.maximum(self.z[:-1, :], self.z[1:, :])

    def max_dt(self, h) -> float:
        hm = float(h.max())
        if hm <= 1e-6:
            return self.fp.dt_max
        return float(np.clip(self.fp.cfl * self.dx / math.sqrt(self.fp.g * hm), self.fp.dt_min, self.fp.dt_max))

    def step(self, s: SolverState, dt: float, rain=None, src=None) -> None:
        """Advance by dt. rain: [ny,nx] m/s of effective (runoff) rain; src: list of (iy, ix, Q m3/s)."""
        g, dx, tol = self.fp.g, self.dx, self.fp.wet_tol
        h, qx, qy = s.h, s.qx, s.qy
        eta = self.z + h
        # x faces
        hf = np.maximum(eta[:, :-1], eta[:, 1:]) - self.zmax_x
        q = qx[:, 1:-1]
        hfc = np.maximum(hf, tol)
        qn = (q - g * hfc * dt * (eta[:, 1:] - eta[:, :-1]) / dx) / (1.0 + g * dt * self.n2x * np.abs(q) / hfc ** (7.0 / 3.0))
        qn[hf < tol] = 0.0
        qx[:, 1:-1] = qn
        # y faces
        hf = np.maximum(eta[:-1, :], eta[1:, :]) - self.zmax_y
        q = qy[1:-1, :]
        hfc = np.maximum(hf, tol)
        qn = (q - g * hfc * dt * (eta[1:, :] - eta[:-1, :]) / dx) / (1.0 + g * dt * self.n2y * np.abs(q) / hfc ** (7.0 / 3.0))
        qn[hf < tol] = 0.0
        qy[1:-1, :] = qn
        # outlet boundary: free outflow at normal depth (zero-gradient surface)
        if self.outlet.any():
            qo = np.where(self.outlet & (h > tol), h ** (5.0 / 3.0) * np.sqrt(self.outlet_slope) / self.n_cell, 0.0)
            e = self.outlet_edge
            if e == "S":
                qy[0, :] = -qo[0, :]
            elif e == "N":
                qy[-1, :] = qo[-1, :]
            elif e == "W":
                qx[:, 0] = -qo[:, 0]
            else:
                qx[:, -1] = qo[:, -1]
        # limit outgoing fluxes so no cell goes negative
        out = (np.maximum(qx[:, 1:], 0) + np.maximum(-qx[:, :-1], 0) +
               np.maximum(qy[1:, :], 0) + np.maximum(-qy[:-1, :], 0))
        need = out * dt
        avail = h * dx
        fac = np.where(need > avail, avail / np.maximum(need, 1e-30), 1.0)
        if (fac < 1.0).any():
            fpx = np.pad(fac, ((0, 0), (1, 1)), constant_values=1.0)
            qx *= np.where(qx > 0, fpx[:, :-1], fpx[:, 1:])
            fpy = np.pad(fac, ((1, 1), (0, 0)), constant_values=1.0)
            qy *= np.where(qy > 0, fpy[:-1, :], fpy[1:, :])
        # boundary outflow volume
        e = self.outlet_edge
        if e == "S":
            bout = -qy[0, :].sum()
        elif e == "N":
            bout = qy[-1, :].sum()
        elif e == "W":
            bout = -qx[:, 0].sum()
        else:
            bout = qx[:, -1].sum()
        dh = dt * ((qx[:, :-1] - qx[:, 1:]) + (qy[:-1, :] - qy[1:, :])) / dx
        vin = 0.0
        if rain is not None:
            dh += dt * rain
            vin += float(rain.sum()) * dt * dx * dx
        if src:
            for iy, ix, Q in src:
                dh[iy, ix] += dt * Q / (dx * dx)
                vin += Q * dt
        h += dh
        neg = h < 0
        if neg.any():
            s.vol_clip += float(-h[neg].sum()) * dx * dx
            h[neg] = 0.0
        s.vol_in += vin
        s.vol_out += float(bout) * dx * dt
        s.t += dt

    def velocity(self, s: SolverState):
        h = s.h
        u = 0.5 * (s.qx[:, :-1] + s.qx[:, 1:])
        v = 0.5 * (s.qy[:-1, :] + s.qy[1:, :])
        ok = h >= self.fp.vel_min_depth
        hh = np.where(ok, h, 1.0)
        return np.where(ok, u / hh, 0.0), np.where(ok, v / hh, 0.0)


# ----------------------------------------------------------------------------- storm
@dataclass
class Storm:
    active: bool
    t_on: float
    duration: float
    i_peak: float          # m/s
    radius: float
    cx0: float
    cy0: float
    vx: float
    vy: float

    def centre(self, t):
        tm = self.t_on + 0.5 * self.duration
        return self.cx0 + self.vx * (t - tm), self.cy0 + self.vy * (t - tm)

    def envelope(self, t):
        if not self.active:
            return 0.0
        tau = (t - self.t_on) / self.duration
        if tau <= 0 or tau >= 1:
            return 0.0
        return math.sin(math.pi * tau)

    def rain(self, x, y, t):
        """Rain intensity (m/s) at world points."""
        env = self.envelope(t)
        if env <= 0:
            return np.zeros(np.broadcast(np.asarray(x), np.asarray(y)).shape)
        cx, cy = self.centre(t)
        r2 = (np.asarray(x) - cx) ** 2 + (np.asarray(y) - cy) ** 2
        return self.i_peak * env * np.exp(-r2 / (2 * self.radius ** 2))

    def accumulated(self, x, y, t, n_sub: int = 24):
        """Rain depth (m) fallen from storm start to t."""
        if not self.active or t <= self.t_on:
            return np.zeros(np.broadcast(np.asarray(x), np.asarray(y)).shape)
        t1 = min(t, self.t_on + self.duration)
        ts = np.linspace(self.t_on, t1, n_sub + 1)
        acc = 0.0
        for a, b in zip(ts[:-1], ts[1:]):
            acc = acc + self.rain(x, y, 0.5 * (a + b)) * (b - a)
        return acc


def make_storm(spec: ScenarioSpec, cfg: Config, rng) -> Storm:
    L = cfg.world.domain_m
    t_on = -spec.start_offset_h * 3600.0
    sp = float(np.clip(cfg.flood.storm_speed_factor * spec.wind10_ms, *cfg.flood.storm_speed_bounds))
    th = spec.wind_to_rad + math.radians(spec.storm_track_deg)
    cx0 = L / 2 + rng.uniform(-0.2, 0.2) * L
    cy0 = L / 2 + rng.uniform(-0.2, 0.2) * L
    return Storm(active=spec.has_pluvial, t_on=t_on, duration=spec.storm_duration_h * 3600.0,
                 i_peak=spec.peak_rain_mmh * spec.climate_factor / 3.6e6, radius=spec.storm_radius_km * 1e3,
                 cx0=cx0, cy0=cy0, vx=sp * math.cos(th), vy=sp * math.sin(th))


def downscale_depth(d100, z_coarse, z50, river50) -> np.ndarray:
    d_up = np.maximum(upsample2(d100), 0)
    eta_up = upsample2(z_coarse + d100)
    head = np.maximum(eta_up - z50, 0)
    return np.where(river50, head, np.minimum(d_up, head)).astype(np.float32)


# ----------------------------------------------------------------------------- timeline
@dataclass
class FloodTimeline:
    depth_snaps: np.ndarray        # [T, ny, nx] float16, every dt_depth from t=0
    vel_snaps: np.ndarray          # [Tv, 2, ny, nx] float16, every dt_vel from t=0
    dt_depth: float
    dt_vel: float
    cell: float
    mass: dict = field(default_factory=dict)
    breach: dict = field(default_factory=dict)
    z_coarse: np.ndarray | None = None
    z50: np.ndarray | None = None
    river50: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def t_end(self) -> float:
        return (self.depth_snaps.shape[0] - 1) * self.dt_depth

    def _tidx(self, t, dt, T):
        f = np.clip(t / dt, 0, T - 1 - 1e-9)
        i = int(f)
        return i, f - i

    def depth_grid(self, t: float) -> np.ndarray:
        i, a = self._tidx(t, self.dt_depth, self.depth_snaps.shape[0])
        d0 = self.depth_snaps[i].astype(np.float32)
        if a <= 1e-9:
            return d0
        return d0 * (1 - a) + self.depth_snaps[i + 1].astype(np.float32) * a

    def velocity_grid(self, t: float):
        i, a = self._tidx(t, self.dt_vel, self.vel_snaps.shape[0])
        v = self.vel_snaps[i].astype(np.float32)
        if a > 1e-9:
            v = v * (1 - a) + self.vel_snaps[i + 1].astype(np.float32) * a
        return v[0], v[1]

    def hazard_grid(self, t: float) -> np.ndarray:
        d = self.depth_grid(t)
        u, v = self.velocity_grid(t)
        return hazard_rating(d, np.hypot(u, v)).astype(np.float32)

    def depth(self, x, y, t: float):
        """True depth at world points (50 m resolution, linear in time)."""
        return np.maximum(bilinear(self.depth_grid50(t), x, y, self.cell / 2), 0.0)

    def velocity(self, x, y, t: float):
        u, v = self.velocity_grid(t)
        return bilinear(u, x, y, self.cell), bilinear(v, x, y, self.cell)

    def _d50_snap(self, i: int) -> np.ndarray:
        key = ("d50", i)
        if key not in self._cache:
            if len(self._cache) > 96:
                for k in [k for k in self._cache if k[0] == "d50"][:48]:
                    del self._cache[k]
            self._cache[key] = self.fine_depth(self.depth_snaps[i].astype(np.float32))
        return self._cache[key]

    def fine_depth(self, d100: np.ndarray) -> np.ndarray:
        """Downscale a 100 m depth grid to 50 m using the water surface against the 50 m terrain.

        Plain linear upsampling smears channel depth onto the banks; instead the water surface is
        interpolated and compared with the fine terrain, capped by the interpolated depth off-channel."""
        if self.z50 is None:
            return np.maximum(upsample2(d100), 0)
        return downscale_depth(d100, self.z_coarse, self.z50, self.river50)

    def depth_grid50(self, t: float) -> np.ndarray:
        i, a = self._tidx(t, self.dt_depth, self.depth_snaps.shape[0])
        d0 = self._d50_snap(i)
        if a <= 1e-6:
            return d0
        return d0 * (1 - a) + self._d50_snap(i + 1) * a

    def velocity_grid50(self, t: float):
        u, v = self.velocity_grid(t)
        return upsample2(u), upsample2(v)

    def hazard_grid50(self, t: float) -> np.ndarray:
        d = self.depth_grid50(t)
        u, v = self.velocity_grid50(t)
        return hazard_rating(d, np.hypot(u, v)).astype(np.float32)

    def max_depth(self) -> np.ndarray:
        if "maxd" not in self._cache:
            self._cache["maxd"] = self.depth_snaps.max(axis=0).astype(np.float32)
        return self._cache["maxd"]

    def max_depth50(self) -> np.ndarray:
        if "maxd50" not in self._cache:
            m = np.zeros_like(self._d50_snap(0))
            for i in range(self.depth_snaps.shape[0]):
                m = np.maximum(m, self.fine_depth(self.depth_snaps[i].astype(np.float32)))
            self._cache["maxd50"] = m
        return self._cache["maxd50"]

    def attach_terrain(self, terrain) -> None:
        self.z50 = terrain.z.astype(np.float32)
        self.river50 = terrain.river_mask
        self._cache.clear()

    @staticmethod
    def upsample(grid: np.ndarray) -> np.ndarray:
        return upsample2(grid)


def run_flood(spec: ScenarioSpec, terrain: Terrain, cfg: Config, rng=None) -> FloodTimeline:
    wc = cfg.world
    fp = cfg.flood
    rng = rng if rng is not None else np.random.default_rng(0)
    cg = coarse_grid(spec, terrain, cfg)
    dx = cg.cell
    storm = make_storm(spec, cfg, rng)
    t_on = -spec.start_offset_h * 3600.0
    t_start = min(t_on, 0.0) - wc.spinup_s
    t_end = wc.episode_s + wc.post_episode_s
    solver = LocalInertialSolver(cg.z, cg.n_cell, dx, fp, outlet=cg.outlet, outlet_slope=cg.outlet_slope,
                                 outlet_edge=spec.outlet_edge)
    ny, nx = cg.z.shape
    h = np.zeros((ny, nx))
    # initial channel at Manning normal depth for base flow
    qb = float(hydrograph(spec, t_start - t_on))
    S = channel_slope(spec)
    h0 = (qb * spec.manning_channel / (spec.channel_width_m * math.sqrt(S))) ** 0.6
    h[cg.river] = h0
    st = SolverState(h=h, qx=np.zeros((ny, nx + 1)), qy=np.zeros((ny + 1, nx)), t=t_start)
    vol0 = float(h.sum()) * dx * dx
    inflow = (int(cg.river_cells[0, 0]), int(cg.river_cells[0, 1]))
    # breach set-up
    breach = {}
    if spec.has_breach and cg.levee.any():
        lv = np.argwhere(cg.levee)
        # breach the levee section closest to the largest town
        tw = max(terrain.towns, key=lambda t: t["r"])
        d = np.hypot((lv[:, 1] + 0.5) * dx - tw["x"], (lv[:, 0] + 0.5) * dx - tw["y"])
        k0 = int(np.argmin(d))
        nb = max(1, int(math.ceil(spec.breach_width_m / dx)))
        dd = np.hypot(lv[:, 0] - lv[k0, 0], lv[:, 1] - lv[k0, 1])
        sel = lv[np.argsort(dd)[:nb]]
        breach = {"t": spec.breach_time_h * 3600.0, "cells": sel.tolist(), "done": False,
                  "x": float((lv[k0, 1] + 0.5) * dx), "y": float((lv[k0, 0] + 0.5) * dx)}
    # rain grid coordinates
    xs = (np.arange(nx) + 0.5) * dx
    X, Y = np.meshgrid(xs, xs)
    n_d = int(round(t_end / wc.depth_snapshot_s)) + 1
    vel_every = int(round(wc.velocity_snapshot_s / wc.depth_snapshot_s))
    n_v = (n_d - 1) // vel_every + 1
    D = np.zeros((n_d, ny, nx), dtype=np.float16)
    V = np.zeros((n_v, 2, ny, nx), dtype=np.float16)
    k = 0
    z_cur = cg.z.copy()
    while k < n_d:
        t_snap = k * wc.depth_snapshot_s
        if st.t >= t_snap - 1e-9:
            D[k] = st.h.astype(np.float16)
            if k % vel_every == 0:
                u, v = solver.velocity(st)
                V[k // vel_every, 0] = u.astype(np.float16)
                V[k // vel_every, 1] = v.astype(np.float16)
            k += 1
            continue
        dt = min(solver.max_dt(st.h), t_snap - st.t)
        tm = st.t + 0.5 * dt
        if breach and not breach["done"] and st.t >= breach["t"]:
            for iy, ix in breach["cells"]:
                z_cur[iy, ix] = max(z_cur[iy, ix] - spec.breach_head_m, cg.bank[iy, ix] - 1.0)
            solver.set_z(z_cur)
            breach["done"] = True
        rain = None
        if storm.active and storm.envelope(tm) > 0:
            rain = storm.rain(X, Y, tm) * cg.runoff
        Q = float(hydrograph(spec, tm - t_on))
        solver.step(st, dt, rain=rain, src=[(inflow[0], inflow[1], Q)])
    stored = float(st.h.sum()) * dx * dx - vol0
    resid = stored - (st.vol_in - st.vol_out)
    mass = {"vol_in": st.vol_in, "vol_out": st.vol_out, "stored": stored, "residual": resid,
            "residual_frac": abs(resid) / max(st.vol_in, 1.0), "clip": st.vol_clip, "t_start": t_start,
            "t_end": t_end}
    return FloodTimeline(depth_snaps=D, vel_snaps=V, dt_depth=wc.depth_snapshot_s, dt_vel=wc.velocity_snapshot_s,
                         cell=dx, mass=mass, breach={k_: v for k_, v in breach.items() if k_ != "done"},
                         z_coarse=cg.z.astype(np.float32), z50=terrain.z.astype(np.float32),
                         river50=terrain.river_mask)


def cache_path(spec: ScenarioSpec, cache_dir) -> Path:
    return Path(cache_dir) / f"{spec.spec_hash}_{FLOOD_VERSION}.npz"


def build_flood(spec: ScenarioSpec, terrain: Terrain, cfg: Config, cache_dir=None):
    """Returns (FloodTimeline, loaded_from_cache)."""
    from . import rng as rngmod
    if cache_dir is not None:
        p = cache_path(spec, cache_dir)
        if p.exists():
            with np.load(p, allow_pickle=False) as f:
                tl = FloodTimeline(depth_snaps=f["D"], vel_snaps=f["V"], dt_depth=float(f["dt_depth"]),
                                   dt_vel=float(f["dt_vel"]), cell=float(f["cell"]),
                                   mass={k[5:]: float(f[k]) for k in f.files if k.startswith("mass_")},
                                   z_coarse=f["z_coarse"])
                if "breach_x" in f.files:
                    tl.breach = {"t": float(f["breach_t"]), "x": float(f["breach_x"]), "y": float(f["breach_y"]),
                                 "cells": f["breach_cells"].tolist()}
            tl.attach_terrain(terrain)
            return tl, True
    tl = run_flood(spec, terrain, cfg, rngmod.stream(spec.seed, spec.attempt, "flood"))
    if cache_dir is not None:
        p = cache_path(spec, cache_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        extra = {f"mass_{k}": v for k, v in tl.mass.items()}
        if tl.breach:
            extra.update(breach_t=tl.breach["t"], breach_x=tl.breach["x"], breach_y=tl.breach["y"],
                         breach_cells=np.asarray(tl.breach["cells"]))
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, D=tl.depth_snaps, V=tl.vel_snaps, dt_depth=tl.dt_depth, dt_vel=tl.dt_vel,
                            cell=tl.cell, z_coarse=tl.z_coarse, **extra)
        tmp.replace(p)
    return tl, False
