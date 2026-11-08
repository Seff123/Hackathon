"""Ground-truth flood: vectorised local-inertial solver (Bates et al. 2010),
river/rain/breach sources, FloodTimeline and the EA hazard rating."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .config import Config, ScenarioSpec
from .grid import bilinear, block_reduce, cell_centres
from .terrain import FOREST, OPEN, URBAN, WATER, Terrain

G = 9.81


# --------------------------------------------------------------------------
# Hazard rating (Environment Agency FD2320, simplified debris factor)
# --------------------------------------------------------------------------


def debris_factor(d):
    d = np.asarray(d, float)
    return np.where(d <= 0.25, 0.0, np.where(d <= 0.75, 0.5, 1.0))


def hazard_rating(d, v):
    d = np.asarray(d, float)
    return np.where(d > 0, d * (np.asarray(v, float) + 0.5) + debris_factor(d), 0.0)


HAZARD_CLASSES = ((0.75, "low"), (1.25, "danger for some"), (2.0, "danger for most"), (np.inf, "danger for all"))


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------


class LocalInertialSolver:
    """Staggered-grid local-inertial shallow water.

    h: (ny, nx) depth at centres. qx: (ny, nx+1) unit discharge on x faces
    (face j sits between cells j-1 and j); qy: (ny+1, nx) likewise in y.
    Domain edges are walls except cells flagged in `outlet`, which drain
    freely across the edge named by `outlet_edge`.
    """

    def __init__(self, z, n_manning, dx, outlet=None, outlet_edge="E", cfl=0.7,
                 dt_min=0.5, dt_max=10.0, wet_tol=1e-3, froude_limit=1.0, outlet_slope=1e-3):
        self.z = np.asarray(z, np.float64)
        ny, nx = self.z.shape
        self.ny, self.nx = ny, nx
        self.n = np.broadcast_to(np.asarray(n_manning, np.float64), (ny, nx)).copy()
        self.n2x = (0.5 * (self.n[:, :-1] + self.n[:, 1:])) ** 2
        self.n2y = (0.5 * (self.n[:-1, :] + self.n[1:, :])) ** 2
        self.dx = float(dx)
        self.h = np.zeros((ny, nx))
        self.qx = np.zeros((ny, nx + 1))
        self.qy = np.zeros((ny + 1, nx))
        self.cfl, self.dt_min, self.dt_max = cfl, dt_min, dt_max
        self.wet_tol, self.froude = wet_tol, froude_limit
        self.outlet = np.zeros((ny, nx), bool) if outlet is None else np.asarray(outlet, bool)
        self.outlet_edge = outlet_edge
        self.outlet_slope = outlet_slope
        # flow width / dx per face, used in continuity only (q stays per unit width)
        self.wx = np.ones((ny, nx + 1))
        self.wy = np.ones((ny + 1, nx))
        self.vol_out = 0.0
        self.vol_in = 0.0

    def stable_dt(self):
        hmax = float(self.h.max())
        if hmax <= 1e-6:
            return self.dt_max
        return float(np.clip(self.cfl * self.dx / math.sqrt(G * hmax), self.dt_min, self.dt_max))

    def _face_update(self, q, eta_a, eta_b, z_a, z_b, n2, dt):
        hf = np.maximum(eta_a, eta_b) - np.maximum(z_a, z_b)
        dry = hf < self.wet_tol
        hf_s = np.where(dry, 1.0, hf)
        num = q - G * hf_s * dt * (eta_b - eta_a) / self.dx
        den = 1.0 + G * dt * n2 * np.abs(q) / hf_s ** (7.0 / 3.0)
        qn = num / den
        if self.froude:
            lim = self.froude * hf_s * np.sqrt(G * hf_s)
            np.clip(qn, -lim, lim, out=qn)
        qn[dry] = 0.0
        return qn

    def _outlet_flux(self, eta):
        """Free outflow (normal depth with the local surface slope) across the outlet edge."""
        e = self.outlet_edge
        h, z = self.h, self.z
        if e == "E":
            he, hi_eta, he_eta, mask, nn = h[:, -1], eta[:, -2], eta[:, -1], self.outlet[:, -1], self.n[:, -1]
        elif e == "W":
            he, hi_eta, he_eta, mask, nn = h[:, 0], eta[:, 1], eta[:, 0], self.outlet[:, 0], self.n[:, 0]
        elif e == "N":
            he, hi_eta, he_eta, mask, nn = h[-1, :], eta[-2, :], eta[-1, :], self.outlet[-1, :], self.n[-1, :]
        else:
            he, hi_eta, he_eta, mask, nn = h[0, :], eta[1, :], eta[0, :], self.outlet[0, :], self.n[0, :]
        s = np.maximum((hi_eta - he_eta) / self.dx, self.outlet_slope)
        q = np.where(mask & (he > self.wet_tol), he ** (5.0 / 3.0) * np.sqrt(s) / nn, 0.0)
        lim = self.froude * he * np.sqrt(G * he) if self.froude else np.inf
        return np.minimum(q, lim)

    def step(self, dt, src=None):
        """Advance by dt. src: (ny, nx) source rate in m/s (>= 0) or None."""
        h, z = self.h, self.z
        eta = z + h
        qx, qy = self.qx, self.qy
        qx[:, 1:-1] = self._face_update(qx[:, 1:-1], eta[:, :-1], eta[:, 1:], z[:, :-1], z[:, 1:], self.n2x, dt)
        qy[1:-1, :] = self._face_update(qy[1:-1, :], eta[:-1, :], eta[1:, :], z[:-1, :], z[1:, :], self.n2y, dt)
        qo = self._outlet_flux(eta)
        e = self.outlet_edge
        if e == "E":
            qx[:, -1] = qo
        elif e == "W":
            qx[:, 0] = -qo
        elif e == "N":
            qy[-1, :] = qo
        else:
            qy[0, :] = -qo
        # negativity limiter: scale each cell's outgoing fluxes to its available water
        Fx = qx * self.wx
        Fy = qy * self.wy
        avail = h if src is None else h + dt * src
        out = (np.maximum(Fx[:, 1:], 0) + np.maximum(-Fx[:, :-1], 0)
               + np.maximum(Fy[1:, :], 0) + np.maximum(-Fy[:-1, :], 0)) * (dt / self.dx)
        over = out > avail
        if over.any():
            f = np.ones_like(h)
            f[over] = avail[over] / out[over]
            fp = np.pad(f, ((0, 0), (1, 1)), constant_values=1.0)
            sx = np.where(Fx > 0, fp[:, :-1], fp[:, 1:])
            qx *= sx
            Fx *= sx
            fp = np.pad(f, ((1, 1), (0, 0)), constant_values=1.0)
            sy = np.where(Fy > 0, fp[:-1, :], fp[1:, :])
            qy *= sy
            Fy *= sy
        div = (Fx[:, :-1] - Fx[:, 1:] + Fy[:-1, :] - Fy[1:, :]) * (dt / self.dx)
        if src is not None:
            div += dt * src
            self.vol_in += float(src.sum()) * dt * self.dx * self.dx
        h += div
        np.maximum(h, 0.0, out=h)
        # outflow volume
        if e == "E":
            o = Fx[:, -1].sum()
        elif e == "W":
            o = -Fx[:, 0].sum()
        elif e == "N":
            o = Fy[-1, :].sum()
        else:
            o = -Fy[0, :].sum()
        self.vol_out += float(o) * dt * self.dx

    def velocity(self, min_depth=0.05):
        h = self.h
        qxc = 0.5 * (self.qx[:, :-1] + self.qx[:, 1:])
        qyc = 0.5 * (self.qy[:-1, :] + self.qy[1:, :])
        ok = h >= min_depth
        hs = np.where(ok, h, 1.0)
        return np.where(ok, qxc / hs, 0.0), np.where(ok, qyc / hs, 0.0)

    @property
    def volume(self):
        return float(self.h.sum()) * self.dx * self.dx


# --------------------------------------------------------------------------
# Scenario-driven run
# --------------------------------------------------------------------------


@dataclass
class FloodGrid:
    """The 100 m flood-grid inputs derived from the terrain."""

    z: np.ndarray
    n: np.ndarray
    runoff: np.ndarray
    channel: np.ndarray
    levee: np.ndarray
    outlet: np.ndarray
    inflow_cells: np.ndarray  # (k, 2)
    cell: float


def coarse_grid(spec: ScenarioSpec, terr: Terrain, cfg: Config) -> FloodGrid:
    wc = cfg.world
    f = int(round(wc.flood_cell_m / terr.cell_size))
    z = block_reduce(terr.z.astype(np.float64), f, "mean")
    zmin = block_reduce(terr.z.astype(np.float64), f, "min")
    zmax = block_reduce(terr.z.astype(np.float64), f, "max")
    ch = terr.coarse_channel
    lev = terr.coarse_levee & ~ch
    z = np.where(ch, zmin, z)
    z = np.where(lev, zmax, z)
    n_cls = {WATER: spec.n_channel, OPEN: spec.n_floodplain, URBAN: spec.n_urban, FOREST: spec.n_forest}
    c_cls = {WATER: wc.runoff_coef["water"], OPEN: wc.runoff_coef["open"],
             URBAN: wc.runoff_coef["urban"], FOREST: wc.runoff_coef["forest"]}
    nf = np.zeros(terr.landcover.shape)
    cf = np.zeros(terr.landcover.shape)
    for k in n_cls:
        m = terr.landcover == k
        nf[m] = n_cls[k]
        cf[m] = c_cls[k]
    n = block_reduce(nf, f, "mean")
    n = np.where(ch, spec.n_channel, n)
    runoff = block_reduce(cf, f, "mean")
    nfc = z.shape[0]
    outlet = np.zeros_like(ch)
    if spec.outlet_edge == "E":
        outlet[:, -1] = True
    elif spec.outlet_edge == "W":
        outlet[:, 0] = True
    elif spec.outlet_edge == "N":
        outlet[-1, :] = True
    else:
        outlet[0, :] = True
    order = terr.coarse_channel_order
    k = min(cfg.world.inflow_spread_cells, len(order))
    return FloodGrid(z=z, n=n, runoff=runoff, channel=ch, levee=lev, outlet=outlet,
                     inflow_cells=order[:k], cell=wc.flood_cell_m)


@dataclass
class Storm:
    x_mid: float
    y_mid: float
    vx: float
    vy: float
    t_start: float  # seconds since onset
    duration: float
    radius: float
    peak_ms: float

    def centre(self, t):
        dtm = t - (self.t_start + 0.5 * self.duration)
        return self.x_mid + self.vx * dtm, self.y_mid + self.vy * dtm

    def envelope(self, t):
        u = (t - self.t_start) / self.duration
        return np.where((u >= 0) & (u <= 1), np.sin(np.pi * np.clip(u, 0, 1)) ** 2, 0.0)

    def rate(self, X, Y, t):
        """Rain rate in m/s on arrays X, Y at time t (seconds since onset)."""
        env = float(self.envelope(t))
        if env <= 0 or self.peak_ms <= 0:
            return np.zeros(np.shape(X))
        cx, cy = self.centre(t)
        r2 = (X - cx) ** 2 + (Y - cy) ** 2
        return self.peak_ms * env * np.exp(-r2 / (2 * self.radius**2))


def make_storm(spec: ScenarioSpec, terr: Terrain, cfg: Config, rng: np.random.Generator) -> Storm:
    speed = spec.wind10_ms * cfg.weather.storm_speed_factor
    th = spec.wind_to_rad + math.radians(spec.storm_track_offset_deg)
    town = terr.towns[int(rng.integers(len(terr.towns)))]
    xm = float(np.clip(town.x + rng.normal(0, 1000), 0, terr.domain_m))
    ym = float(np.clip(town.y + rng.normal(0, 1000), 0, terr.domain_m))
    peak = spec.rain_peak_mmh / 1000.0 / 3600.0 if spec.has_rain else 0.0
    return Storm(xm, ym, speed * math.cos(th), speed * math.sin(th), 0.0,
                 spec.storm_duration_h * 3600.0, spec.storm_radius_km * 1000.0, peak)


@dataclass
class FloodTimeline:
    """Ground-truth flood, episode time t in seconds since episode start.

    The solver state is stored on the 100 m flood grid. Depths on the 50 m grid
    are downscaled the usual way for flood maps: each wet 100 m cell's water
    surface minus the 50 m terrain, so land beside the river channel stays dry
    until the water surface reaches it.
    """

    depth_snap: np.ndarray  # float16 [T, nyc, nxc] coarse depth
    u_snap: np.ndarray  # float16 [Tv, nyc, nxc]
    v_snap: np.ndarray
    dt_depth: float
    dt_vel: float
    cell: float  # coarse cell (100 m)
    z_coarse: np.ndarray  # float32 [nyc, nxc]
    z_fine: np.ndarray  # float32 [ny, nx]
    fine_factor: int = 2
    wet_tol: float = 0.01
    mass_residual: float = 0.0
    breach_cell: tuple | None = None
    breach_t: float | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def t_end(self) -> float:
        return (self.depth_snap.shape[0] - 1) * self.dt_depth

    @property
    def fine_cell(self) -> float:
        return self.cell / self.fine_factor

    def _interp(self, snaps, dt, t):
        T = snaps.shape[0]
        if T == 1:
            return snaps[0].astype(np.float32)
        f = float(np.clip(t / dt, 0, T - 1))
        i = min(int(f), T - 2)
        w = f - i
        return snaps[i].astype(np.float32) * (1 - w) + snaps[i + 1].astype(np.float32) * w

    def coarse_depth_grid(self, t) -> np.ndarray:
        return self._interp(self.depth_snap, self.dt_depth, t)

    def _blocks(self):
        if "_zs" not in self._cache:
            f = self.fine_factor
            nyc, nxc = self.z_coarse.shape
            zb = self.z_fine.reshape(nyc, f, nxc, f).transpose(0, 2, 1, 3).reshape(nyc, nxc, f * f)
            zs = np.sort(zb.astype(np.float64), axis=-1)
            self._cache["_zs"] = (zs, np.cumsum(zs, axis=-1))
        return self._cache["_zs"]

    def _eta(self, h):
        """Water level per 100 m block: the lower of the solver stage and the
        level that holds the block's water volume over the 50 m terrain."""
        h = np.asarray(h, np.float64)
        zs, cs = self._blocks()
        k_n = zs.shape[-1]
        V = h * k_n
        eta_vol = np.full(h.shape, np.inf)
        found = np.zeros(h.shape, bool)
        for k in range(1, k_n + 1):
            e = (V + cs[..., k - 1]) / k
            ok = ~found & ((e <= zs[..., k]) if k < k_n else True)
            eta_vol = np.where(ok, e, eta_vol)
            found |= ok
        eta = np.minimum(self.z_coarse + h, eta_vol)
        return np.where(h > self.wet_tol, eta, -np.inf).astype(np.float32)

    def _eta_t(self, t):
        key = ("eta", float(t))
        if key not in self._cache:
            if len(self._cache) > 64:
                zs = self._cache.get("_zs")
                self._cache.clear()
                if zs is not None:
                    self._cache["_zs"] = zs
            self._cache[key] = self._eta(self.coarse_depth_grid(t))
        return self._cache[key]

    def downscale(self, h_coarse) -> np.ndarray:
        f = self.fine_factor
        eta = np.repeat(np.repeat(self._eta(h_coarse), f, -2), f, -1)
        return np.maximum(eta - self.z_fine, 0.0).astype(np.float32)

    def depth_grid(self, t) -> np.ndarray:
        """Depth on the 50 m grid at episode time t."""
        key = ("d", float(t))
        if key not in self._cache:
            f = self.fine_factor
            eta = np.repeat(np.repeat(self._eta_t(t), f, -2), f, -1)
            self._cache[key] = np.maximum(eta - self.z_fine, 0.0).astype(np.float32)
        return self._cache[key]

    def coarse_velocity_grid(self, t):
        return (self._interp(self.u_snap, self.dt_vel, t), self._interp(self.v_snap, self.dt_vel, t))

    def velocity_grid(self, t):
        """(u, v) on the 50 m grid (bilinear zoom of the 100 m field)."""
        u, v = self.coarse_velocity_grid(t)
        return self.upsample(u), self.upsample(v)

    def speed_grid(self, t):
        u, v = self.velocity_grid(t)
        return np.hypot(u, v)

    def hazard_grid(self, t):
        return hazard_rating(self.depth_grid(t), self.speed_grid(t)).astype(np.float32)

    def depth(self, x, y, t):
        nyc, nxc = self.z_coarse.shape
        cx = np.clip((np.asarray(x) / self.cell).astype(np.int64), 0, nxc - 1)
        cy = np.clip((np.asarray(y) / self.cell).astype(np.int64), 0, nyc - 1)
        fc = self.fine_cell
        n = self.z_fine.shape[0]
        fx = np.clip((np.asarray(x) / fc).astype(np.int64), 0, n - 1)
        fy = np.clip((np.asarray(y) / fc).astype(np.int64), 0, n - 1)
        eta = self._eta_t(t)[cy, cx]
        return np.maximum(eta - self.z_fine[fy, fx], 0.0)

    def velocity(self, x, y, t):
        u, v = self.coarse_velocity_grid(t)
        return bilinear(u, x, y, self.cell), bilinear(v, x, y, self.cell)

    def hazard(self, x, y, t):
        d = self.depth(x, y, t)
        u, v = self.velocity(x, y, t)
        return hazard_rating(d, np.hypot(u, v))

    def upsample(self, grid):
        return ndimage.zoom(np.asarray(grid, np.float32), self.fine_factor, order=1)

    def max_depth(self, t0=0.0, t1=None):
        """Maximum 50 m depth over episode times [t0, t1]."""
        t1 = self.t_end if t1 is None else t1
        i0 = max(int(np.floor(t0 / self.dt_depth)), 0)
        i1 = int(np.ceil(t1 / self.dt_depth)) + 1
        eta = np.full(self.z_coarse.shape, -np.inf, np.float32)
        for i in range(i0, min(i1, self.depth_snap.shape[0])):
            eta = np.maximum(eta, self._eta(self.depth_snap[i].astype(np.float32)))
        f = self.fine_factor
        eta = np.repeat(np.repeat(eta, f, 0), f, 1)
        return np.maximum(eta - self.z_fine, 0.0).astype(np.float32)


def run_flood(spec: ScenarioSpec, terr: Terrain, cfg: Config, rng: np.random.Generator,
              progress=False) -> tuple[FloodTimeline, Storm]:
    from .scenario import hydrograph, solver_window

    wc = cfg.world
    fg = coarse_grid(spec, terr, cfg)
    storm = make_storm(spec, terr, cfg, rng)
    solver = LocalInertialSolver(fg.z, fg.n, fg.cell, outlet=fg.outlet, outlet_edge=spec.outlet_edge,
                                 cfl=wc.cfl, dt_min=wc.dt_min_s, dt_max=wc.dt_max_s,
                                 wet_tol=wc.wet_tol_m, froude_limit=wc.froude_limit,
                                 outlet_slope=max(terr.river_slope, 1e-4))
    # the 100 m channel cells carry a channel_width_m wide river: faces between
    # two channel cells (and outlet faces of channel cells) get that width
    wch = min(spec.channel_width_m / fg.cell, 1.0)
    ch = fg.channel
    solver.wx[:, 1:-1][ch[:, :-1] & ch[:, 1:]] = wch
    solver.wy[1:-1, :][ch[:-1, :] & ch[1:, :]] = wch
    # initial channel depth: Manning normal depth for base flow
    q_base = spec.base_flow_frac * spec.peak_inflow_m3s
    S = max(terr.river_slope, 1e-4)
    h0 = (q_base * spec.n_channel / (spec.channel_width_m * math.sqrt(S))) ** 0.6
    solver.h[fg.channel] = h0
    vol0 = solver.volume

    t0, t_ep, t1 = solver_window(spec, cfg)
    t_breach = t_ep + spec.breach_time_h * 3600.0 if spec.has_breach else None
    breach_cell = None
    if spec.has_breach and fg.levee.any():
        ly, lx = np.nonzero(fg.levee)
        cx, cy = (lx + 0.5) * fg.cell, (ly + 0.5) * fg.cell
        d = np.min([np.hypot(cx - t.x, cy - t.y) for t in terr.towns], axis=0)
        cands = np.argsort(d)[:5]
        k = int(cands[int(rng.integers(len(cands)))])
        breach_cell = (int(ly[k]), int(lx[k]))

    X, Y = cell_centres(fg.z.shape[0], fg.cell)
    inflow = fg.inflow_cells
    k_in = len(inflow)
    n_depth = int(round((t1 - t_ep) / wc.depth_snapshot_s)) + 1
    n_vel = int(round((t1 - t_ep) / wc.velocity_snapshot_s)) + 1
    ny, nx = fg.z.shape
    depth = np.zeros((n_depth, ny, nx), np.float16)
    us = np.zeros((n_vel, ny, nx), np.float16)
    vs = np.zeros((n_vel, ny, nx), np.float16)
    snap_times = t_ep + np.arange(n_depth) * wc.depth_snapshot_s
    vel_every = int(round(wc.velocity_snapshot_s / wc.depth_snapshot_s))
    t = t0
    i_snap = 0
    breached = False
    src = np.zeros((ny, nx))
    rain_on = spec.has_rain and storm.peak_ms > 0
    while i_snap < n_depth:
        target = snap_times[i_snap]
        if t >= target - 1e-9:
            if t >= t_ep - 1e-9:
                depth[i_snap] = solver.h
                if i_snap % vel_every == 0:
                    u, v = solver.velocity(wc.vel_min_depth_m)
                    us[i_snap // vel_every] = u
                    vs[i_snap // vel_every] = v
            i_snap += 1
            continue
        if breach_cell is not None and not breached and t >= t_breach:
            by, bx = breach_cell
            solver.z[by, bx] -= spec.breach_drop_m
            wf = min(spec.breach_width_m / fg.cell, 1.0)
            solver.wx[by, bx] = solver.wx[by, bx + 1] = wf
            solver.wy[by, bx] = solver.wy[by + 1, bx] = wf
            breached = True
        dt = min(solver.stable_dt(), target - t)
        if breach_cell is not None and not breached:
            dt = min(dt, max(t_breach - t, 1e-3))
        dt = max(dt, 1e-3)
        src[:] = 0.0
        q = float(hydrograph(spec, t + 0.5 * dt))
        src[inflow[:, 0], inflow[:, 1]] = q / (k_in * fg.cell * fg.cell)
        if rain_on:
            src += storm.rate(X, Y, t + 0.5 * dt) * fg.runoff
        solver.step(dt, src)
        t += dt
    resid = (solver.volume - vol0) - (solver.vol_in - solver.vol_out)
    mass_residual = abs(resid) / max(solver.vol_in, 1.0)
    tl = FloodTimeline(depth_snap=depth, u_snap=us, v_snap=vs, dt_depth=wc.depth_snapshot_s,
                       dt_vel=wc.velocity_snapshot_s, cell=fg.cell,
                       z_coarse=solver.z.astype(np.float32), z_fine=terr.z.astype(np.float32),
                       fine_factor=int(round(fg.cell / terr.cell_size)), mass_residual=mass_residual,
                       breach_cell=breach_cell,
                       breach_t=(t_breach - t_ep) if breach_cell is not None else None)
    return tl, storm
