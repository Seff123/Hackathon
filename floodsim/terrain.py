"""Terrain: spectral noise, carved meandering valley and river, towns, roads, levees, land cover."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from . import hydrology
from .config import ScenarioSpec, WorldConfig

OPEN, WATER, URBAN, FOREST = 0, 1, 2, 3
LC_NAMES = {OPEN: "open", WATER: "water", URBAN: "urban", FOREST: "forest"}
TOWN_NAMES = ["Lowmead", "Hillcrest", "Brookby", "Fenwold", "Ashcombe", "Marlow End", "Wetherfield", "Kingsmere"]


@dataclass
class Terrain:
    z: np.ndarray
    z_bare: np.ndarray            # before levees
    z_fill: np.ndarray
    surface: np.ndarray
    slope: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray
    landcover: np.ndarray
    obstacle_h: np.ndarray
    flow_dir: np.ndarray
    flow_acc: np.ndarray
    drainage_mask: np.ndarray
    drain_id: np.ndarray
    hand: np.ndarray
    dist_to_river: np.ndarray
    river_cells: np.ndarray       # flat indices ordered from inflow to outlet
    river_mask: np.ndarray
    levee_mask: np.ndarray
    building_density: np.ndarray
    road_mask: np.ndarray
    bridge_mask: np.ndarray
    depression_depth: np.ndarray
    towns: list
    population: np.ndarray
    base_xy: tuple
    cell_size: float
    river_xy: np.ndarray          # dense centreline points [N, 2]
    river_s: np.ndarray           # arc length along centreline
    river_bed: np.ndarray         # bed elevation along centreline
    outlet_edge: str
    drain_outlet_frac: float
    roads: list = field(default_factory=list)   # polylines [[x, y], ...] for plotting
    levee_lines: list = field(default_factory=list)
    river_drain_id: np.ndarray | None = None   # first main-river cell downstream (-1: none)
    hand_river: np.ndarray | None = None       # height above that river cell

    @property
    def n(self) -> int:
        return self.z.shape[0]

    def outlet_edge_mask(self) -> np.ndarray:
        return edge_mask(self.n, self.outlet_edge)


def edge_mask(n: int, edge: str) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    if edge == "S":
        m[0, :] = True
    elif edge == "N":
        m[-1, :] = True
    elif edge == "W":
        m[:, 0] = True
    else:
        m[:, -1] = True
    return m


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def spectral_noise(rng, n: int, beta: float) -> np.ndarray:
    """White noise -> FFT -> amplitude * f^(-beta/2) -> inverse FFT, rescaled to [0, 1]."""
    white = rng.standard_normal((n, n))
    F = np.fft.fft2(white)
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n)))
    k[0, 0] = 1.0
    F *= k ** (-beta / 2.0)
    F[0, 0] = 0.0
    z = np.real(np.fft.ifft2(F))
    z -= z.min()
    return z / max(z.max(), 1e-12)


def smooth_noise(rng, n: int, sigma_cells: float) -> np.ndarray:
    z = gaussian_filter(rng.standard_normal((n, n)), sigma_cells, mode="wrap")
    z -= z.min()
    return z / max(z.max(), 1e-12)


def _uv_to_xy(u, v, edge: str, L: float):
    if edge == "S":
        return v, L - u
    if edge == "N":
        return v, u
    if edge == "W":
        return L - u, v
    return u, v


def _xy_to_u(x, y, edge: str, L: float):
    """Distance from the inflow edge along the flow axis."""
    if edge == "S":
        return L - y
    if edge == "N":
        return y
    if edge == "W":
        return L - x
    return x


def river_centreline(spec: ScenarioSpec, rng, L: float, step: float = 5.0):
    """Straight line + 2-4 sinusoids, amplitude scaled by bisection to match the sinuosity."""
    u = np.arange(0.0, L + step / 2, step)
    v0 = L / 2 + spec.river_start_frac * L
    v1 = L / 2 + spec.river_end_frac * L
    base = v0 + (v1 - v0) * u / L
    k = int(rng.integers(2, 5))
    freqs = rng.uniform(1.0, 4.5, k)
    phases = rng.uniform(0, 2 * np.pi, k)
    amps = rng.uniform(0.5, 1.0, k) / freqs
    shape = (amps[:, None] * np.sin(2 * np.pi * freqs[:, None] * u[None, :] / L + phases[:, None])).sum(0)
    shape /= max(np.abs(shape).max(), 1e-9)
    margin = 400.0

    def path(A):
        v = np.clip(base + A * shape, margin, L - margin)
        return v

    def sinu(A):
        v = path(A)
        arc = np.hypot(np.diff(u), np.diff(v)).sum()
        return arc / np.hypot(L, v[-1] - v[0])

    lo, hi = 0.0, 500.0
    while sinu(hi) < spec.sinuosity and hi < 20000:
        hi *= 2
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        if sinu(mid) < spec.sinuosity:
            lo = mid
        else:
            hi = mid
    v = path(0.5 * (lo + hi))
    # light smoothing of the polyline (clipping can leave kinks)
    v = gaussian_filter(v, 6.0, mode="nearest")
    x, y = _uv_to_xy(u, v, spec.outlet_edge, L)
    x = np.clip(x, 1.0, L - 1.0)
    y = np.clip(y, 1.0, L - 1.0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    return np.stack([x, y], 1), s, u


def rasterize4(xy: np.ndarray, cell: float, n: int) -> np.ndarray:
    """Ordered, 4-connected list of (iy, ix) cells along a dense polyline. Returns [M, 3] with point index."""
    ix = np.clip((xy[:, 0] // cell).astype(np.int64), 0, n - 1)
    iy = np.clip((xy[:, 1] // cell).astype(np.int64), 0, n - 1)
    out = [(iy[0], ix[0], 0)]
    for p in range(1, len(xy)):
        py, px, _ = out[-1]
        cy, cx = iy[p], ix[p]
        if (cy, cx) == (py, px):
            continue
        if cy != py and cx != px:  # diagonal step: insert a 4-connecting cell
            out.append((py, cx, p))
        out.append((cy, cx, p))
    # remove revisits (keep first occurrence) to obtain a simple ordered path
    seen = set()
    res = []
    for c in out:
        if (c[0], c[1]) not in seen:
            seen.add((c[0], c[1]))
            res.append(c)
    return np.array(res, dtype=np.int64)


def polyline(a, b, rng, wiggle: float, step: float = 10.0) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = np.linalg.norm(b - a)
    m = max(int(d / step), 2)
    t = np.linspace(0, 1, m)
    pts = a[None] + t[:, None] * (b - a)[None]
    nrm = np.array([-(b - a)[1], (b - a)[0]]) / max(d, 1e-9)
    f = rng.uniform(1.0, 3.0)
    ph = rng.uniform(0, 2 * np.pi)
    off = wiggle * np.sin(np.pi * t) * np.sin(2 * np.pi * f * t + ph)
    return pts + off[:, None] * nrm[None]


def generate_terrain(spec: ScenarioSpec, wc: WorldConfig, rng: np.random.Generator) -> Terrain:
    n, c, L = wc.n, wc.cell_m, wc.domain_m
    xs = (np.arange(n) + 0.5) * c
    X, Y = np.meshgrid(xs, xs)
    U = _xy_to_u(X, Y, spec.outlet_edge, L)

    # --- river centreline and raster
    rxy, rs, ru = river_centreline(spec, rng, L)
    cells = rasterize4(rxy, c, n)
    river_mask = np.zeros((n, n), dtype=bool)
    river_mask[cells[:, 0], cells[:, 1]] = True
    river_cells = cells[:, 0] * n + cells[:, 1]

    plane = spec.regional_slope * (L - U)
    bed_noise = gaussian_filter(rng.standard_normal(len(rs)), 40.0) * 0.3
    bed_noise = np.clip(bed_noise / max(np.abs(bed_noise).max(), 1e-9) * 0.3, -0.3, 0.3)
    river_bed = 20.0 + spec.regional_slope * (L - ru) - spec.channel_depth_m + bed_noise

    d, (niy, nix) = distance_transform_edt(~river_mask, sampling=c, return_indices=True)
    hw = spec.valley_width_m / 2.0
    fp = 0.75 * hw
    valley = spec.valley_depth_m * smoothstep((d - fp) / max(hw - fp, 1.0))
    ramp = 0.3 * spec.relief_m * smoothstep((d - hw) / 5000.0)
    noise = 0.7 * spec.relief_m * spectral_noise(rng, n, spec.spectral_beta)
    damp = smoothstep((d - fp) / max(2 * hw, 1500.0))
    fpn = smooth_noise(rng, n, 4.0) - 0.5
    fpn = fpn / max(np.abs(fpn).max(), 1e-9) * spec.floodplain_noise_m
    fp_w = 1.0 - smoothstep(d / max(hw, 1.0))
    lateral = (0.25 * spec.regional_slope + 0.0005) * d   # floodplain slopes gently towards the river
    z = 20.0 + plane + valley + ramp + noise * damp + 0.5 * fpn * fp_w + lateral
    # river cells: bed elevation at the path point that created the cell
    z[cells[:, 0], cells[:, 1]] = river_bed[cells[:, 2]]
    z = z.astype(np.float32)

    gy, gx = np.gradient(z.astype(np.float64), c)
    slope = np.hypot(gx, gy)

    # --- towns: on low-slope floodplain edges within 1.5 km of the river
    towns = []
    cand = (~river_mask) & (d >= 0.4 * hw) & (d <= wc.town_max_river_dist_m)
    edge_ok = (X > 800) & (X < L - 800) & (Y > 800) & (Y < L - 800)
    cand &= edge_ok
    w = np.where(cand, np.exp(-slope / 0.03) * np.exp(-np.abs(d - hw) / 600.0), 0.0)
    for t in range(spec.n_towns):
        ww = w.copy()
        for tw in towns:
            ww[np.hypot(X - tw["x"], Y - tw["y"]) < wc.town_min_separation_m] = 0.0
        if ww.sum() <= 0:
            ww = w.copy()
            for tw in towns:
                ww[np.hypot(X - tw["x"], Y - tw["y"]) < 0.5 * wc.town_min_separation_m] = 0.0
        if ww.sum() <= 0:
            break
        idx = rng.choice(ww.size, p=(ww / ww.sum()).reshape(-1))
        iy, ix = divmod(int(idx), n)
        towns.append({"name": TOWN_NAMES[t % len(TOWN_NAMES)], "x": float(xs[ix]), "y": float(xs[iy]),
                      "r": float(spec.town_radius_m[t])})

    # --- urban mask and building density
    edge_noise = smooth_noise(rng, n, 3.0) - 0.5
    urban = np.zeros((n, n), dtype=bool)
    bd = np.zeros((n, n))
    for tw in towns:
        dt_ = np.hypot(X - tw["x"], Y - tw["y"])
        urban |= (dt_ / tw["r"] + 0.5 * edge_noise) < 1.0
        bd = np.maximum(bd, np.exp(-dt_ ** 2 / (2 * (0.6 * tw["r"]) ** 2)))
    urban &= ~river_mask
    bd = np.where(urban, bd, 0.0)

    # --- base site and roads
    largest = max(towns, key=lambda t: t["r"])
    dlt = np.hypot(X - largest["x"], Y - largest["y"]) - largest["r"]
    base_cand = (dlt > 0) & (dlt <= wc.base_town_dist_m) & (~river_mask) & (~urban) & \
                (X > 300) & (X < L - 300) & (Y > 300) & (Y < L - 300) & (d > hw)
    if not base_cand.any():
        base_cand = (dlt > 0) & (dlt <= wc.base_town_dist_m * 1.5) & (~river_mask) & (~urban)
    zc = np.where(base_cand, z, -np.inf)
    bidx = int(np.argmax(zc))
    site = (float(xs[bidx % n]), float(xs[bidx // n]))
    road = np.zeros((n, n), dtype=bool)
    roads = []
    for tw in towns:
        pl = polyline((tw["x"], tw["y"]), site, rng, wiggle=120.0)
        roads.append(pl.tolist())
        rc = rasterize4(pl, c, n)
        road[rc[:, 0], rc[:, 1]] = True
    bridge = road & river_mask
    # base: highest road cell within 2 km of the largest town's edge
    road_c = road & (dlt <= wc.base_town_dist_m) & (~river_mask) & (~urban)
    if not road_c.any():
        road_c = road & (~river_mask)
    zr = np.where(road_c, z, -np.inf)
    bidx = int(np.argmax(zr))
    base_xy = (float(xs[bidx % n]), float(xs[bidx // n]))

    # --- levees on both banks where the river passes within 1 km of a town
    z_bare = z.copy()
    levee = np.zeros((n, n), dtype=bool)
    levee_lines = []
    if spec.levee_height_m > 0.05:
        nb = np.zeros((n, n), dtype=bool)
        nb[1:, :] |= river_mask[:-1, :]
        nb[:-1, :] |= river_mask[1:, :]
        nb[:, 1:] |= river_mask[:, :-1]
        nb[:, :-1] |= river_mask[:, 1:]
        nb &= ~river_mask
        near_town = np.zeros((n, n), dtype=bool)
        for tw in towns:
            near_town |= np.hypot(X - tw["x"], Y - tw["y"]) < tw["r"] + wc.levee_town_dist_m
        levee = nb & near_town & ~bridge
        z = np.where(levee, z + spec.levee_height_m, z).astype(np.float32)
        for tw in towns:
            sel = np.hypot(rxy[:, 0] - tw["x"], rxy[:, 1] - tw["y"]) < tw["r"] + wc.levee_town_dist_m
            if sel.any():
                levee_lines.append(rxy[sel][::20].tolist())

    # --- forest: noise threshold favouring steep slopes, away from towns and roads
    land = ~river_mask
    away = distance_transform_edt(~urban, sampling=c) > 200.0 if urban.any() else np.ones_like(urban)
    dbase = np.hypot(X - base_xy[0], Y - base_xy[1])
    score = smooth_noise(rng, n, 5.0) + 0.6 * slope / max(slope.max(), 1e-9)
    elig = land & ~urban & ~road & away & (dbase > 300.0)
    forest = np.zeros((n, n), dtype=bool)
    n_target = int(spec.forest_fraction * land.sum())
    if elig.sum() > 0 and n_target > 0:
        thr = np.sort(score[elig])[::-1][min(n_target, elig.sum()) - 1]
        forest = elig & (score >= thr)

    landcover = np.full((n, n), OPEN, dtype=np.int8)
    landcover[forest] = FOREST
    landcover[urban] = URBAN
    landcover[river_mask] = WATER

    tree_h = wc.forest_h[0] + (wc.forest_h[1] - wc.forest_h[0]) * smooth_noise(rng, n, 2.0)
    obstacle = np.zeros((n, n), dtype=np.float32)
    obstacle[forest] = tree_h[forest]
    obstacle[urban] = (wc.urban_h[0] + (wc.urban_h[1] - wc.urban_h[0]) * bd[urban]).astype(np.float32)
    obstacle[dbase < wc.base_clear_radius_m] = 0.0
    surface = (z + obstacle).astype(np.float32)

    gy, gx = np.gradient(z.astype(np.float64), c)
    slope = np.hypot(gx, gy)

    # --- hydrology
    z_fill = hydrology.priority_flood(z)
    down = hydrology.d8_directions(z_fill)
    acc, order = hydrology.flow_accumulation(z_fill, down)
    drainage = (acc * c * c >= wc.drainage_area_km2 * 1e6) | river_mask
    drain_id, hand = hydrology.drainage_and_hand(z_fill, down, order, drainage)
    outlet = hydrology.outlet_of(down, order)
    oe = edge_mask(n, spec.outlet_edge).reshape(-1)
    drains_out = oe[outlet] & land.reshape(-1)
    drain_frac = float(drains_out.sum() / max(land.sum(), 1))
    dist_to_river = distance_transform_edt(~river_mask, sampling=c).astype(np.float32)
    river_drain_id, hand_river = hydrology.drainage_and_hand(z_fill, down, order, river_mask)

    pop = 0.8 * bd + 0.2 * road.astype(float)
    pop = pop / max(pop.sum(), 1e-12)

    return Terrain(
        z=z, z_bare=z_bare, z_fill=z_fill, surface=surface, slope=slope.astype(np.float32),
        grad_x=gx.astype(np.float32), grad_y=gy.astype(np.float32), landcover=landcover,
        obstacle_h=obstacle, flow_dir=down.reshape(n, n), flow_acc=acc.astype(np.float32),
        drainage_mask=drainage, drain_id=drain_id, hand=hand, dist_to_river=dist_to_river,
        river_cells=river_cells, river_mask=river_mask, levee_mask=levee, building_density=bd.astype(np.float32),
        road_mask=road, bridge_mask=bridge, depression_depth=(z_fill - z).astype(np.float32), towns=towns,
        population=pop.astype(np.float32), base_xy=base_xy, cell_size=c, river_xy=rxy, river_s=rs,
        river_bed=river_bed, outlet_edge=spec.outlet_edge, drain_outlet_frac=drain_frac, roads=roads,
        levee_lines=levee_lines, river_drain_id=river_drain_id, hand_river=hand_river,
    )
