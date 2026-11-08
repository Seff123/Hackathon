"""Terrain: spectral noise, carved meandering valley and river, land cover,
towns, roads, levees. Hydrology layers are attached from hydrology.py."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .config import ScenarioSpec, WorldConfig
from .grid import cell_centres, rasterise_polyline, smoothstep, upsample_repeat, bilinear
from .hydrology import compute_hydrology, priority_flood

WATER, OPEN, FOREST, URBAN = 0, 1, 2, 3
LANDCOVER_NAMES = {WATER: "water", OPEN: "open", FOREST: "forest", URBAN: "urban"}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


@dataclass
class Town:
    x: float
    y: float
    radius: float


@dataclass
class Terrain:
    cell_size: float
    n: int
    z: np.ndarray
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
    river_cells: np.ndarray
    levee_mask: np.ndarray
    building_density: np.ndarray
    road_mask: np.ndarray
    towns: list
    population: np.ndarray
    base_xy: tuple
    depression_depth: np.ndarray
    terminal: np.ndarray
    outlet_edge: str
    river_line: np.ndarray  # dense centreline polyline (N, 2), inflow -> outlet
    river_order: np.ndarray  # fine river raster cells (K, 2) [iy, ix] in flow order
    coarse_channel: np.ndarray  # flood-grid channel mask
    coarse_channel_order: np.ndarray  # flood-grid channel cells in flow order
    coarse_levee: np.ndarray
    bridge_mask: np.ndarray
    raw_obstacle_h: np.ndarray = field(repr=False, default=None)
    river_slope: float = 0.001
    river_length_m: float = 0.0

    @property
    def land(self) -> np.ndarray:
        return self.landcover != WATER

    @property
    def domain_m(self) -> float:
        return self.n * self.cell_size

    def set_base(self, xy, clear_radius_m: float):
        """Place the base and clear obstacles (a landing area) around it."""
        self.base_xy = (float(xy[0]), float(xy[1]))
        X, Y = cell_centres(self.n, self.cell_size)
        near = np.hypot(X - xy[0], Y - xy[1]) <= clear_radius_m
        self.obstacle_h = np.where(near, 0.0, self.raw_obstacle_h).astype(np.float32)
        self.surface = (self.z + self.obstacle_h).astype(np.float32)

    def surface_at(self, x, y):
        return bilinear(self.surface, x, y, self.cell_size)

    def ground_at(self, x, y):
        return bilinear(self.z, x, y, self.cell_size)


# --------------------------------------------------------------------------


def spectral_noise(rng: np.random.Generator, n: int, beta: float) -> np.ndarray:
    """Fractal noise with power spectrum ~ f^-beta, rescaled to [0, 1]."""
    w = rng.standard_normal((n, n))
    F = np.fft.fft2(w)
    fx = np.fft.fftfreq(n)
    f = np.hypot(fx[None, :], fx[:, None])
    f[0, 0] = np.inf
    F *= f ** (-beta / 2.0)
    out = np.real(np.fft.ifft2(F))
    out -= out.min()
    return out / max(out.max(), 1e-12)


def edge_distance(X, Y, L, edge):
    return {"N": L - Y, "S": Y, "E": L - X, "W": X}[edge]


def edge_point(L, edge, u):
    eps = 1e-3
    return {"N": (u, L - eps), "S": (u, eps), "E": (L - eps, u), "W": (eps, u)}[edge]


def _river_centreline(rng, spec: ScenarioSpec, wc: WorldConfig):
    L = wc.domain_m
    out_e = spec.outlet_edge
    in_e = OPPOSITE[out_e]
    j = wc.river_start_jitter
    u0 = L * (0.5 + j * rng.uniform(-1, 1))
    u1 = L * (0.5 + j * rng.uniform(-1, 1))
    p0 = np.array(edge_point(L, in_e, u0))
    p1 = np.array(edge_point(L, out_e, u1))
    axis = p1 - p0
    length = np.linalg.norm(axis)
    a = axis / length
    perp = np.array([-a[1], a[0]])
    t = np.linspace(0.0, 1.0, 2001)
    nsin = int(rng.integers(2, 5))
    ks = rng.choice(np.arange(1, 8), size=nsin, replace=False)
    amps = rng.uniform(0.3, 1.0, nsin) / ks
    phases = rng.uniform(0, 2 * np.pi, nsin)
    o = np.zeros_like(t)
    for k, am, ph in zip(ks, amps, phases):
        o += am * np.sin(k * np.pi * t + ph)
    o -= (1 - t) * o[0] + t * o[-1]
    o /= max(np.abs(o).max(), 1e-9)
    m = wc.river_margin_m
    flow_is_y = out_e in ("N", "S")

    def build(A):
        pts = p0[None, :] + t[:, None] * axis[None, :] + (A * o)[:, None] * perp[None, :]
        if flow_is_y:
            pts[:, 0] = np.clip(pts[:, 0], m, L - m)
            pts[:, 1] = np.clip(pts[:, 1], 1e-3, L - 1e-3)
        else:
            pts[:, 1] = np.clip(pts[:, 1], m, L - m)
            pts[:, 0] = np.clip(pts[:, 0], 1e-3, L - 1e-3)
        return pts

    def sinuosity(A):
        pts = build(A)
        return np.hypot(*np.diff(pts, axis=0).T).sum() / length

    lo, hi = 0.0, L
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        if sinuosity(mid) < spec.sinuosity:
            lo = mid
        else:
            hi = mid
    pts = build(0.5 * (lo + hi))
    # resample densely by arc length (10 m)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    ss = np.arange(0, s[-1], 10.0)
    ss = np.append(ss, s[-1])
    return np.stack([np.interp(ss, s, pts[:, 0]), np.interp(ss, s, pts[:, 1])], 1)


def _line_cells(p, q, n, cell, rng, wiggle=0.03):
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    length = np.linalg.norm(d)
    if length < 1:
        return np.zeros((0, 2), np.int64)
    perp = np.array([-d[1], d[0]]) / length
    t = np.linspace(0, 1, 200)
    k = int(rng.integers(1, 4))
    ph = rng.uniform(0, 2 * np.pi)
    off = wiggle * length * np.sin(k * np.pi * t + ph)
    off -= (1 - t) * off[0] + t * off[-1]
    pts = p[None] + t[:, None] * d[None] + off[:, None] * perp[None]
    L = n * cell
    pts = np.clip(pts, 1.0, L - 1.0)
    return rasterise_polyline(pts[:, 0], pts[:, 1], cell, n)


def build_terrain(spec: ScenarioSpec, wc: WorldConfig, rng: np.random.Generator) -> Terrain:
    n, c, L = wc.n, wc.cell_m, wc.domain_m
    X, Y = cell_centres(n, c)

    # 1. base surface: spectral noise + rims on non-outlet edges + regional slope
    # beta is the 1D profile spectral slope; in 2D the amplitude goes as f^-(beta+1)/2
    z0 = spectral_noise(rng, n, spec.spectral_beta + 1.0) * spec.relief_m
    rim = np.zeros((n, n))
    for e in ("N", "E", "S", "W"):
        if e != spec.outlet_edge:
            rim = np.maximum(rim, smoothstep(1 - edge_distance(X, Y, L, e) / wc.edge_rim_width_m))
    z0 += wc.edge_rim_frac * spec.relief_m * rim
    z0 += spec.regional_slope * edge_distance(X, Y, L, spec.outlet_edge)
    z0 += 20.0  # keep elevations positive

    # 2. river centreline
    line = _river_centreline(rng, spec, wc)
    river_order = rasterise_polyline(line[:, 0], line[:, 1], c, n)
    river_line_mask = np.zeros((n, n), bool)
    river_line_mask[river_order[:, 0], river_order[:, 1]] = True

    # along-river coordinate for each raster cell, then for every cell (nearest)
    steps = np.hypot(*np.diff(river_order.astype(float), axis=0).T) * c
    s_cells = np.concatenate([[0.0], np.cumsum(steps)])
    s_total = s_cells[-1]
    s_grid_line = np.zeros((n, n))
    s_grid_line[river_order[:, 0], river_order[:, 1]] = s_cells
    dist, (iy_n, ix_n) = ndimage.distance_transform_edt(~river_line_mask, return_indices=True)
    dist = dist * c
    s_near = s_grid_line[iy_n, ix_n]

    # 3. valley profile: centreline elevation falls linearly, always below the land
    zc = z0[river_order[:, 0], river_order[:, 1]] - spec.valley_depth_m
    remain = np.maximum(s_total - s_cells, 1e-6)
    gmin = wc.min_river_slope
    e_out = float(np.min(zc - gmin * remain))
    gr = (zc[:-1] - e_out) / remain[:-1]
    g = max(float(np.min(gr)) if len(gr) else gmin, gmin)
    E_line = e_out + g * (s_total - s_cells)
    E_grid = e_out + g * (s_total - s_near)

    half = spec.valley_width_m / 2.0
    w_fp = wc.floodplain_frac * half
    fp_amp = rng.uniform(*wc.floodplain_noise_m)
    fp_noise = spectral_noise(rng, n, 3.0) * fp_amp
    z1 = z0 - spec.valley_depth_m * smoothstep(1 - dist / half)
    beta = 1.0 - smoothstep((dist - w_fp) / max(half - w_fp, 1.0))
    z = (1 - beta) * z1 + beta * (E_grid + fp_noise)

    # channel: centreline + width, lowered by channel depth with small noise
    ch_noise = ndimage.gaussian_filter1d(rng.standard_normal(len(river_order)), 5) * 3.0
    ch_noise = np.clip(ch_noise, -1, 1) * wc.centreline_noise_m
    channel = river_line_mask | (dist <= spec.channel_width_m / 2.0)
    bed = E_grid.copy()
    bed[river_order[:, 0], river_order[:, 1]] = E_line + ch_noise
    bed_near = bed[iy_n, ix_n]
    z = np.where(channel, bed_near - spec.channel_depth_m, z)

    # flood-grid channel (4-connected raster at the coarse resolution)
    f = int(round(wc.flood_cell_m / c))
    nfc = n // f
    coarse_order = rasterise_polyline(line[:, 0], line[:, 1], wc.flood_cell_m, nfc)
    coarse_channel = np.zeros((nfc, nfc), bool)
    coarse_channel[coarse_order[:, 0], coarse_order[:, 1]] = True

    # 9. towns (before land cover, needed for levees and roads)
    slope_tmp = np.hypot(*np.gradient(z, c))
    edge_d = np.minimum.reduce([X, Y, L - X, L - Y])
    wt = np.exp(-(((dist - half) / max(half, 300.0)) ** 2))
    cand = (dist >= 0.3 * half + 50) & (dist <= wc.town_river_max_m) & (slope_tmp < 0.06) & (edge_d > 1000) & ~channel
    towns: list[Town] = []
    for r in spec.town_radii_m:
        ok = cand.copy()
        for t in towns:
            ok &= np.hypot(X - t.x, Y - t.y) >= wc.town_min_sep_m
        if not ok.any():
            ok = cand.copy()
        if not ok.any():
            ok = ~channel & (edge_d > 500)
        w = (wt * ok).ravel()
        if w.sum() <= 0:
            w = ok.ravel().astype(float)
        k = int(rng.choice(n * n, p=w / w.sum()))
        towns.append(Town(float(X.flat[k]), float(Y.flat[k]), float(r)))
    town_noise = spectral_noise(rng, n, 2.5) * 2 - 1
    urban = np.zeros((n, n), bool)
    density = np.zeros((n, n))
    town_edge_d = np.full((n, n), np.inf)
    for t in towns:
        d = np.hypot(X - t.x, Y - t.y)
        urban |= d < t.radius * (1 + 0.25 * town_noise)
        density = np.maximum(density, np.exp(-0.5 * (d / (0.6 * t.radius)) ** 2))
        town_edge_d = np.minimum(town_edge_d, np.maximum(d - t.radius, 0.0))
    urban &= ~channel
    density = np.where(urban, density, 0.0)

    # 4. levees (built on the flood grid, mirrored on the fine grid)
    nb = ndimage.binary_dilation(coarse_channel, structure=np.ones((3, 3), bool)) & ~coarse_channel
    town_edge_c = town_edge_d.reshape(nfc, f, nfc, f).min(axis=(1, 3))
    coarse_levee = nb & (town_edge_c <= wc.levee_town_dist_m) if spec.levee_height_m > 0.05 else np.zeros_like(nb)
    levee_mask = upsample_repeat(coarse_levee, f) & ~channel
    z = np.where(levee_mask, z + spec.levee_height_m, z)

    # 10. roads: town-town links and radial roads out of each town
    road = np.zeros((n, n), bool)
    for i, t in enumerate(towns):
        for t2 in towns[i + 1:]:
            cells = _line_cells((t.x, t.y), (t2.x, t2.y), n, c, rng)
            road[cells[:, 0], cells[:, 1]] = True
        k = int(np.argmin(np.hypot(line[:, 0] - t.x, line[:, 1] - t.y)))
        away = math.atan2(t.y - line[k, 1], t.x - line[k, 0])
        for j in range(wc.roads_per_town):
            ang = away + rng.uniform(-0.6, 0.6) if j == 0 else rng.uniform(0, 2 * np.pi)
            ln = rng.uniform(*wc.road_len_m)
            q = (t.x + ln * math.cos(ang), t.y + ln * math.sin(ang))
            cells = _line_cells((t.x, t.y), q, n, c, rng)
            road[cells[:, 0], cells[:, 1]] = True
    bridge = road & channel

    # 8. land cover
    landcover = np.full((n, n), OPEN, np.int8)
    landcover[urban] = URBAN
    slope = np.hypot(*np.gradient(z, c))
    elig = ~channel & ~urban & ~road & (town_edge_d > 200)
    fnoise = spectral_noise(rng, n, 2.2)
    score = fnoise + 0.6 * slope / max(np.percentile(slope, 99), 1e-6)
    n_land = int((~channel).sum())
    n_forest = min(int(spec.forest_fraction * n_land), int(elig.sum()))
    forest = np.zeros((n, n), bool)
    if n_forest > 0:
        idx = np.argsort(np.where(elig, score, -np.inf).ravel())[::-1][:n_forest]
        forest.flat[idx] = True
    landcover[forest] = FOREST
    landcover[channel] = WATER

    # obstacle heights
    fh = wc.forest_h_m[0] + (wc.forest_h_m[1] - wc.forest_h_m[0]) * spectral_noise(rng, n, 1.5)
    uh = wc.urban_h_m[0] + (wc.urban_h_m[1] - wc.urban_h_m[0]) * np.clip(
        density + 0.2 * town_noise, 0, 1)
    obstacle = np.where(forest, fh, 0.0)
    obstacle = np.where(urban, uh, obstacle)
    obstacle = np.where(road | channel, 0.0, obstacle).astype(np.float32)

    # 5-7. hydrology. Real landscapes have few deep closed basins; spectral noise
    # has many, so partly fill depressions deeper than max_pond_depth_m.
    zfill0 = priority_flood(z, eps=wc.prefill_eps_m)
    z = np.maximum(z, zfill0 - wc.max_pond_depth_m)
    zf32 = z.astype(np.float32)
    hyd = compute_hydrology(zf32, c, wc.drainage_area_km2, force_drainage=channel)
    gy, gx = np.gradient(zf32.astype(np.float64), c)
    dist_to_river = (ndimage.distance_transform_edt(~channel) * c).astype(np.float32)

    # 11. population map (planner-visible)
    pop = 0.8 * density / max(density.max(), 1e-9) + 0.2 * road
    pop = (pop / max(pop.sum(), 1e-12)).astype(np.float32)

    terr = Terrain(
        cell_size=c, n=n, z=zf32, z_fill=hyd.z_fill.astype(np.float32),
        surface=(zf32 + obstacle).astype(np.float32),
        slope=np.hypot(gx, gy).astype(np.float32), grad_x=gx.astype(np.float32),
        grad_y=gy.astype(np.float32), landcover=landcover, obstacle_h=obstacle,
        flow_dir=hyd.flow_dir, flow_acc=hyd.flow_acc, drainage_mask=hyd.drainage_mask,
        drain_id=hyd.drain_id, hand=hyd.hand, dist_to_river=dist_to_river,
        river_cells=channel, levee_mask=levee_mask, building_density=density.astype(np.float32),
        road_mask=road, towns=towns, population=pop, base_xy=(0.0, 0.0),
        depression_depth=hyd.depression_depth, terminal=hyd.terminal,
        outlet_edge=spec.outlet_edge, river_line=line, river_order=river_order,
        coarse_channel=coarse_channel, coarse_channel_order=coarse_order,
        coarse_levee=coarse_levee, bridge_mask=bridge, raw_obstacle_h=obstacle.copy(),
        river_slope=float(g), river_length_m=float(s_total),
    )
    base = pick_base(terr, wc, excluded=None)
    terr.set_base(base, wc.base_clear_radius_m)
    return terr


def pick_base(terr: Terrain, wc: WorldConfig, excluded: np.ndarray | None):
    """Highest road cell within base_town_dist of the largest town's edge."""
    n, c = terr.n, terr.cell_size
    X, Y = cell_centres(n, c)
    big = max(terr.towns, key=lambda t: t.radius)
    d_edge = np.maximum(np.hypot(X - big.x, Y - big.y) - big.radius, 0.0)
    edge_d = np.minimum.reduce([X, Y, terr.domain_m - X, terr.domain_m - Y])
    ok = terr.road_mask & ~terr.river_cells & (d_edge <= wc.base_town_dist_m) & (edge_d > 600)
    if excluded is not None:
        ok &= ~excluded
    if not ok.any():  # fall back: any land cell near the town
        ok = ~terr.river_cells & (d_edge <= wc.base_town_dist_m) & (edge_d > 600)
        if excluded is not None:
            ok &= ~excluded
    if not ok.any():
        ok = ~terr.river_cells & (edge_d > 600)
    k = int(np.argmax(np.where(ok, terr.z, -np.inf)))
    return float(X.flat[k]), float(Y.flat[k])
