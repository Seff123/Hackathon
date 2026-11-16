"""Autopilot: flight modes and the safety layer that runs last and always wins.

Per-drone logic in scalar Python (at most 8 drones); the physics step it feeds is vectorised.
The autopilot only uses agent-side information: own state, the drone's wind estimate and variometer,
static maps, teammates' traffic-beacon positions (separation and thermal occupancy only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import G, RHO, AutopilotParams, GliderParams
from .glider import GliderState, PerfTable
from .goals import RTB, Goal, Search, Transit, Verify

GROUND, TRANSIT, SEARCH, THERMAL, VERIFY, RTBM, DOWN = range(7)
MODE_NAMES = ["ground", "transit", "search", "thermal", "verify", "rtb", "down"]
TWO_PI = 2 * math.pi


def wrap(a: float) -> float:
    return (a + math.pi) % TWO_PI - math.pi


def lawnmower_box(box, leg_angle: float, spacing: float, start_xy=None) -> list:
    """Boustrophedon legs at `leg_angle` (rad) covering a rectangle (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    dx, dy = math.cos(leg_angle), math.sin(leg_angle)
    nx_, ny_ = -dy, dx
    corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    ns = [cx * nx_ + cy * ny_ for cx, cy in corners]
    nmin, nmax = min(ns), max(ns)
    spacing = max(spacing, 10.0)
    k = max(1, int(math.ceil((nmax - nmin) / spacing)))
    off = (nmax - nmin - (k - 1) * spacing) / 2
    legs = []
    for i in range(k):
        n = nmin + off + i * spacing
        px, py = n * nx_, n * ny_
        tlo, thi = -1e9, 1e9
        for p, dd, lo, hi in ((px, dx, x0, x1), (py, dy, y0, y1)):
            if abs(dd) < 1e-12:
                if p < lo or p > hi:
                    tlo, thi = 1, 0
                continue
            ta, tb = (lo - p) / dd, (hi - p) / dd
            tlo, thi = max(tlo, min(ta, tb)), min(thi, max(ta, tb))
        if thi <= tlo:
            continue
        a = (px + tlo * dx, py + tlo * dy)
        b = (px + thi * dx, py + thi * dy)
        legs.append((a, b) if i % 2 == 0 else (b, a))
    wps = [p for leg in legs for p in leg]
    if start_xy is not None and len(wps) > 1:
        wps = min((wps, wps[::-1]), key=lambda w: math.hypot(w[0][0] - start_xy[0], w[0][1] - start_xy[1]))
    return wps


@dataclass
class DroneAP:
    mode: int = GROUND
    goal: Goal | None = None
    launching: bool = False
    climbing: bool = False
    wps: list = field(default_factory=list)
    wpi: int = 0
    loiter: bool = False
    lx: float = 0.0
    ly: float = 0.0
    agl: float = 84.0
    speed: float = 13.0
    band: tuple = (120.0, 170.0)
    tx: float = 0.0
    ty: float = 0.0
    lift_count: float = 0.0
    lift_x: float = 0.0
    lift_y: float = 0.0
    prev_mode: int = TRANSIT
    th_cx: float = 0.0
    th_cy: float = 0.0
    th_t0: float = 0.0
    th_zc: float = 0.0
    th_tc: float = 0.0
    v_cx: float = 0.0
    v_cy: float = 0.0
    v_time: float = 0.0
    v_done: bool = False
    last_cp: float = -1e9
    terrain_warn: bool = False
    e_return: float = 0.0
    rtb_t: float = -1.0
    hold_until: float = -1.0
    geo_until: float = -1.0


class Autopilot:
    def __init__(self, n: int, gp: GliderParams, table: PerfTable, ap: AutopilotParams, surface: np.ndarray,
                 terrain_z: np.ndarray, cell: float, domain: float, base_xy, ceiling_agl: float, zi: float,
                 e_soar: float = 0.6, w_enter: float = 0.8):
        self.n = n
        self.gp, self.table, self.p = gp, table, ap
        self.surface = surface.astype(np.float64)
        self.terr = terrain_z.astype(np.float64)
        self.cell, self.domain = cell, domain
        self.nn = surface.shape[0]
        self._surf_l = self.surface.reshape(-1).tolist()
        self._terr_l = self.terr.reshape(-1).tolist()
        self.bx, self.by = base_xy
        self.bz = self.terrain_at(self.bx, self.by)
        self.ceiling = ceiling_agl
        self.zi = zi
        self.e_soar, self.w_enter = e_soar, w_enter
        self.d = [DroneAP() for _ in range(n)]
        self.cap = gp.capacity_wh
        self.reserve = gp.reserve_wh
        self.v_th = ap.thermal_speed_factor * float(gp.stall_speed(math.radians(ap.thermal_bank_deg)))
        self.counters = {"stall_protect": np.zeros(n, int), "terrain_warning": np.zeros(n, int),
                         "geofence": np.zeros(n, int), "cannot_progress": np.zeros(n, int)}
        self.cruise_power = None
        self._events = []

    # ------------------------------------------------------------------ map lookups (scalar)
    def _bil(self, g: list, x: float, y: float) -> float:
        """Bilinear lookup on a flat Python list (much faster than numpy scalar indexing)."""
        c, n = self.cell, self.nn
        fx = x / c - 0.5
        fy = y / c - 0.5
        if fx < 0.0:
            fx = 0.0
        elif fx > n - 1.000001:
            fx = n - 1.000001
        if fy < 0.0:
            fy = 0.0
        elif fy > n - 1.000001:
            fy = n - 1.000001
        ix, iy = int(fx), int(fy)
        ax, ay = fx - ix, fy - iy
        k = iy * n + ix
        return (g[k] * (1 - ax) + g[k + 1] * ax) * (1 - ay) + (g[k + n] * (1 - ax) + g[k + n + 1] * ax) * ay

    def surf_at(self, x, y):
        return self._bil(self._surf_l, float(x), float(y))

    def terrain_at(self, x, y):
        return self._bil(self._terr_l, float(x), float(y))

    def sink_scalar(self, V: float, bank: float = 0.0) -> float:
        return self.drag(V, bank) * V / self.gp.weight

    # ------------------------------------------------------------------ physics helpers
    def drag(self, V: float, bank: float = 0.0) -> float:
        gp = self.gp
        q = 0.5 * RHO * V * V * gp.wing_area
        cl = min(gp.weight / (q * math.cos(bank)), gp.cl_max)
        return q * (gp.cd0 + gp.k * cl * cl)

    def level_power(self, V: float) -> float:
        return self.drag(V) * V / self.gp.prop_eff + self.gp.payload_w

    def stall(self, bank: float) -> float:
        gp = self.gp
        return math.sqrt(2 * gp.weight / (RHO * gp.wing_area * gp.cl_max * max(math.cos(bank), 0.05)))

    def cruise_speed(self, headwind: float) -> float:
        p = self.p
        return min(p.transit_speed + 0.5 * max(headwind, 0.0), p.transit_speed_cap)

    def e_return_wh(self, x, y, wu, wv) -> tuple[float, float, float, float]:
        """Autopilot return-energy rule. Returns (E_return, V_c, ground speed, distance)."""
        p = self.p
        dx, dy = self.bx - x, self.by - y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return self.reserve, p.transit_speed, p.transit_speed, 0.0
        chi = math.atan2(dy, dx)
        head = -(wu * math.cos(chi) + wv * math.sin(chi))
        vc = self.cruise_speed(head)
        gs = max(p.rtb_min_gs, vc - head)
        e = dist / gs * self.level_power(vc) / 3600.0 * p.rtb_energy_factor + self.reserve
        return e, vc, gs, dist

    # ------------------------------------------------------------------ goal handling
    def layer_band(self, i: int) -> tuple:
        p = self.p
        layer = p.layer_base_agl + p.layer_step * (i % p.layer_cycle)
        return (layer - p.layer_half_band, layer + p.layer_half_band)

    def set_goal(self, i: int, goal: Goal, gs: GliderState, wind_dir: float, t: float) -> None:
        d = self.d[i]
        if d.mode in (GROUND, DOWN):
            d.goal = goal
            return
        if d.mode == RTBM and not isinstance(goal, RTB):
            # the energy rule fired: the drone is going home, goals wait until relaunch
            d.goal = goal
            return
        d.goal = goal
        d.loiter = False
        x, y = float(gs.x[i]), float(gs.y[i])
        if isinstance(goal, Transit):
            d.mode = TRANSIT
            d.tx, d.ty = goal.xy
            d.band = goal.alt_band or self.layer_band(i)
        elif isinstance(goal, Search):
            d.mode = SEARCH
            d.agl = goal.agl
            d.speed = goal.speed or 13.0
            if goal.waypoints is not None:
                d.wps = [tuple(p) for p in goal.waypoints]
            else:
                ang = wind_dir + math.radians(goal.leg_angle_deg if goal.leg_angle_deg is not None else 90.0)
                sp = goal.spacing or 2 * goal.agl * math.tan(math.radians(30)) * 0.85
                d.wps = lawnmower_box(goal.box, ang, sp, start_xy=(x, y))
            d.wpi = min(goal.start_index, max(len(d.wps) - 1, 0))
            if not d.wps:
                d.wps = [(x, y)]
        elif isinstance(goal, Verify):
            d.mode = VERIFY
            if goal.xy is not None:
                d.v_cx, d.v_cy = goal.xy
            else:
                iy, ix = divmod(int(goal.cell), self.nn)
                d.v_cx, d.v_cy = (ix + 0.5) * self.cell, (iy + 0.5) * self.cell
            d.v_time = 0.0
            d.v_done = False
        elif isinstance(goal, RTB):
            self._enter_rtb(i, t, "planner")

    def _enter_rtb(self, i: int, t: float, reason: str) -> None:
        d = self.d[i]
        if d.mode == THERMAL:
            d.prev_mode = TRANSIT
        d.mode = RTBM
        d.loiter = False
        d.rtb_t = t
        self._events.append((i, "rtb", {"reason": reason}))

    def launch(self, i: int, gs: GliderState, t: float, slot: int = 0) -> None:
        d = self.d[i]
        d.launching = True
        d.hold_until = t + slot * self.p.launch_stagger_s
        d.climbing = True
        d.mode = TRANSIT if not isinstance(d.goal, (Search, Verify)) else (SEARCH if isinstance(d.goal, Search) else VERIFY)
        gs.airborne[i] = True
        gs.landed[i] = False
        gs.z[i] = self.surf_at(gs.x[i], gs.y[i]) + 1.0
        gs.V[i] = self.gp.climb_speed
        gs.phi[i] = 0.0
        if d.goal is not None:
            tgt = self._goal_point(i, d.goal, gs)
            gs.psi[i] = math.atan2(tgt[1] - gs.y[i], tgt[0] - gs.x[i])

    def _goal_point(self, i, goal, gs):
        if isinstance(goal, Transit):
            return goal.xy
        if isinstance(goal, Verify):
            if goal.xy is not None:
                return goal.xy
            iy, ix = divmod(int(goal.cell), self.nn)
            return ((ix + 0.5) * self.cell, (iy + 0.5) * self.cell)
        if isinstance(goal, Search):
            if goal.waypoints:
                return goal.waypoints[min(goal.start_index, len(goal.waypoints) - 1)]
            if goal.box:
                return (0.5 * (goal.box[0] + goal.box[2]), 0.5 * (goal.box[1] + goal.box[3]))
        return (self.bx, self.by)

    # ------------------------------------------------------------------ main step
    def step(self, gs: GliderState, netto: np.ndarray, wind_est: np.ndarray, t: float,
             sep_dz: np.ndarray | None = None, sep_turn: np.ndarray | None = None,
             sep_cap: np.ndarray | None = None):
        """Returns (V_cmd, phi_cmd, tau, events). events: list of (drone, kind, data)."""
        n = self.n
        self._events = []
        V = gs.V.copy()
        PHI = np.zeros(n)
        TAU = np.zeros(n)
        thermal_pos = [(gs.x[j], gs.y[j]) for j in range(n) if self.d[j].mode == THERMAL]
        for i in range(n):
            d = self.d[i]
            if not gs.airborne[i] or d.mode in (GROUND, DOWN):
                continue
            v, phi, tau = self._step_one(i, gs, float(netto[i]), float(wind_est[i, 0]), float(wind_est[i, 1]), t,
                                         math.nan if sep_dz is None else float(sep_dz[i]),
                                         False if sep_turn is None else bool(sep_turn[i]), thermal_pos,
                                         math.nan if sep_cap is None else float(sep_cap[i]))
            V[i], PHI[i], TAU[i] = v, phi, tau
        return V, PHI, TAU, self._events

    def _step_one(self, i, gs, netto, wu, wv, t, sep_dz, sep_turn, thermal_pos, sep_cap=math.nan):
        p, gp = self.p, self.gp
        d = self.d[i]
        x, y, z = float(gs.x[i]), float(gs.y[i]), float(gs.z[i])
        psi, Vnow = float(gs.psi[i]), float(gs.V[i])
        E = float(gs.E[i])
        surf = self.surf_at(x, y)
        terr = self.terrain_at(x, y)
        hs, hg = z - surf, z - terr
        dbase = math.hypot(x - self.bx, y - self.by)

        # --- launch phase: hold over the base (staggered departures), then climb straight out
        if d.launching and t < d.hold_until:
            chi, ff = self._orbit(x, y, self.bx, self.by, p.launch_hold_r, Vnow)
            zt = self.bz + p.launch_hold_agl
            tau = min(max(self.drag(gp.climb_speed) / gp.thrust_max + 0.05 * (zt - z), 0.0), 1.0)
            psi_d = self._air_heading(chi, gp.climb_speed, wu, wv)
            phi = max(-math.radians(35), min(math.radians(35), ff + math.atan(Vnow * 0.5 * wrap(psi_d - psi) / G)))
            return gp.climb_speed, phi, tau
        if d.launching:
            lo = d.band[0] if d.mode == TRANSIT else 60.0
            if (hs >= min(lo, 100.0) and dbase > 200.0) or hs >= 150.0:
                d.launching = False
                d.climbing = False

        # --- return-to-base energy rule
        if d.mode in (TRANSIT, SEARCH, THERMAL, VERIFY) and not d.launching:
            if p.rtb_fixed_trigger is not None:
                trig = E <= p.rtb_fixed_trigger * self.cap
                d.e_return = p.rtb_fixed_trigger * self.cap
            else:
                er, vc, gsp, dist = self.e_return_wh(x, y, wu, wv)
                d.e_return = er
                glide_ok = (z - self.bz) >= dist * self.sink_scalar(vc, 0.0) / (p.glide_home_eff * gsp) and \
                    E > self.reserve + 0.02 * self.cap
                trig = E <= er and not glide_ok
            if trig:
                self._enter_rtb(i, t, "energy")
        elif d.mode == RTBM:
            d.e_return = self.e_return_wh(x, y, wu, wv)[0]

        # --- thermal entry
        if d.mode in (TRANSIT, SEARCH) and not d.launching:
            # the height limit is checked when lift is first felt (the climb in the lift must not cancel entry)
            soar_ok = (d.goal is not None and d.goal.allow_soaring and E / self.cap < self.e_soar and
                       (hg < self.ceiling - 100.0 or d.lift_count > 0))
            if soar_ok and netto > self.w_enter:
                if d.lift_count == 0:
                    d.lift_x, d.lift_y = x, y
                d.lift_count += 1.0
                if d.lift_count >= p.thermal_enter_s:
                    occupied = any(math.hypot(tx - d.lift_x, ty - d.lift_y) < p.thermal_occupied_m
                                   for tx, ty in thermal_pos)
                    if not occupied:
                        d.prev_mode = d.mode
                        d.mode = THERMAL
                        d.th_cx, d.th_cy = d.lift_x, d.lift_y
                        d.th_t0 = t
                        d.th_zc, d.th_tc = z, t
                        self._events.append((i, "thermal_enter", {}))
                    d.lift_count = 0.0
            else:
                d.lift_count = 0.0

        # --- mode behaviour -> desired track chi (or bank override), speed, throttle
        bank_ff = 0.0
        bank_lim = math.radians(p.bank_cmd_limit_deg)
        chi = None
        Vc = Vnow
        tau = 0.0
        if d.mode == TRANSIT:
            if d.loiter:
                chi, bank_ff = self._orbit(x, y, d.lx, d.ly, 150.0, Vnow)
            else:
                chi = math.atan2(d.ty - y, d.tx - x)
                if math.hypot(d.tx - x, d.ty - y) < p.arrive_radius:
                    d.loiter, d.lx, d.ly = True, d.tx, d.ty
                    self._events.append((i, "arrived", {}))
            head = -(wu * math.cos(chi) + wv * math.sin(chi))
            Vc = self.cruise_speed(head) + (p.sink_bonus_speed if netto < -1.0 else 0.0)
            if d.launching:
                tau, Vc = 1.0, gp.climb_speed
            else:
                Vc, tau = self._band(d, hs, Vc)
            if not d.loiter:
                gsp = self._ground_speed(chi, Vc, wu, wv)
                if gsp < p.cannot_progress_gs and t - d.last_cp > 60.0:
                    d.last_cp = t
                    self.counters["cannot_progress"][i] += 1
                    self._events.append((i, "cannot_progress", {}))
        elif d.mode == SEARCH:
            if d.loiter:
                chi, bank_ff = self._orbit(x, y, d.lx, d.ly, 120.0, Vnow)
            else:
                chi = self._carrot(d, x, y)
                if chi is None:
                    d.loiter = True
                    d.lx, d.ly = d.wps[-1]
                    self._events.append((i, "search_done", {}))
                    chi, bank_ff = self._orbit(x, y, d.lx, d.ly, 120.0, Vnow)
            Vc, tau = self._hold_agl(x, y, z, psi, Vnow, wu, wv, d.agl, d.speed)
            if d.launching:
                tau, Vc = 1.0, gp.climb_speed
        elif d.mode == THERMAL:
            chi, bank_ff = self._orbit(x, y, d.th_cx, d.th_cy, p.thermal_orbit_r, self.v_th)
            bank_lim = math.radians(p.thermal_bank_deg)
            Vc, tau = self.v_th, 0.0
            period = TWO_PI * p.thermal_orbit_r / self.v_th
            exit_reason = None
            if t - d.th_tc >= period:
                rate = (z - d.th_zc) / (t - d.th_tc)
                if rate < p.thermal_min_climb:
                    exit_reason = "weak"
                d.th_zc, d.th_tc = z, t
            if hg >= min(self.ceiling, self.zi - 50.0):
                exit_reason = "top"
            if t - d.th_t0 >= p.thermal_max_s:
                exit_reason = "timeout"
            if exit_reason:
                d.mode = d.prev_mode if d.prev_mode in (TRANSIT, SEARCH) else TRANSIT
                self._events.append((i, "thermal_exit", {"reason": exit_reason}))
        elif d.mode == VERIFY:
            R = p.verify_radius
            chi, bank_ff = self._orbit(x, y, d.v_cx, d.v_cy, R, Vnow)
            # orbits over nearby detections are stacked: one level up per earlier verifier within verify_stack_m
            level = sum(1 for j in range(i) if self.d[j].mode == VERIFY
                        and math.hypot(self.d[j].v_cx - d.v_cx, self.d[j].v_cy - d.v_cy) < p.verify_stack_m)
            v_agl = p.verify_agl + level * p.verify_stack_dz
            Vc, tau = self._hold_agl(x, y, z, psi, Vnow, wu, wv, v_agl, max(d.speed, 13.0))
            dc = math.hypot(x - d.v_cx, y - d.v_cy)
            if abs(dc - R) < 40.0 and abs(hs - v_agl) < 40.0:
                d.v_time += 1.0
            if d.v_time >= p.verify_s and not d.v_done:
                d.v_done = True
                self._events.append((i, "verify_done", {"xy": (d.v_cx, d.v_cy)}))
        elif d.mode == RTBM:
            if dbase > 300.0:
                chi = math.atan2(self.by - y, self.bx - x)
                head = -(wu * math.cos(chi) + wv * math.sin(chi))
                Vc = self.cruise_speed(head)
                er, vc, gsp, dist = self.e_return_wh(x, y, wu, wv)
                z_need = self.bz + p.rtb_approach_agl + dist * self.sink_scalar(vc, 0.0) / gsp * 1.25
                z_floor = max(self._surf_ahead(x, y, psi, Vnow, wu, wv) + p.rtb_approach_agl, self.bz + p.rtb_approach_agl)
                if z >= z_need and z >= z_floor + 20.0:
                    tau = 0.0
                elif z_floor - z > 30.0:
                    tau, Vc = 1.0, gp.climb_speed
                else:
                    zt = max(z_floor, min(z, z_need))
                    tau = min(max(self.drag(Vc) / gp.thrust_max + p.search_tau_gain * (zt - z), 0.0), 1.0)
            else:
                R = p.rtb_orbit_r
                chi, bank_ff = self._orbit(x, y, self.bx, self.by, R, Vnow)
                tau = 0.0
                Vc = 20.0 if hg > 40.0 else 13.0
                if hg > 40.0 and dbase > 150.0:
                    Vc = 15.0

        # --- separation: altitude floor 50 m above the conflicting drone below (never pushed down)
        if not math.isnan(sep_dz) and z < sep_dz + 5.0:
            if sep_dz - z > 5.0:
                tau, Vc = 1.0, gp.climb_speed
            else:
                tau = max(tau, min(max(self.drag(Vc) / gp.thrust_max + 0.05 * (sep_dz + 5.0 - z), 0.0), 1.0))
        elif not math.isnan(sep_cap) and z > sep_cap - 5.0:
            tau = 0.0       # lower drone of a closing conflict: motor off, glide while the upper one climbs away
        psi_d = None
        if chi is not None:
            psi_d = self._air_heading(chi, Vc, wu, wv)
        if sep_turn:
            psi_d = psi - math.radians(p.sep_turn_deg)
            bank_ff = 0.0
            bank_lim = max(bank_lim, math.radians(p.bank_cmd_limit_deg))

        # --- geofence: within the margin, heading outward and predicted to leave within the look-ahead
        #     (latched for a few seconds) -> turn back towards the centre
        m = p.geofence_margin
        L = self.domain
        gvx, gvy = Vnow * math.cos(psi) + wu, Vnow * math.sin(psi) + wv
        tl = p.lookahead_s
        near = x < m or x > L - m or y < m or y > L - m
        px, py = x + gvx * tl, y + gvy * tl
        leaving = near and (px < 0 or px > L or py < 0 or py > L)
        if leaving and t >= d.geo_until:
            d.geo_until = t + 15.0
            self.counters["geofence"][i] += 1
        if t < d.geo_until:
            psi_d = self._air_heading(math.atan2(L / 2 - y, L / 2 - x), Vc, wu, wv)
            bank_ff = 0.0

        # --- heading -> bank
        if psi_d is None:
            phi = 0.0
        else:
            err = wrap(psi_d - psi)
            phi = bank_ff + math.atan(Vnow * 0.5 * err / G)
            phi = max(-bank_lim, min(bank_lim, phi))

        # --- terrain safety
        exempt = d.launching or (d.mode == RTBM and dbase < p.rtb_exempt_radius)
        if not exempt:
            zdot = float(gs.zdot[i]) if gs.zdot is not None else 0.0
            worst = 1e9
            for tt in (2.5, 5.0, 7.5, p.lookahead_s):
                px, py = x + gvx * tt, y + gvy * tt
                worst = min(worst, z + zdot * tt - self.surf_at(px, py))
            if worst < p.safety_floor or hs < p.safety_floor:
                if not d.terrain_warn:
                    self.counters["terrain_warning"][i] += 1
                    self._events.append((i, "terrain_warning", {"mode": MODE_NAMES[d.mode], "hs": round(hs, 1),
                                                                 "worst": round(worst, 1), "zdot": round(zdot, 2),
                                                                 "launching": d.launching}))
                d.terrain_warn = True
                tau, Vc = 1.0, self.table.v_fast_climb
                look = p.lookahead_s * Vnow
                left = self.surf_at(x + look * math.cos(psi + 0.8), y + look * math.sin(psi + 0.8))
                right = self.surf_at(x + look * math.cos(psi - 0.8), y + look * math.sin(psi - 0.8))
                lim = math.radians(45.0)
                phi = lim if left < right else -lim
            else:
                d.terrain_warn = False

        # --- ceiling
        if hg > self.ceiling and (math.isnan(sep_dz) or z > sep_dz + 5.0):
            tau = 0.0
            if d.mode == THERMAL:
                d.mode = d.prev_mode if d.prev_mode in (TRANSIT, SEARCH) else TRANSIT
                self._events.append((i, "thermal_exit", {"reason": "ceiling"}))

        # --- stall protection
        vmin = p.stall_margin * self.stall(phi)
        if Vc < vmin:
            Vc = vmin
            self.counters["stall_protect"][i] += 1
        Vc = min(max(Vc, gp.v_min), gp.v_max)
        return Vc, phi, tau

    # ------------------------------------------------------------------ mode helpers
    def _air_heading(self, chi, V, wu, wv):
        wc = -wu * math.sin(chi) + wv * math.cos(chi)
        return chi - math.asin(max(-1.0, min(1.0, wc / max(V, 1e-3))))

    def _ground_speed(self, chi, V, wu, wv):
        psi_d = self._air_heading(chi, V, wu, wv)
        return V * math.cos(psi_d - chi) + (wu * math.cos(chi) + wv * math.sin(chi))

    def _orbit(self, x, y, cx, cy, R, V):
        dx, dy = x - cx, y - cy
        dist = math.hypot(dx, dy)
        beta = math.atan2(dy, dx)
        chi = beta + math.pi / 2 + math.atan(2.0 * (dist - R) / R)
        ff = math.atan(V * V / (G * R)) if abs(dist - R) < R else 0.0
        return chi, ff

    def _carrot(self, d: DroneAP, x, y):
        p = self.p
        while d.wpi < len(d.wps):
            bx, by = d.wps[d.wpi]
            if d.wpi == 0:
                ax, ay = x, y
            else:
                ax, ay = d.wps[d.wpi - 1]
            sx, sy = bx - ax, by - ay
            L2 = sx * sx + sy * sy
            if math.hypot(bx - x, by - y) < p.waypoint_tol or L2 < 1.0:
                d.wpi += 1
                continue
            s = ((x - ax) * sx + (y - ay) * sy) / L2
            if s >= 1.0:
                d.wpi += 1
                continue
            Lm = math.sqrt(L2)
            s2 = min(max(s, 0.0) + p.carrot_m / Lm, 1.0)
            cxp, cyp = ax + s2 * sx, ay + s2 * sy
            return math.atan2(cyp - y, cxp - x)
        return None

    def _surf_ahead(self, x, y, psi, V, wu, wv):
        gvx, gvy = V * math.cos(psi) + wu, V * math.sin(psi) + wv
        return max(self.surf_at(x, y), self.surf_at(x + 5 * gvx, y + 5 * gvy), self.surf_at(x + 10 * gvx, y + 10 * gvy))

    def _hold_agl(self, x, y, z, psi, V, wu, wv, agl, speed):
        gp, p = self.gp, self.p
        zt = self._surf_ahead(x, y, psi, V, wu, wv) + agl
        if z - zt > 20.0:
            return p.search_descent_speed, 0.0
        if zt - z > 30.0:
            return gp.climb_speed, 1.0
        tau = self.drag(speed) / gp.thrust_max + p.search_tau_gain * (zt - z)
        return speed, min(max(tau, 0.0), 1.0)

    def _band(self, d: DroneAP, hs, Vc):
        lo, hi = d.band
        if d.climbing:
            if hs >= lo + self.p.band_hysteresis:
                d.climbing = False
            else:
                return self.gp.climb_speed, 1.0
        if hs < lo:
            d.climbing = True
            return self.gp.climb_speed, 1.0
        if hs > hi:
            return self.p.fast_descent_speed, 0.0
        return Vc, 0.0

    # ------------------------------------------------------------------ status
    def mode_of(self, i: int) -> int:
        return self.d[i].mode

    def modes(self) -> np.ndarray:
        return np.array([d.mode for d in self.d])
