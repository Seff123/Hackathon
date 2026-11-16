"""Autopilot: flight modes + safety layer + swarm separation, vectorised over drones.

Goals from the controller set per-drone guidance arrays; each 1 s step turns
them into commanded airspeed, bank and throttle. The safety layer runs last
and always wins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import AutopilotParams, ControllerParams
from .glider import GliderState, Performance
from .goals import RTB, Goal, Loiter, Search, Transit, Verify

GROUND, LAUNCH, TRANSIT, SEARCH, THERMAL, VERIFY, RTB_M, LOITER, DOWN = range(9)
MODE_NAMES = ("GROUND", "LAUNCH", "TRANSIT", "SEARCH", "THERMAL", "VERIFY", "RTB", "LOITER", "DOWN")
WAYPOINT, LINE, ORBIT = 0, 1, 2
A_BAND, A_HOLD, A_DESCEND, A_CLIMB, A_GLIDE = range(5)
G = 9.81


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _clip(a, lo, hi):
    return np.minimum(np.maximum(a, lo), hi)


@dataclass
class APEvent:
    drone: int
    kind: str  # arrived, pattern_done, verify_done, thermal_exit, rtb, cannot_progress, launched
    t: float
    info: dict


class Autopilot:
    def __init__(self, n: int, perf: Performance, ap: AutopilotParams, cp: ControllerParams,
                 surface_at, ground_at, domain_m: float, base_xy, ceiling_agl: float):
        self.n = n
        self.perf = perf
        self.ap = ap
        self.cp = cp
        self.surface_at = surface_at
        self.ground_at = ground_at
        self.L = domain_m
        self.base = np.array(base_xy, float)
        self.base_z = float(ground_at(np.array([base_xy[0]]), np.array([base_xy[1]]))[0])
        self.ceiling = ceiling_agl
        self.reserve_wh = perf.gp.reserve_frac * perf.capacity_wh
        z = np.zeros(n)
        self.mode = np.full(n, GROUND, np.int64)
        self.goal: list[Goal | None] = [None] * n
        self.pending: list[Goal | None] = [None] * n
        self.gm = np.zeros(n, np.int64)
        self.tx, self.ty, self.lx, self.ly = z.copy(), z.copy(), z.copy(), z.copy()
        self.ox, self.oy, self.orad, self.odir = z.copy(), z.copy(), np.full(n, 100.0), np.ones(n)
        self.alt = np.zeros(n, np.int64)
        self.band_lo, self.band_hi = np.full(n, 150.0), np.full(n, 200.0)
        self.hold = np.full(n, 100.0)
        self.climbing = np.zeros(n, bool)
        self.spd = np.full(n, ap.transit_speed)
        self.fixed_spd = np.zeros(n, bool)
        self.bank_cap = np.full(n, math.radians(ap.bank_cmd_limit_deg))
        # search
        self.legs: list[np.ndarray | None] = [None] * n
        self.leg_idx = np.zeros(n, np.int64)
        self.search_phase = np.zeros(n, np.int64)  # 0 approach, 1 on legs
        # verify
        self.v_start = np.full(n, np.nan)
        # rtb
        self.rtb_phase = np.zeros(n, np.int64)
        self.launch_phase = np.zeros(n, np.int64)
        self.launch_t = np.zeros(n)
        # thermal
        self.netto_count = np.zeros(n)
        self.lift_x, self.lift_y = z.copy(), z.copy()
        self.th_t0 = z.copy()
        self.th_turn = z.copy()
        self.th_prev_psi = z.copy()
        self.th_zmark = z.copy()
        self.th_tmark = z.copy()
        self.th_circles = np.zeros(n, np.int64)
        self.resume: list[Goal | None] = [None] * n
        # separation
        self.avoid_until = np.full(n, -1.0)
        self.avoid_psi = z.copy()
        self.avoid_z = np.full(n, np.nan)
        # safety bookkeeping
        self.escape = np.zeros(n, bool)
        self.bz = np.zeros(n, bool)
        self.esc_clear = np.zeros(n)
        self.esc_dir = np.ones(n)
        self.stall_prot = np.zeros(n, bool)
        self.geo = np.zeros(n, bool)
        self.cant = np.zeros(n, bool)
        self.counts = {k: np.zeros(n, np.int64) for k in
                       ("terrain_warning", "stall_protect", "geofence_turn", "cannot_progress",
                        "separation_manoeuvre", "rtb_energy")}
        self.events: list[APEvent] = []
        self.t = 0.0
        self.layer = ap.transit_layer_base_m + ap.transit_layer_step_m * (np.arange(n) % ap.transit_layer_cycle)

    # ------------------------------------------------------------------ goals
    def default_band(self, i):
        lay = self.layer[i]
        h = self.ap.transit_band_half_m
        return (lay - h, lay + h)

    def set_goal(self, i: int, goal: Goal | None, t: float | None = None):
        if goal is None:
            return
        if self.mode[i] in (GROUND, DOWN) or self.mode[i] == LAUNCH:
            self.pending[i] = goal
            return
        if self.mode[i] == RTB_M and not isinstance(goal, RTB):
            return  # RTB is final until the drone lands
        if self.mode[i] == THERMAL and not isinstance(goal, (RTB, Verify)):
            self.resume[i] = goal  # finish the climb first
            return
        self._apply(i, goal)

    def _apply(self, i, goal: Goal):
        ap = self.ap
        self.goal[i] = goal
        self.bank_cap[i] = math.radians(ap.bank_cmd_limit_deg)
        self.fixed_spd[i] = False
        self.v_start[i] = np.nan
        if isinstance(goal, Transit):
            self.mode[i] = TRANSIT
            self._waypoint(i, goal.x, goal.y)
            lo, hi = goal.band or self.default_band(i)
            self._band(i, lo, hi)
        elif isinstance(goal, Search):
            self.mode[i] = SEARCH
            self.legs[i] = np.asarray(goal.legs, float)
            self.leg_idx[i] = min(goal.start_leg, len(self.legs[i]) - 1)
            self.search_phase[i] = 0
            a = self.legs[i][self.leg_idx[i]][0]
            self._waypoint(i, a[0], a[1])
            self.alt[i] = A_HOLD
            self.hold[i] = goal.agl
            self.spd[i] = goal.speed
            self.fixed_spd[i] = True
        elif isinstance(goal, Verify):
            self.mode[i] = VERIFY
            self._waypoint(i, goal.x, goal.y)
            self.alt[i] = A_HOLD
            self.hold[i] = ap.verify_agl_m
            self.spd[i] = max(self.cp.sweep_speed_min, 13.0)
            self.fixed_spd[i] = True
        elif isinstance(goal, RTB):
            self.mode[i] = RTB_M
            self.rtb_phase[i] = 0
            self._waypoint(i, *self.base)
            self._band(i, *self.default_band(i))
        elif isinstance(goal, Loiter):
            self.mode[i] = LOITER
            self._orbit(i, goal.x, goal.y, 150.0)
            self.alt[i] = A_HOLD
            self.hold[i] = goal.agl
            self.spd[i] = 13.0
            self.fixed_spd[i] = True

    def _waypoint(self, i, x, y):
        self.gm[i] = WAYPOINT
        self.tx[i], self.ty[i] = x, y

    def _line(self, i, a, b):
        self.gm[i] = LINE
        self.lx[i], self.ly[i] = a
        self.tx[i], self.ty[i] = b

    def _orbit(self, i, x, y, r, d=1.0):
        self.gm[i] = ORBIT
        self.ox[i], self.oy[i], self.orad[i], self.odir[i] = x, y, r, d

    def _band(self, i, lo, hi):
        self.alt[i] = A_BAND
        self.band_lo[i], self.band_hi[i] = lo, hi

    def _loiter_here(self, i, x, y):
        self._apply(i, Loiter(x=float(x), y=float(y), agl=max(self.hold[i], 100.0)))

    def _event(self, i, kind, t, **info):
        self.events.append(APEvent(int(i), kind, t, info))

    # ------------------------------------------------------------------ launch / landing
    def launch(self, i, s: GliderState, t: float):
        """Drone i leaves the pad: climb straight out towards its pending goal."""
        self.mode[i] = LAUNCH
        self.launch_phase[i] = 0
        self.launch_t[i] = t
        # spiral up over the pad first (bases can sit on slopes), then head out
        self._orbit(i, self.base[0], self.base[1], self.ap.launch_orbit_m)
        self.alt[i] = A_CLIMB
        self.bank_cap[i] = math.radians(self.ap.bank_cmd_limit_deg)
        self.netto_count[i] = 0
        self.avoid_until[i] = -1

    def _goal_point(self, g, s, i):
        if isinstance(g, (Transit, Verify, Loiter)):
            return g.x, g.y
        if isinstance(g, Search):
            return tuple(np.asarray(g.legs)[g.start_leg][0])
        return float(s.x[i] + 1000 * math.cos(s.psi[i])), float(s.y[i] + 1000 * math.sin(s.psi[i]))

    def landed(self, i):
        self.mode[i] = GROUND
        self.goal[i] = None
        self.resume[i] = None

    def down(self, i):
        self.mode[i] = DOWN
        self.goal[i] = None

    # ------------------------------------------------------------------ helpers
    def in_base_zone(self, s: GliderState):
        d = np.hypot(s.x - self.base[0], s.y - self.base[1])
        near = d < self.ap.base_zone_r_m
        traffic = ((self.mode == RTB_M) & (self.rtb_phase == 1)) | ((self.mode == LAUNCH) & (self.launch_phase == 0))
        return near & ((s.z - self.base_z < self.ap.base_zone_h_m) | traffic)

    def battery_frac(self, s):
        return s.E / self.perf.capacity_wh

    def _transit_speed(self, headwind, crosswind, netto=None):
        """15 m/s + half the headwind, at least |crosswind| + margin, capped at 25 m/s."""
        ap = self.ap
        v = np.maximum(ap.transit_speed + 0.5 * headwind, np.abs(crosswind) + ap.crosswind_margin)
        v = _clip(v, self.perf.v_min, ap.cruise_cap)
        if netto is None:
            return v
        return np.where(netto < ap.sink_netto, np.minimum(v + ap.sink_speed_add, ap.cruise_cap), v)

    def ground_speed(self, V, headwind, crosswind):
        crab = np.arcsin(_clip(crosswind / V, -1, 1))
        return V * np.cos(crab) - headwind

    # ------------------------------------------------------------------ main step
    def step(self, t: float, s: GliderState, wind_est, netto, ground=None):
        """wind_est: (ue, ve) mean-wind estimate at each drone. Returns V_cmd, phi_cmd, tau."""
        self.t = t
        ap, perf = self.ap, self.perf
        n = self.n
        act = s.active & (self.mode != GROUND) & (self.mode != DOWN)
        ue, ve = wind_est
        x, y, z = s.x, s.y, s.z
        if ground is None:
            ground = self.ground_at(x, y)
        self.bz = self.in_base_zone(s)
        # one batched surface lookup: here, then 2.5/5/7.5/10 s ahead along the
        # current turn (air velocity rotates at constant bank, wind adds linearly)
        ks = (2.5, 5.0, 7.5, ap.lookahead_s)
        K = np.array((0.0,) + ks)[:, None]
        Vh = np.maximum(s.V, 1.0)
        cps, sps = np.cos(s.psi), np.sin(s.psi)
        wx_, wy_ = s.vx - Vh * cps, s.vy - Vh * sps
        om = G * np.tan(s.phi) / Vh
        om = np.where(np.abs(om) < 1e-4, 1e-4, om)
        a = s.psi + om * K
        PX = (x + Vh / om * (np.sin(a) - sps) + wx_ * K).ravel()
        PY = (y + Vh / om * (cps - np.cos(a)) + wy_ * K).ravel()
        S_all = self.surface_at(PX, PY).reshape(len(ks) + 1, n)
        surf = S_all[0]
        h_surf = z - surf
        h_gnd = z - ground

        self._transitions(t, s, act, h_surf, h_gnd, netto, ue, ve)

        # ---- desired ground course
        chi = np.arctan2(self.ty - y, self.tx - x)
        is_line = self.gm == LINE
        if is_line.any():
            chi_l = np.arctan2(self.ty - self.ly, self.tx - self.lx)
            e = -(x - self.lx) * np.sin(chi_l) + (y - self.ly) * np.cos(chi_l)
            xmax = math.radians(ap.xtrack_max_deg)
            chi = np.where(is_line, chi_l - _clip(ap.xtrack_gain * e, -xmax, xmax), chi)
        is_orb = self.gm == ORBIT
        if is_orb.any():
            dxo, dyo = x - self.ox, y - self.oy
            do = np.hypot(dxo, dyo)
            th = np.arctan2(dyo, dxo)
            chi_orb = th + self.odir * (np.pi / 2 + np.arctan(1.5 * (do - self.orad) / self.orad))
            chi = np.where(is_orb, chi_orb, chi)
        else:
            do = self.orad * 10.0

        # geofence: near the edge and heading out -> turn towards the centre
        edge = np.minimum.reduce([x, y, self.L - x, self.L - y])
        cx, cy = self.L / 2 - x, self.L / 2 - y
        cn = np.maximum(np.hypot(cx, cy), 1.0)
        vn = np.maximum(np.hypot(s.vx, s.vy), 0.1)
        inward = (s.vx * cx + s.vy * cy) / (cn * vn)  # cos of angle between velocity and centre
        in_zone = edge < ap.geofence_margin_m
        start = act & in_zone & (inward < 0)
        # latch until heading clearly inward or out of the edge zone (prevents heading chatter)
        geo = act & (start | (self.geo & in_zone & (inward < 0.7)))
        chi = np.where(geo, np.arctan2(cy, cx), chi)
        self.counts["geofence_turn"] += geo & ~self.geo
        self.geo = geo

        # wind-corrected heading
        V_now = np.maximum(s.V, perf.v_min)
        wc = -ue * np.sin(chi) + ve * np.cos(chi)
        psi_d = chi - np.arcsin(_clip(wc / V_now, -1, 1))
        along = ue * np.cos(chi) + ve * np.sin(chi)
        gs = V_now * np.cos(psi_d - chi) + along
        cant = act & (gs < ap.cannot_progress_ms)
        for i in np.nonzero(cant & ~self.cant)[0]:
            self.counts["cannot_progress"][i] += 1
            self._event(i, "cannot_progress", t)
        self.cant = cant

        # ---- speed and throttle by altitude mode
        headwind = -along
        v_tr = self._transit_speed(headwind, wc, netto)
        V = np.where(self.fixed_spd, self.spd, v_tr)
        tau = np.zeros(n)
        # band
        band = self.alt == A_BAND
        below = h_surf < self.band_lo
        self.climbing = np.where(band & below, True, self.climbing)
        self.climbing = np.where(band & (h_surf >= self.band_lo + ap.band_hysteresis_m), False, self.climbing)
        above = band & (h_surf > self.band_hi) & ~self.climbing
        tau = np.where(band & self.climbing, 1.0, tau)
        v_climb = _clip(headwind + ap.climb_headwind_margin, perf.gp.climb_speed, perf.fast_climb_speed)
        V = np.where(band & self.climbing, v_climb, V)
        V = np.where(above, ap.band_fast_speed, V)
        # hold height above the surface (search, verify, loiter)
        hold = self.alt == A_HOLD
        ahead = S_all[2]
        z_t = np.maximum(surf, ahead) + self.hold
        high = hold & (z > z_t + 30.0)
        V = np.where(high, np.maximum(V, 20.0), V)
        # climb / descend / glide
        clim = self.alt == A_CLIMB
        tau = np.where(clim, 1.0, tau)
        # climb faster than the wind so the launch orbit holds position
        v_cl = _clip(np.hypot(ue, ve) + ap.crosswind_margin, perf.gp.climb_speed, perf.fast_climb_speed)
        V = np.where(clim, v_cl, V)
        desc = self.alt == A_DESCEND
        V = np.where(desc, ap.rtb_orbit_speed, V)

        # separation manoeuvres (beacon positions)
        self._separation(t, s, act)
        av = act & (self.avoid_until >= t)
        if av.any():
            chi = np.where(av, self.avoid_psi, chi)
            psi_d = np.where(av, self.avoid_psi, psi_d)
            up = av & np.isfinite(self.avoid_z) & (self.avoid_z > z + 5)
            dn = av & np.isfinite(self.avoid_z) & (self.avoid_z < z - 5)
            tau = np.where(up, 1.0, tau)
            V = np.where(up, perf.fast_climb_speed, V)
            tau = np.where(dn, 0.0, tau)
            V = np.where(dn, np.maximum(V, ap.band_fast_speed), V)

        # the geofence is part of the safety layer: it overrides separation headings too
        if (geo & av).any():
            ga = geo & av
            chi_c = np.arctan2(cy, cx)
            wcg = -ue * np.sin(chi_c) + ve * np.cos(chi_c)
            psi_d = np.where(ga, chi_c - np.arcsin(_clip(wcg / V_now, -1, 1)), psi_d)

        # ---- bank
        err = wrap(psi_d - s.psi)
        phi = np.arctan(V_now * ap.heading_gain * err / G)
        near_orb = (self.gm == ORBIT) & (np.abs(do - self.orad) < self.orad) & ~av
        phi_ff = self.odir * np.arctan(np.maximum(gs, 1.0) ** 2 / (G * self.orad))
        phi = np.where(near_orb, phi + phi_ff, phi)
        cap = self.bank_cap.copy()
        srch_leg = (self.mode == SEARCH) & (self.search_phase == 1) & (np.abs(err) < math.radians(30))
        cap = np.where(srch_leg, math.radians(ap.search_bank_deg), cap)
        phi = _clip(phi, -cap, cap)

        # throttle for hold (needs bank for the feed-forward)
        tff = perf.level_throttle(V, phi)
        tau = np.where(hold & ~av, _clip(tff + ap.alt_gain * (z_t - z), 0, 1), tau)

        # ---- safety layer (runs last, always wins)
        # 1. terrain
        exempt = ((self.mode == RTB_M) & (np.hypot(x - self.base[0], y - self.base[1]) < ap.rtb_floor_exempt_m)) \
            | ((self.mode == LAUNCH) & (self.launch_phase == 0))
        quiet = (self.mode == LAUNCH) & self.bz  # low near the pad is expected
        hmin = h_surf.copy()
        for kk, k in enumerate(ks):
            hmin = np.minimum(hmin, z + s.vz * k - S_all[kk + 1])
        trigger = act & ~exempt & (hmin < ap.floor_m)
        # latch: stay in escape until there is extra margin for escape_clear_s seconds
        self.esc_clear = np.where(hmin >= ap.floor_m + ap.escape_margin_m, self.esc_clear + 1, 0)
        danger = act & ~exempt & (trigger | (self.escape & (self.esc_clear < ap.escape_clear_s)))
        new = danger & ~self.escape
        if danger.any():
            spd = np.maximum(np.hypot(s.vx, s.vy), 5.0)
            hdg = np.arctan2(s.vy, s.vx)
            T10 = ap.lookahead_s
            if new.any():
                sl = self.surface_at(x + spd * T10 * np.cos(hdg + 0.8), y + spd * T10 * np.sin(hdg + 0.8))
                sr = self.surface_at(x + spd * T10 * np.cos(hdg - 0.8), y + spd * T10 * np.sin(hdg - 0.8))
                self.esc_dir = np.where(new, np.where(sl < sr, 1.0, -1.0), self.esc_dir)
            s0 = S_all[len(ks)]  # predicted point 10 s ahead
            turn = s0 > surf + 5.0  # ground ahead is rising: keep turning towards lower terrain
            bank = self.esc_dir * math.radians(45.0)
            phi = np.where(danger, np.where(turn, bank, 0.0), phi)
            tau = np.where(danger, 1.0, tau)
            V = np.where(danger, perf.fast_climb_speed, V)
        self.counts["terrain_warning"] += new & ~quiet
        self.escape = danger
        # 2. geofence handled in the course above
        # 3. stall protection
        vmin = ap.stall_margin * perf.stall_speed(phi)
        sp = act & (V < vmin)
        self.counts["stall_protect"] += sp & ~self.stall_prot
        self.stall_prot = sp
        V = np.maximum(V, vmin)
        # 4. ceiling
        ceil = h_gnd >= self.ceiling
        tau = np.where(ceil, 0.0, tau)
        for i in np.nonzero(act & ceil & (self.mode == THERMAL))[0]:
            self._exit_thermal(i, t, "ceiling")

        V = np.where(act, V, s.V)
        phi = np.where(act, phi, 0.0)
        tau = np.where(act, tau, 0.0)
        return V, phi, tau

    # ------------------------------------------------------------------ transitions
    def _transitions(self, t, s, act, h_surf, h_gnd, netto, ue, ve):
        ap, perf = self.ap, self.perf
        x, y, z = s.x, s.y, s.z
        mode = self.mode
        dwp = np.hypot(self.tx - x, self.ty - y)

        # launch: orbit climb over the pad, then straight out to the transit layer
        # each drone spirals up to its own transit layer (layers are 50 m apart by id)
        climb_to = np.maximum(self.ap.launch_orbit_climb_m, self.default_band(np.arange(self.n))[0])
        for i in np.nonzero(act & (mode == LAUNCH) & (self.launch_phase == 0) &
                            (z - self.base_z >= climb_to))[0]:
            # leave the orbit only when pointing at the goal, so drones peel off at
            # different points and times instead of crossing over the pad
            tx, ty = self._goal_point(self.pending[i], s, i)
            self.launch_phase[i] = 1
            self._waypoint(i, tx, ty)
        for i in np.nonzero(act & (mode == LAUNCH) & (self.launch_phase == 1) &
                            (h_surf >= self.default_band(np.arange(self.n))[0] - 10))[0]:
            g = self.pending[i]
            self.pending[i] = None
            self._apply(i, g if g is not None else Loiter(x=float(x[i]), y=float(y[i])))
            self._event(i, "launched", t)

        # transit arrival
        for i in np.nonzero(act & (mode == TRANSIT) & (dwp < ap.transit_arrive_m))[0]:
            self._event(i, "arrived", t, x=float(x[i]), y=float(y[i]))
            self._loiter_here(i, self.tx[i], self.ty[i])

        # search: approach -> legs -> next leg -> done
        for i in np.nonzero(act & (mode == SEARCH))[0]:
            legs = self.legs[i]
            k = self.leg_idx[i]
            if self.search_phase[i] == 0:
                if dwp[i] < 60.0:
                    self.search_phase[i] = 1
                    self._line(i, legs[k][0], legs[k][1])
                continue
            a, b = legs[k]
            d = b - a
            ln = float(np.hypot(*d))
            prog = ((x[i] - a[0]) * d[0] + (y[i] - a[1]) * d[1]) / max(ln, 1e-6)
            if prog >= ln - ap.search_wp_m:
                k += 1
                if k >= len(legs):
                    self._event(i, "pattern_done", t)
                    self._loiter_here(i, x[i], y[i])
                else:
                    self.leg_idx[i] = k
                    self._line(i, legs[k][0], legs[k][1])

        # verify: approach then a timed orbit
        for i in np.nonzero(act & (mode == VERIFY))[0]:
            g = self.goal[i]
            if np.isnan(self.v_start[i]):
                if dwp[i] < ap.verify_arrive_m:
                    self.v_start[i] = t
                    self._orbit(i, g.x, g.y, ap.verify_radius_m)
            elif t - self.v_start[i] >= ap.verify_s:
                self._event(i, "verify_done", t, cell=g.cell, x=g.x, y=g.y)
                self._loiter_here(i, g.x, g.y)

        # RTB: transit then a descending orbit over the base
        dbase = np.hypot(x - self.base[0], y - self.base[1])
        for i in np.nonzero(act & (mode == RTB_M) & (self.rtb_phase == 0) &
                            (dbase < ap.rtb_orbit_radius_m + 150.0))[0]:
            self.rtb_phase[i] = 1
            self._orbit(i, self.base[0], self.base[1], ap.rtb_orbit_radius_m)
            self.alt[i] = A_DESCEND
            self.bank_cap[i] = math.radians(ap.rtb_orbit_bank_deg)

        # energy rule
        cand = act & (mode != RTB_M) & (mode != LAUNCH)
        if cand.any() and (t % ap.rtb_check_s) < 1e-9 or (cand & (s.E <= ap.rtb_hard_floor * self.reserve_wh)).any():
            chi_h = np.arctan2(self.base[1] - y, self.base[0] - x)
            U_h = -(ue * np.cos(chi_h) + ve * np.sin(chi_h))
            W_c = -ue * np.sin(chi_h) + ve * np.cos(chi_h)
            Vc = self._transit_speed(U_h, W_c)
            gs = np.maximum(ap.min_ground_speed, self.ground_speed(Vc, U_h, W_c))
            e_land = np.maximum(z - self.base_z, 0.0) / ap.rtb_descent_rate * perf.payload / 3600.0
            e_ret = dbase / gs * perf.level_power(Vc) * ap.rtb_factor / 3600.0 + e_land + self.reserve_wh
            glide = (z - self.base_z) >= dbase * perf.sink(Vc, 0.0) / (ap.glide_margin * gs)
            trig = cand & (((s.E <= e_ret) & ~glide) | (s.E <= ap.rtb_hard_floor * self.reserve_wh))
            for i in np.nonzero(trig)[0]:
                if mode[i] == THERMAL:
                    self._exit_thermal(i, t, "rtb")
                self.counts["rtb_energy"][i] += 1
                self._apply(i, RTB())
                self._event(i, "rtb", t)

        # thermals: entry
        w_enter = self.cp.w_enter
        self.netto_count = np.where(act & (netto > w_enter), self.netto_count + 1, 0)
        first = act & (self.netto_count == 1)
        self.lift_x = np.where(first, x, self.lift_x)
        self.lift_y = np.where(first, y, self.lift_y)
        frac = self.battery_frac(s)
        can = act & ((mode == TRANSIT) | (mode == SEARCH) | (mode == LOITER)) & (self.netto_count >= ap.thermal_confirm_s) \
            & (frac < self.cp.e_soar) & (h_gnd < self.ceiling - ap.ceiling_margin_m)
        for i in np.nonzero(can)[0]:
            g = self.goal[i]
            if g is None or not g.allow_soaring:
                continue
            circling = (mode == THERMAL) & (np.arange(self.n) != i)
            if np.any(circling & (np.hypot(self.ox - self.lift_x[i], self.oy - self.lift_y[i]) < ap.thermal_exclusion_m)):
                continue
            self.enter_thermal(i, self.lift_x[i], self.lift_y[i], t, s)

        # thermals: exit
        for i in np.nonzero(act & (mode == THERMAL))[0]:
            dpsi = wrap(s.psi[i] - self.th_prev_psi[i])
            self.th_prev_psi[i] = s.psi[i]
            self.th_turn[i] += abs(dpsi)
            top = min(self.ceiling, self.zi_top(i)) if hasattr(self, "zi") else self.ceiling
            if h_gnd[i] >= top:
                self._exit_thermal(i, t, "top")
            elif t - self.th_t0[i] >= ap.thermal_max_s:
                self._exit_thermal(i, t, "timeout")
            elif self.th_turn[i] >= 2 * np.pi:
                rate = (z[i] - self.th_zmark[i]) / max(t - self.th_tmark[i], 1.0)
                self.th_turn[i] = 0.0
                self.th_zmark[i], self.th_tmark[i] = z[i], t
                self.th_circles[i] += 1
                if rate < ap.thermal_exit_climb:
                    self._exit_thermal(i, t, "weak")

    def zi_top(self, i):
        return self.zi - self.ap.thermal_zi_margin_m

    def enter_thermal(self, i, cx, cy, t, s: GliderState):
        ap, perf = self.ap, self.perf
        g = self.goal[i]
        if isinstance(g, Search):
            g = Search(legs=g.legs, agl=g.agl, speed=g.speed, start_leg=int(self.leg_idx[i]),
                       allow_soaring=g.allow_soaring)
        self.resume[i] = g
        self.mode[i] = THERMAL
        bank = math.radians(ap.thermal_bank_deg)
        v = ap.thermal_speed_factor * float(perf.stall_speed(bank))
        r = v * v / (G * math.tan(bank))
        self._orbit(i, cx, cy, r, 1.0)
        self.alt[i] = A_GLIDE
        self.spd[i] = v
        self.fixed_spd[i] = True
        self.bank_cap[i] = math.radians(ap.thermal_bank_deg + 3.0)
        self.th_t0[i] = t
        self.th_turn[i] = 0.0
        self.th_prev_psi[i] = s.psi[i]
        self.th_zmark[i], self.th_tmark[i] = s.z[i], t
        self.th_circles[i] = 0

    def _exit_thermal(self, i, t, why):
        if self.mode[i] != THERMAL:
            return
        g = self.resume[i]
        self.resume[i] = None
        self._event(i, "thermal_exit", t, why=why)
        if g is None:
            self._loiter_here(i, self.ox[i], self.oy[i])
        else:
            self._apply(i, g)
            if isinstance(g, Search) and self.search_phase[i] == 0:
                # rejoin the current leg directly
                k = self.leg_idx[i]
                self.search_phase[i] = 1
                self._line(i, g.legs[k][0], g.legs[k][1])

    # ------------------------------------------------------------------ separation
    def _separation(self, t, s: GliderState, act):
        ap = self.ap
        # avoid everyone except pad traffic (launching or in the landing orbit); the base-zone
        # exemption applies only to breach counting, so pairs never enter it already in conflict
        traffic = ((self.mode == LAUNCH) & (self.launch_phase == 0)) | ((self.mode == RTB_M) & (self.rtb_phase == 1))
        ok = act & ~(traffic & self.bz)
        idx = np.nonzero(ok)[0]
        if len(idx) < 2:
            return
        x, y, z = s.x[idx], s.y[idx], s.z[idx]
        px = x[None, :] - x[:, None]
        py = y[None, :] - y[:, None]
        pz = z[None, :] - z[:, None]
        dh = np.hypot(px, py)
        dz = np.abs(pz)
        conf = (dh < ap.sep_h_m) & (dz < ap.sep_v_m)
        # predictive: closest point of approach within sep_lookahead_s
        vx = s.vx[idx][None, :] - s.vx[idx][:, None]
        vy = s.vy[idx][None, :] - s.vy[idx][:, None]
        vz = s.vz[idx][None, :] - s.vz[idx][:, None]
        v2 = np.maximum(vx * vx + vy * vy, 1e-6)
        tc = _clip(-(px * vx + py * vy) / v2, 0.0, ap.sep_lookahead_s)
        dc = np.hypot(px + vx * tc, py + vy * tc)
        dzc = np.abs(pz + vz * tc)
        conf |= (dc < ap.sep_cpa_m) & (dzc < ap.sep_v_m) & (dh < ap.sep_detect_m)
        np.fill_diagonal(conf, False)
        if not conf.any():
            return
        turn = math.radians(ap.sep_turn_deg)
        # horizontal: steer away from each conflicting drone's closest-approach
        # position, biased 30 deg to the right so head-on pairs split right-right
        rx = np.zeros(len(idx))
        ry = np.zeros(len(idx))
        ex = px + vx * tc  # from a to b at CPA
        ey = py + vy * tc
        dist = np.maximum(np.hypot(ex, ey), 1.0)
        w = conf / dist
        rx -= (w * ex / dist).sum(axis=1)
        ry -= (w * ey / dist).sum(axis=1)
        for a in np.nonzero(conf.any(axis=1))[0]:
            k = int(idx[a])
            if self.avoid_until[k] < t:
                self.counts["separation_manoeuvre"][k] += 1
            right = s.psi[k] - turn
            n = math.hypot(rx[a], ry[a])
            bx, by = math.cos(right), math.sin(right)
            if n > 0:
                bx, by = bx + rx[a] / n * 1.5, by + ry[a] / n * 1.5
            self.avoid_psi[k] = math.atan2(by, bx)
            self.avoid_until[k] = t + ap.sep_hold_s
        # vertical: within each connected conflict cluster the lowest id keeps its height and
        # the others stack 50 m apart above the cluster's lowest drone, in id order (the plan's
        # "higher id moves 50 m away", generalised so three-way conflicts cannot share a height)
        m = len(idx)
        comp = -np.ones(m, np.int64)
        c_id = 0
        for a0 in range(m):
            if comp[a0] >= 0 or not conf[a0].any():
                continue
            stack = [a0]
            comp[a0] = c_id
            while stack:
                a = stack.pop()
                for b in np.nonzero(conf[a] | conf[:, a])[0]:
                    if comp[b] < 0:
                        comp[b] = c_id
                        stack.append(b)
            c_id += 1
        for c in range(c_id):
            mem = np.nonzero(comp == c)[0]  # ascending index = ascending drone id
            ids = idx[mem]
            z0 = float(s.z[ids].min())
            gnd = float(self.ground_at(s.x[ids], s.y[ids]).max())
            top = z0 + ap.sep_dz_m * (len(ids) - 1)
            shift = max(0.0, top - (gnd + self.ceiling - 20))
            for k, d in enumerate(ids[1:], start=1):
                self.avoid_z[d] = z0 + ap.sep_dz_m * k - shift
        done = self.avoid_until < t
        self.avoid_z = np.where(done, np.nan, self.avoid_z)
