"""Flight core: glider physics + autopilot + wind + termination, for the whole swarm.

Used by Sim (with people, sensors, radio and a controller on top) and directly
by the autopilot tests on synthetic terrain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .autopilot import (DOWN, GROUND, LAUNCH, MODE_NAMES, RTB_M, THERMAL, Autopilot)
from .config import AutopilotParams, ControllerParams, GliderParams
from .glider import GliderState, Performance, glider_step


@dataclass
class SafetyLog:
    crashes: int = 0
    collisions: int = 0
    forced_landings: int = 0
    geofence_violations: int = 0
    separation_breaches: int = 0
    stalls: int = 0
    events: list = field(default_factory=list)  # (t, kind, drone(s), info)


class FlightCore:
    def __init__(self, n: int, gp: GliderParams, ap: AutopilotParams, cp: ControllerParams,
                 surface_at, ground_at, domain_m: float, base_xy, ceiling_agl: float,
                 wind_at, mean_wind_at, rng: np.random.Generator, zi: float | None = None,
                 vario_noise: float = 0.1, wind_est_noise: float = 1.0, mean_wind_agl=None):
        self.n = n
        self.perf = Performance(gp)
        self.ap = ap
        self.surface_at = surface_at
        self.ground_at = ground_at
        self.L = domain_m
        self.base = np.asarray(base_xy, float)
        self.wind_at = wind_at
        self.mean_wind_at = mean_wind_at
        self.mean_wind_agl = mean_wind_agl
        self.rng = rng
        self.vario_noise = vario_noise
        self.wind_est_noise = wind_est_noise
        self.auto = Autopilot(n, self.perf, ap, cp, surface_at, ground_at, domain_m, base_xy, ceiling_agl)
        if zi is not None:
            self.auto.zi = zi
        bz = float(ground_at(self.base[:1], self.base[1:])[0])
        self.base_ground = bz
        self.s = GliderState.at(n, self.base[0], self.base[1], bz, V=self.perf.gp.climb_speed,
                                E=self.perf.capacity_wh)
        self.s.active[:] = False
        self.est_err = np.zeros((n, 2))
        self.netto = np.zeros(n)
        self.log = SafetyLog()
        self._breach = np.zeros((n, n), bool)
        self._stall_prev = np.zeros(n, bool)
        # accumulators
        self.energy_used = np.zeros(n)
        self.motor_climb = np.zeros(n)
        self.lift_climb = np.zeros(n)
        self.mode_time = np.zeros((n, len(MODE_NAMES)))
        self.airborne_time = np.zeros(n)

    # ----------------------------------------------------------------- lifecycle
    def launch(self, i: int, t: float, heading: float | None = None):
        s = self.s
        # catapults sit evenly around the launch orbit, so a simultaneous launch is spread out
        ang = 2 * math.pi * i / self.n
        r = self.ap.launch_orbit_m
        s.x[i], s.y[i] = self.base[0] + r * math.cos(ang), self.base[1] + r * math.sin(ang)
        s.z[i] = float(self.surface_at(s.x[i:i + 1], s.y[i:i + 1])[0]) + self.ap.launch_release_m
        s.E[i] = self.perf.capacity_wh
        s.V[i] = self.perf.gp.climb_speed
        s.active[i] = True
        s.landed[i] = False
        s.phi[i] = 0.0
        self.est_err[i] = self.rng.normal(0.0, self.wind_est_noise, 2)
        self.auto.launch(i, s, t)
        s.psi[i] = ang + math.pi / 2 if heading is None else heading  # tangent to the anticlockwise orbit

    def place(self, i, x, y, z, psi, V=13.0, E=None, t=0.0):
        """Test helper: put drone i in the air (mode set by a following set_goal)."""
        s = self.s
        s.x[i], s.y[i], s.z[i], s.psi[i], s.V[i] = x, y, z, psi, V
        s.E[i] = self.perf.capacity_wh if E is None else E
        s.active[i] = True
        self.auto.mode[i] = 7  # LOITER placeholder until a goal arrives
        self.auto.goal[i] = None
        self.est_err[i] = self.rng.normal(0.0, self.wind_est_noise, 2)

    def wind_estimate(self):
        s = self.s
        u, v = self.mean_wind_at(s.x, s.y, s.z)
        return u + self.est_err[:, 0], v + self.est_err[:, 1]

    # ----------------------------------------------------------------- step
    def step(self, t: float, dt: float = 1.0):
        s, auto = self.s, self.auto
        u, v, w, ground = self.wind_at(s.x, s.y, s.z, return_ground=True)
        self.netto = w + self.rng.normal(0.0, self.vario_noise, self.n)
        if self.mean_wind_agl is not None:
            ue, ve = self.mean_wind_agl(s.z - ground)
            est = (ue + self.est_err[:, 0], ve + self.est_err[:, 1])
        else:
            est = self.wind_estimate()
        V, phi, tau = auto.step(t, s, est, self.netto, ground=ground)
        E0 = s.E.copy()
        act = s.active.copy()
        glider_step(s, V, phi, tau, (u, v, w), self.perf, dt)
        # accumulators
        self.energy_used += np.where(act, E0 - s.E, 0.0)
        motor = tau * self.perf.T * V / self.perf.W
        self.motor_climb += np.where(act, np.maximum(motor - s.sink, 0.0) * dt, 0.0)
        self.lift_climb += np.where(act, np.maximum(w, 0.0) * dt, 0.0)
        self.mode_time[np.arange(self.n), auto.mode] += dt
        self.airborne_time += np.where(act, dt, 0.0)
        st = act & s.stalled
        self.log.stalls += int((st & ~self._stall_prev).sum())
        self._stall_prev = st
        return self._terminations(t + dt)

    def _terminations(self, t):
        s, auto, ap = self.s, self.auto, self.ap
        out = []
        act = s.active
        if not act.any():
            return out
        surf = self.surface_at(s.x, s.y)
        dbase = np.hypot(s.x - self.base[0], s.y - self.base[1])
        contact = act & (s.z <= surf)
        for i in np.nonzero(contact)[0]:
            if auto.mode[i] == RTB_M and dbase[i] <= ap.land_radius_m:
                s.z[i] = surf[i]
                s.active[i] = False
                s.landed[i] = True
                auto.landed(i)
                out.append(("landed", int(i)))
                self.log.events.append((t, "landed", int(i), {"E": float(s.E[i])}))
            else:
                self._kill(i, t, "crash", {"z": float(s.z[i]), "surface": float(surf[i]),
                                           "mode": MODE_NAMES[auto.mode[i]]})
                self.log.crashes += 1
                out.append(("crash", int(i)))
        act = s.active
        outside = np.maximum.reduce([-s.x, -s.y, s.x - self.L, s.y - self.L])
        for i in np.nonzero(act & (outside > ap.geofence_violation_m))[0]:
            self._kill(i, t, "geofence_violation", {})
            self.log.geofence_violations += 1
            out.append(("geofence_violation", int(i)))
        act = s.active
        empty = act & (s.E <= 0)
        gnd = self.ground_at(s.x, s.y) if empty.any() else s.z
        for i in np.nonzero(empty & (s.z - gnd < ap.forced_landing_agl_m))[0]:
            self._kill(i, t, "forced_landing", {})
            self.log.forced_landings += 1
            out.append(("forced_landing", int(i)))
        # pairwise checks outside the base zone
        act = s.active & ~auto.bz
        idx = np.nonzero(act)[0]
        if len(idx) >= 2:
            x, y, z = s.x[idx], s.y[idx], s.z[idx]
            dh = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
            dz = np.abs(z[:, None] - z[None, :])
            d3 = np.hypot(dh, dz)
            coll = np.triu(d3 < ap.collision_m, 1)
            for a, b in zip(*np.nonzero(coll)):
                i, j = int(idx[a]), int(idx[b])
                for k in (i, j):
                    if s.active[k]:
                        self._kill(k, t, "collision", {"with": j if k == i else i})
                        self.log.crashes += 1
                self.log.collisions += 1
                out.append(("collision", i, j))
            br = np.triu((dh < ap.breach_h_m) & (dz < ap.breach_v_m), 1)
            cur = np.zeros_like(self._breach)
            cur[np.ix_(idx, idx)] = br
            new = cur & ~self._breach
            for i, j in zip(*np.nonzero(new)):
                self.log.separation_breaches += 1
                self.log.events.append((t, "separation_breach", (int(i), int(j)), {}))
            self._breach = cur
        else:
            self._breach[:] = False
        return out

    def _kill(self, i, t, kind, info):
        s = self.s
        s.active[i] = False
        s.crashed[i] = kind in ("crash", "collision")
        self.auto.down(i)
        self.log.events.append((t, kind, int(i), info))
