"""Small flight harness: autopilot + glider physics + termination on synthetic or real terrain."""
import math

import numpy as np

from floodsim.autopilot import GROUND, RTBM, Autopilot
from floodsim.config import AutopilotParams, GliderParams
from floodsim.glider import GliderState, PerfTable, glider_step


class Harness:
    def __init__(self, surface, terrain=None, n=1, base=(5000.0, 5000.0), cell=50.0, wind=(0.0, 0.0),
                 thermals=(), zi=600.0, gp=None, ap=None, e_soar=0.6, w_enter=0.8, ceiling=400.0):
        self.surface = np.asarray(surface, dtype=np.float64)
        self.terrain = self.surface if terrain is None else np.asarray(terrain, dtype=np.float64)
        self.gp = gp or GliderParams()
        self.table = PerfTable(self.gp)
        self.p = ap or AutopilotParams()
        self.cell = cell
        self.L = self.surface.shape[0] * cell
        self.ap = Autopilot(n, self.gp, self.table, self.p, self.surface, self.terrain, cell, self.L, base, ceiling,
                            zi, e_soar=e_soar, w_enter=w_enter)
        self.base = base
        self.n = n
        self.gs = GliderState.create(n, base[0], base[1], 0.0, self.gp.capacity_wh)
        self.wind = np.asarray(wind, dtype=float)
        self.thermals = list(thermals)      # (x, y, w_core, r_t)
        self.zi = zi
        self.t = 0.0
        self.crashed = np.zeros(n, bool)
        self.landed = np.zeros(n, bool)
        self.land_E = np.full(n, np.nan)
        self.land_xy = [None] * n
        self.log = []

    def place(self, i, x, y, z, psi=0.0, V=13.0, E=None, mode=None, goal=None):
        g = self.gs
        g.x[i], g.y[i], g.z[i], g.psi[i], g.V[i] = x, y, z, psi, V
        g.airborne[i] = True
        if E is not None:
            g.E[i] = E
        if goal is not None:
            self.ap.d[i].mode = 1
            self.ap.set_goal(i, goal, g, math.atan2(self.wind[1], self.wind[0]), self.t)

    def w_at(self, x, y, z):
        w = np.zeros(self.n)
        for (tx, ty, wc, rt) in self.thermals:
            r2 = (x - tx) ** 2 + (y - ty) ** 2
            s = (z - self._terr(x, y)) / self.zi
            from floodsim.weather import thermal_shape
            w += wc * np.exp(-r2 / rt ** 2) * thermal_shape(s)
        return w

    def _terr(self, x, y):
        return np.array([self.ap.terrain_at(a, b) for a, b in zip(np.atleast_1d(x), np.atleast_1d(y))])

    def step(self, sep_dz=None):
        g = self.gs
        n = self.n
        u = np.full(n, self.wind[0])
        v = np.full(n, self.wind[1])
        w = self.w_at(g.x, g.y, g.z)
        est = np.tile(self.wind, (n, 1))
        V, phi, tau, ev = self.ap.step(g, w, est, self.t, sep_dz)
        info = glider_step(g, V, phi, tau, (u, v, w), self.gp, self.table, 1.0)
        for i in np.flatnonzero(g.airborne):
            surf = self.ap.surf_at(g.x[i], g.y[i])
            d = self.ap.d[i]
            if g.z[i] <= surf:
                if d.mode == RTBM and math.hypot(g.x[i] - self.base[0], g.y[i] - self.base[1]) <= self.p.land_radius:
                    self.landed[i] = True
                    self.land_E[i] = g.E[i]
                    self.land_xy[i] = (g.x[i], g.y[i])
                    d.mode = GROUND
                else:
                    self.crashed[i] = True
                g.airborne[i] = False
        self.log.append((self.t, g.x.copy(), g.y.copy(), g.z.copy(), [d.mode for d in self.ap.d], tau.copy(), phi.copy()))
        self.t += 1.0
        return ev, info

    def run(self, steps, until=None):
        evs = []
        for _ in range(steps):
            ev, _ = self.step()
            evs += ev
            if until is not None and until(self, evs):
                break
            if not self.gs.airborne.any():
                break
        return evs


def flat(n=200, z=50.0):
    return np.full((n, n), z)
