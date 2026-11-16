"""Quasi-steady 3D point-mass motor-glider, vectorised over the swarm."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from .config import GliderParams


@dataclass
class GliderState:
    """Truth state, every field an array over drones."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    V: np.ndarray
    psi: np.ndarray
    phi: np.ndarray
    E: np.ndarray  # Wh
    stalled: np.ndarray
    crashed: np.ndarray
    landed: np.ndarray
    active: np.ndarray  # airborne and simulated
    vx: np.ndarray  # last ground velocity (for look-ahead)
    vy: np.ndarray
    vz: np.ndarray
    tau: np.ndarray  # last throttle
    w_air: np.ndarray  # last vertical air velocity at the glider
    sink: np.ndarray = None  # last still-air sink rate

    @classmethod
    def at(cls, n: int, x, y, z, psi=0.0, V=12.0, E=100.0) -> "GliderState":
        def f(v):
            return np.full(n, v, np.float64) if np.isscalar(v) else np.asarray(v, np.float64).copy()

        zb = np.zeros(n, bool)
        return cls(x=f(x), y=f(y), z=f(z), V=f(V), psi=f(psi), phi=np.zeros(n), E=f(E),
                   stalled=zb.copy(), crashed=zb.copy(), landed=zb.copy(), active=np.ones(n, bool),
                   vx=np.zeros(n), vy=np.zeros(n), vz=np.zeros(n), tau=np.zeros(n), w_air=np.zeros(n),
                   sink=np.zeros(n))

    @property
    def n(self) -> int:
        return len(self.x)

    def copy(self) -> "GliderState":
        return GliderState(**{f.name: getattr(self, f.name).copy() for f in fields(self)})


class Performance:
    """Drag-polar derived quantities and the sink-rate table s(V, phi)."""

    def __init__(self, gp: GliderParams):
        self.gp = gp
        self.m = gp.mass_kg
        self.g = gp.g
        self.W = gp.weight_n
        self.T = gp.thrust_max_n
        self.eta = gp.prop_eff
        self.payload = gp.payload_w
        self.v_min, self.v_max = gp.v_min, gp.v_max
        self.Vs = np.arange(gp.v_min, gp.v_max + 1e-9, gp.table_v_step)
        self.bank_deg = np.arange(0.0, gp.table_bank_max_deg + 1e-9, gp.table_bank_step_deg)
        Vg, Bg = np.meshgrid(self.Vs, np.radians(self.bank_deg), indexing="ij")
        self.table = (gp.drag(Vg, Bg) * Vg / self.W).astype(np.float64)  # [nV, nB]
        self.capacity_wh = gp.battery_capacity_wh
        climb = self.T * self.Vs / self.W - self.table[:, 0]
        self.fast_climb_speed = float(self.Vs[int(np.argmax(climb))])
        self.fast_climb_rate = float(climb.max())
        self.stall0 = float(gp.stall_speed(0.0))

    def sink(self, V, phi):
        V = np.minimum(np.maximum(np.asarray(V, float), self.Vs[0]), self.Vs[-1])
        b = np.minimum(np.abs(np.asarray(phi, float)) * (180.0 / np.pi), self.bank_deg[-1])
        fv = (V - self.Vs[0]) * (1.0 / self.gp.table_v_step)
        fb = b * (1.0 / self.gp.table_bank_step_deg)
        iv = np.minimum(fv.astype(np.int64), len(self.Vs) - 2)
        ib = np.minimum(fb.astype(np.int64), len(self.bank_deg) - 2)
        tv = fv - iv
        tb = fb - ib
        nb = self.table.shape[1]
        t = self.table.ravel()
        k = iv * nb + ib
        a = t[k] + (t[k + 1] - t[k]) * tb
        c = t[k + nb] + (t[k + nb + 1] - t[k + nb]) * tb
        return a + (c - a) * tv

    def drag(self, V, phi=0.0):
        return self.gp.drag(V, phi)

    def level_throttle(self, V, phi=0.0):
        return self.drag(V, phi) / self.T

    def level_power(self, V):
        """Electrical power (W) for level flight at airspeed V."""
        return self.drag(V) * np.asarray(V, float) / self.eta + self.payload

    def stall_speed(self, phi=0.0):
        return self.stall0 / np.sqrt(np.cos(np.minimum(np.abs(phi), 1.4)))

    def cl(self, V, phi):
        return 2 * self.W / (self.gp.rho * np.asarray(V, float) ** 2 * self.gp.wing_area_m2 * np.cos(phi))


def glider_step(s: GliderState, V_cmd, phi_cmd, tau, wind, perf: Performance, dt: float):
    """Advance all active drones in place. wind = (u, v, w) arrays."""
    a = s.active
    gp = perf.gp
    lim = math.radians(gp.bank_limit_safety_deg)
    V = np.minimum(np.maximum(V_cmd, perf.v_min), perf.v_max)
    phi = np.minimum(np.maximum(phi_cmd, -lim), lim)
    tau = np.where(s.E > 0, np.minimum(np.maximum(tau, 0.0), 1.0), 0.0)
    s.stalled[:] = perf.cl(V, phi) > gp.cl_max
    sink = perf.sink(V, phi)
    s.sink = sink
    hdot = tau * perf.T * V / perf.W - sink
    psidot = perf.g * np.tan(phi) / V
    psi_mid = s.psi + 0.5 * psidot * dt
    Vh = np.sqrt(np.maximum(V * V - hdot * hdot, 0.25 * V * V))
    u, v, w = wind
    vx = Vh * np.cos(psi_mid) + u
    vy = Vh * np.sin(psi_mid) + v
    vz = hdot + w
    s.x = np.where(a, s.x + vx * dt, s.x)
    s.y = np.where(a, s.y + vy * dt, s.y)
    s.z = np.where(a, s.z + vz * dt, s.z)
    s.psi = np.where(a, np.mod(s.psi + psidot * dt + np.pi, 2 * np.pi) - np.pi, s.psi)
    s.V = np.where(a, V, s.V)
    s.phi = np.where(a, phi, s.phi)
    s.E = np.where(a, np.maximum(s.E - (tau * perf.T * V / perf.eta + perf.payload) / 3600.0 * dt, 0.0), s.E)
    s.vx, s.vy, s.vz = np.where(a, vx, 0.0), np.where(a, vy, 0.0), np.where(a, vz, 0.0)
    s.tau = np.where(a, tau, 0.0)
    s.w_air = np.where(a, w, 0.0)
    return s


def terrain_contact(s: GliderState, surface_at) -> np.ndarray:
    """Active drones at or below the surface (terrain + trees + buildings)."""
    return s.active & (s.z <= surface_at(s.x, s.y))
