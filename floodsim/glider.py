"""Quasi-steady 3D point-mass motor-glider, vectorised across drones."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import G, RHO, GliderParams


class PerfTable:
    """Sink rate s(V, bank) tabulated from the drag polar at load, bilinear at run time."""

    def __init__(self, gp: GliderParams):
        self.gp = gp
        v0, v1, dv = gp.table_v
        b0, b1, db = gp.table_bank_deg
        self.v = np.arange(v0, v1 + dv / 2, dv)
        self.b = np.radians(np.arange(b0, b1 + db / 2, db))
        V, B = np.meshgrid(self.v, self.b, indexing="ij")
        self.sink_tab = gp.sink(V, B)          # [nV, nB]
        self.dv = dv
        self.db = math.radians(db)
        self.v0 = v0
        # fastest climb speed (max (T - D) V) and min-sink / best-glide speeds
        vs = np.linspace(gp.v_min, gp.v_max, 421)
        climb = (gp.thrust_max - gp.drag(vs, 0.0)) * vs / gp.weight
        self.v_fast_climb = float(vs[np.argmax(climb)])
        self.max_climb = float(climb.max())
        ld = gp.weight / gp.drag(vs, 0.0)
        self.v_best_glide = float(vs[np.argmax(ld)])

    def sink(self, V, bank):
        V = np.asarray(V, dtype=float)
        bank = np.abs(np.asarray(bank, dtype=float))
        fv = np.clip((V - self.v0) / self.dv, 0, len(self.v) - 1.000001)
        fb = np.clip(bank / self.db, 0, len(self.b) - 1.000001)
        i = fv.astype(np.int64)
        j = fb.astype(np.int64)
        a = fv - i
        c = fb - j
        T = self.sink_tab
        return (T[i, j] * (1 - a) + T[i + 1, j] * a) * (1 - c) + (T[i, j + 1] * (1 - a) + T[i + 1, j + 1] * a) * c

    def sink1(self, V: float, bank: float) -> float:
        return float(self.sink(V, bank))


@dataclass
class GliderState:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    V: np.ndarray
    psi: np.ndarray
    phi: np.ndarray
    E: np.ndarray
    stalled: np.ndarray
    crashed: np.ndarray
    landed: np.ndarray
    airborne: np.ndarray
    zdot: np.ndarray = None

    @staticmethod
    def create(n: int, x0: float, y0: float, z0: float, E0: float, psi0: float = 0.0) -> "GliderState":
        f = lambda v: np.full(n, float(v))
        b = lambda: np.zeros(n, dtype=bool)
        return GliderState(x=f(x0), y=f(y0), z=f(z0), V=f(12.0), psi=f(psi0), phi=f(0.0), E=f(E0), stalled=b(),
                           crashed=b(), landed=b(), airborne=b(), zdot=f(0.0))

    @property
    def n(self) -> int:
        return len(self.x)

    def copy(self) -> "GliderState":
        return GliderState(**{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()})


@dataclass
class StepInfo:
    power_w: np.ndarray
    motor_climb: np.ndarray       # m gained from the motor this step
    lift_climb: np.ndarray        # m gained from rising air this step (w > 0)
    h_air: np.ndarray
    stall_now: np.ndarray


def glider_step(s: GliderState, V_cmd, phi_cmd, tau, wind, gp: GliderParams, table: PerfTable, dt: float) -> StepInfo:
    """Advance all airborne drones by dt (Euler, mid-step heading for position). In place."""
    a = s.airborne
    u, v, w = wind
    V = np.clip(np.asarray(V_cmd, dtype=float), gp.v_min, gp.v_max)
    lim = math.radians(gp.bank_limit_safety_deg)
    phi = np.clip(np.asarray(phi_cmd, dtype=float), -lim, lim)
    tau = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
    tau = np.where(s.E <= 0.0, 0.0, tau)
    q = 0.5 * RHO * V * V * gp.wing_area
    cl = gp.weight / (q * np.cos(phi))
    stall_now = cl > gp.cl_max
    sink = table.sink(V, phi)
    climb_motor = tau * gp.thrust_max * V / gp.weight
    h_air = climb_motor - sink
    psidot = G * np.tan(phi) / V
    psi_mid = s.psi + 0.5 * psidot * dt
    Vh = np.sqrt(np.maximum(V * V - h_air * h_air, 0.25 * V * V))
    dx = (Vh * np.cos(psi_mid) + u) * dt
    dy = (Vh * np.sin(psi_mid) + v) * dt
    dz = (h_air + w) * dt
    power = tau * gp.thrust_max * V / gp.prop_eff + gp.payload_w
    s.x = np.where(a, s.x + dx, s.x)
    s.y = np.where(a, s.y + dy, s.y)
    s.z = np.where(a, s.z + dz, s.z)
    s.zdot = np.where(a, h_air + w, 0.0)
    s.psi = np.where(a, (s.psi + psidot * dt + np.pi) % (2 * np.pi) - np.pi, s.psi)
    s.V = np.where(a, V, s.V)
    s.phi = np.where(a, phi, s.phi)
    s.E = np.where(a, np.maximum(s.E - power * dt / 3600.0, 0.0), s.E)
    s.stalled = s.stalled | (a & stall_now)
    return StepInfo(power_w=np.where(a, power, 0.0), motor_climb=np.where(a, climb_motor * dt, 0.0),
                    lift_climb=np.where(a, np.maximum(w, 0.0) * dt, 0.0), h_air=h_air, stall_now=a & stall_now)
