"""Simulation loop, PlannerView (the information boundary) and EpisodeResult."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .autopilot import DOWN, GROUND, LAUNCH, MODE_NAMES, RTB_M, SEARCH, THERMAL, VERIFY
from .config import Config, SimConfig, TeamInputs
from .controllers import Controller
from .flight import FlightCore
from .people import People
from .radio import BASE_ID, Radio
from .rng import stream
from .sensor import Sensor


# --------------------------------------------------------------------------
# Views: whitelisted copies only
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnState:
    x: float
    y: float
    z: float
    V: float
    psi: float
    phi: float
    E_wh: float
    battery_frac: float
    height_above_ground: float
    airborne: bool
    alive: bool


@dataclass
class PlannerView:
    """Everything one drone's controller may know. Built by copying whitelisted fields."""

    t: float
    drone_id: int
    n_drones: int
    own: OwnState
    netto: float
    wind_est: tuple  # mean wind estimate at the drone's height (u, v)
    wind10_est: tuple  # the same estimate referred to 10 m
    forecast: object  # FloodForecast (arrays only)
    maps: object  # StaticMaps
    base_xy: tuple
    team: TeamInputs
    status: dict  # autopilot status: mode, leg_idx, ...
    events: list  # (kind, info) since the last decision
    messages: list  # radio messages received since the last decision
    frames: tuple  # all of this drone's sensor frames so far (incl. empty ones)
    new_frames: tuple
    new_detections: list  # (t, cell)
    new_verifications: list  # (t, cell, confirmed)
    episode_s: float
    outbox: list = field(default_factory=list)

    def send(self, payload, to: int | None = None):
        self.outbox.append((payload, to))


@dataclass
class EpisodeResult:
    spec_hash: str
    seed: int
    controller: str
    controller_params: dict
    n_drones: int
    metrics: dict
    trajectory: dict  # arrays [T, n_drones] + "t"
    finds: list  # (person, t, state_at_find, drone)
    verifications: list  # (t, drone, cell, confirmed, person or None)
    detections: list  # (t, drone, cell, true_person or None)
    mode_time: dict
    energy_used_wh: list
    safety_events: list
    people_final: dict
    snapshots: dict
    wall_s: float = 0.0


# --------------------------------------------------------------------------


class Sim:
    def __init__(self, world, controller: Controller, sim_cfg: SimConfig | None = None,
                 n_drones: int | None = None, cfg: Config | None = None):
        self.world = world
        self.cfg = cfg or world.cfg
        self.sc = sim_cfg or self.cfg.sim
        self.controller = controller
        spec = world.spec
        self.n = int(n_drones or self.cfg.swarm.n_drones or spec.n_drones)
        terr = world.terrain
        self.terr = terr
        self.episode_s = self.sc.episode_s or spec.episode_s
        team = self.cfg.team
        self.team = team
        self.flight = FlightCore(
            self.n, self.cfg.glider, self.cfg.autopilot, self.cfg.controller, terr.surface_at, terr.ground_at,
            terr.domain_m, terr.base_xy, team.ceiling_agl_m, world.weather.wind_at, self._mean_wind_at,
            stream(spec.seed, "sim", spec.attempt), zi=spec.zi_m,
            vario_noise=self.cfg.sensor.vario_noise_ms, wind_est_noise=self.cfg.sensor.wind_est_noise_ms,
            mean_wind_agl=world.weather.mean_wind)
        self.people = People(world.people0, spec, terr, self.cfg.people)
        self.sensor = Sensor(self.cfg.sensor, terr, team.camera_range_m, self.n,
                             stream(spec.seed, "sensor", spec.attempt))
        self.radio = Radio(team.signal_radius_km * 1000.0, perfect=self.cfg.swarm.perfect_comms)
        self.maps = world.forecast.maps
        # per-drone agent-side buffers
        self.frames = [[] for _ in range(self.n)]
        self.frame_mark = [0] * self.n
        self.ev = [[] for _ in range(self.n)]
        self.inbox = [[] for _ in range(self.n)]
        self.new_det = [[] for _ in range(self.n)]
        self.new_ver = [[] for _ in range(self.n)]
        self.last_decision = np.zeros(self.n)
        self.ready_t = np.full(self.n, np.inf)
        self.retired = np.zeros(self.n, bool)
        # truth-side logs and accumulators
        N = terr.n * terr.n
        self.first_seen = np.full(N, np.inf, np.float32)
        self.seen_count = np.zeros(N, np.int32)
        self.last_seen_by = np.full((self.n, N), -np.inf, np.float32)
        self.repeat = np.zeros(N, bool)
        self.detections: list = []
        self.verifs: list = []
        self.finds: list = []
        self.verify_fa_time = 0.0
        self.relaunches: list = []
        self.verify_started = {}
        self.snapshots = {"t": [], "seen": []}
        self.log_dt = self.sc.fast_log_dt_s if self.sc.fast else self.sc.log_dt_s
        self.traj = {k: [] for k in ("t", "x", "y", "z", "V", "phi", "E", "mode", "netto")}

    def _mean_wind_at(self, x, y, z):
        wx = self.world.weather
        return wx.mean_wind(np.asarray(z) - self.terr.ground_at(x, y))

    # ------------------------------------------------------------------ views
    def _status(self, i):
        a = self.flight.auto
        return {"mode": MODE_NAMES[a.mode[i]], "leg_idx": int(a.leg_idx[i]),
                "search_phase": int(a.search_phase[i]), "rtb_phase": int(a.rtb_phase[i])}

    def view(self, i, t) -> PlannerView:
        s = self.flight.s
        cap = self.flight.perf.capacity_wh
        gz = float(self.terr.ground_at(s.x[i:i + 1], s.y[i:i + 1])[0])
        mode = self.flight.auto.mode[i]
        own = OwnState(float(s.x[i]), float(s.y[i]), float(s.z[i]), float(s.V[i]), float(s.psi[i]),
                       float(s.phi[i]), float(s.E[i]), float(s.E[i] / cap), float(s.z[i] - gz),
                       bool(s.active[i]), bool(mode != DOWN))
        ue, ve = self.flight.wind_estimate()
        h = max(own.height_above_ground, 2.0)
        z0 = self.cfg.weather.z0_m
        f = math.log(min(h, self.cfg.weather.profile_top_m) / z0) / math.log(10 / z0)
        mark = self.frame_mark[i]
        fr = self.frames[i]
        v = PlannerView(
            t=float(t), drone_id=i, n_drones=self.n, own=own, netto=float(self.flight.netto[i]),
            wind_est=(float(ue[i]), float(ve[i])), wind10_est=(float(ue[i] / f), float(ve[i] / f)),
            forecast=self.world.forecast, maps=self.maps, base_xy=tuple(self.terr.base_xy), team=self.team,
            status=self._status(i), events=list(self.ev[i]), messages=list(self.inbox[i]),
            frames=tuple(fr), new_frames=tuple(fr[mark:]), new_detections=list(self.new_det[i]),
            new_verifications=list(self.new_ver[i]), episode_s=float(self.episode_s))
        self.ev[i].clear()
        self.inbox[i].clear()
        self.new_det[i].clear()
        self.new_ver[i].clear()
        self.frame_mark[i] = len(fr)
        return v

    def _apply_goals(self, goals, views, t):
        for i, g in (goals or {}).items():
            if g is not None:
                if self.flight.auto.mode[i] == VERIFY and i in self.verify_started and not hasattr(g, "cell"):
                    self.verify_started.pop(i, None)
                if hasattr(g, "cell"):
                    self.verify_started[i] = t
                self.flight.auto.set_goal(i, g, t)
        for i, v in views.items():
            for payload, to in v.outbox:
                self.radio.send(i, payload, t, to)

    # ------------------------------------------------------------------ run
    def run(self) -> EpisodeResult:
        t0 = time.perf_counter()
        sc = self.sc
        fl = self.flight
        n = self.n
        views = {i: self.view(i, 0.0) for i in range(n)}
        goals = self.controller.reset(views)
        self._apply_goals(goals, views, 0.0)
        for i in range(n):
            fl.launch(i, 0.0)
        t = 0.0
        dt = sc.dt_s
        sensor_every = int(round(sc.sensor_dt_s / dt))
        people_every = int(round(sc.people_dt_s / dt))
        radio_every = int(round(sc.radio_dt_s / dt))
        log_every = int(round(self.log_dt / dt))
        snap_every = int(round(sc.snapshot_dt_s / dt))
        replan = self.cfg.controller.replan_s
        step = 0
        ep = self.episode_s
        cov_win = sc.coverage_window_s
        terr = self.terr
        while t < ep - 1e-9:
            # physics, autopilot, safety, separation
            term = fl.step(t, dt)
            step += 1
            t = step * dt
            for ev in term:
                kind, i = ev[0], ev[1]
                if kind == "landed":
                    self.ev[i].append(("landed", {}))
                    if sc.battery_swap:
                        self.ready_t[i] = t + self.team.charge_time_min * 60.0
                else:
                    for j in ev[1:]:
                        self.ev[j].append((kind, {}))
                        self.retired[j] = True
                        self.verify_started.pop(j, None)
            for e in fl.auto.events:
                self.ev[e.drone].append((e.kind, e.info))
                if e.kind == "verify_done":
                    self._resolve_verify(e.drone, int(e.info["cell"]), t)
            fl.auto.events.clear()
            # sensing
            if step % sensor_every == 0:
                self._sense(t, cov_win)
            # radio
            if step % radio_every == 0 and self.radio.queue:
                nodes = {BASE_ID: tuple(terr.base_xy)}
                for i in range(n):
                    if not self.retired[i]:
                        nodes[i] = (float(fl.s.x[i]), float(fl.s.y[i]))
                inbox = self.radio.deliver(nodes)
                for k, msgs in inbox.items():
                    if k != BASE_ID and msgs:
                        self.inbox[k].extend(msgs)
            # people
            if step % people_every == 0:
                self.people.step(t, self.world.flood, sc.people_dt_s)
            # battery swap and relaunch
            for i in np.nonzero(t >= self.ready_t)[0]:
                self.ready_t[i] = np.inf
                if ep - t >= sc.relaunch_min_remaining_s:
                    self.relaunches.append((t, int(i)))
                    self.ev[i].append(("relaunch", {}))
                    v = self.view(i, t)
                    g = self.controller.decide({i: v})
                    self._apply_goals(g, {i: v}, t)
                    fl.launch(i, t)
                else:
                    self.retired[i] = True
            # decisions
            due = [i for i in range(n) if not self.retired[i] and
                   (self.ev[i] or self.new_det[i] or self.new_ver[i] or t - self.last_decision[i] >= replan)]
            if due:
                vs = {i: self.view(i, t) for i in due}
                for i in due:
                    self.last_decision[i] = t
                self._apply_goals(self.controller.decide(vs), vs, t)
            # logging
            if step % log_every == 0:
                self._log(t)
            if self.sc.record_grids and not self.sc.fast and step % snap_every == 0:
                self.snapshots["t"].append(t)
                self.snapshots["seen"].append(self.seen_count.astype(np.uint16).copy())
            # termination
            if self.retired.all() or (~self.people.active).all():
                break
        t_end = t
        # unfound people keep accumulating danger until episode end + 2 h
        horizon = ep + self.cfg.people.horizon_after_s
        tp = (math.floor(t_end / sc.people_dt_s) + 1) * sc.people_dt_s
        while tp <= horizon + 1e-9:
            self.people.step(tp, self.world.flood, sc.people_dt_s)
            tp += sc.people_dt_s
        from .metrics import compute_metrics

        metrics = compute_metrics(self, t_end)
        traj = {k: np.asarray(v) for k, v in self.traj.items()}
        p = self.people
        res = EpisodeResult(
            spec_hash=self.world.spec.spec_hash, seed=self.world.spec.seed, controller=self.controller.name,
            controller_params=self.controller.params(), n_drones=n, metrics=metrics, trajectory=traj,
            finds=self.finds, verifications=self.verifs, detections=self.detections,
            mode_time={MODE_NAMES[k]: float(fl.mode_time[:, k].sum()) for k in range(len(MODE_NAMES))},
            energy_used_wh=[float(e) for e in fl.energy_used], safety_events=fl.log.events,
            people_final={"x": p.x.copy(), "y": p.y.copy(), "state": p.state.copy(), "found": p.found.copy(),
                          "survival": p.survival(), "t_affected": p.t_affected.copy(),
                          "x0": self.world.people0.x.copy(), "y0": self.world.people0.y.copy()},
            snapshots={k: np.asarray(v) for k, v in self.snapshots.items()},
            wall_s=time.perf_counter() - t0)
        return res

    # ------------------------------------------------------------------ pieces
    def _sense(self, t, cov_win):
        fl = self.flight
        s = fl.s
        act = np.nonzero(s.active)[0]
        if len(act) == 0:
            return
        gz = self.terr.ground_at(s.x[act], s.y[act])
        for k, i in enumerate(act):
            f, truth = self.sensor.frame(t, int(i), float(s.x[i]), float(s.y[i]), float(s.z[i]), self.people,
                                         float(gz[k]))
            if f is None:
                continue
            self.frames[i].append(f)
            c = f.cell_idx
            fs = self.first_seen[c]
            self.first_seen[c] = np.minimum(fs, t)
            self.seen_count[c] += 1
            if self.n > 1:
                self.last_seen_by[i, c] = -np.inf  # exclude this drone's own looks
                self.repeat[c] |= (self.last_seen_by[:, c] >= t - self.sc.repeat_window_s).any(axis=0)
            self.last_seen_by[i, c] = t
            if f.detections:
                true_cells = {cell: person for person, cell in truth}
                for cell in f.detections:
                    self.new_det[i].append((t, int(cell)))
                    self.detections.append((t, int(i), int(cell), true_cells.get(cell)))

    def _resolve_verify(self, i, cell, t):
        ok, person = self.sensor.verify(cell, self.people, t)
        if ok and person is not None:
            self.people.mark_found(person, t)
            self.finds.append((person, t, int(self.people.state_at_find[person]), int(i)))
        t_start = self.verify_started.pop(i, t - self.cfg.autopilot.verify_s)
        if person is None:
            self.verify_fa_time += t - t_start
        self.verifs.append((t, int(i), int(cell), bool(ok), person))
        self.new_ver[i].append((t, int(cell), bool(ok)))

    def _log(self, t):
        s = self.flight.s
        tr = self.traj
        tr["t"].append(t)
        tr["x"].append(s.x.copy())
        tr["y"].append(s.y.copy())
        tr["z"].append(s.z.copy())
        tr["V"].append(s.V.copy())
        tr["phi"].append(s.phi.copy())
        tr["E"].append(s.E.copy())
        tr["mode"].append(self.flight.auto.mode.copy())
        tr["netto"].append(self.flight.netto.copy())
