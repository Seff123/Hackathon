"""Simulation loop, the information boundary (PlannerView) and EpisodeResult."""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import rng as rngmod
from .autopilot import DOWN, GROUND, MODE_NAMES, RTBM, SEARCH, THERMAL, TRANSIT, VERIFY, Autopilot
from .config import AutopilotParams, ControllerParams, GliderParams, SimConfig
from .controllers import Controller
from .forecast import FloodForecast
from .glider import GliderState, PerfTable, glider_step
from .goals import Search, Verify
from .people import STATE_NAMES, hazard_rating
from .radio import BASE, Radio
from .sensor import Sensor, SensorFrame


# ----------------------------------------------------------------------------- agent-side types
@dataclass(frozen=True)
class StaticMaps:
    dem: np.ndarray
    surface: np.ndarray
    landcover: np.ndarray
    road_mask: np.ndarray
    population: np.ndarray
    hand: np.ndarray
    depression_depth: np.ndarray
    building_density: np.ndarray
    dist_to_river: np.ndarray
    cell: float
    domain: float
    towns: tuple

    @staticmethod
    def from_terrain(t, domain: float) -> "StaticMaps":
        ro = lambda a: _readonly(np.array(a, copy=True))
        return StaticMaps(dem=ro(t.z), surface=ro(t.surface), landcover=ro(t.landcover), road_mask=ro(t.road_mask),
                          population=ro(t.population), hand=ro(t.hand), depression_depth=ro(t.depression_depth),
                          building_density=ro(t.building_density), dist_to_river=ro(t.dist_to_river),
                          cell=float(t.cell_size), domain=float(domain),
                          towns=tuple((tw["name"], tw["x"], tw["y"], tw["r"]) for tw in t.towns))


def _readonly(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


@dataclass
class OwnState:
    x: float
    y: float
    z: float
    V: float
    psi: float
    phi: float
    E: float
    mode: str
    airborne: bool
    down: bool
    goal_kind: str | None
    wpi: int | None
    n_wps: int
    loiter: bool
    e_return: float
    height_above_surface: float


@dataclass
class PlannerView:
    t: float
    drone_id: int
    n_drones: int
    own: OwnState
    battery_frac: float
    netto: float
    wind_est: tuple
    forecast: FloodForecast
    maps: StaticMaps
    base_xy: tuple
    inbox: list
    frames: list
    new_frames: list
    new_detections: list
    verifications: list
    new_verifications: list
    rain_radar: np.ndarray | None
    n_hat: float
    episode_end_s: float
    horizon_s: float
    camera_range_m: float
    signal_radius_m: float
    glider: GliderParams
    autopilot: AutopilotParams
    connected_to_base: bool
    reasons: list = field(default_factory=list)


@dataclass
class EpisodeResult:
    spec_hash: str
    seed: int
    controller: str
    controller_params: dict
    n_drones: int
    metrics: dict
    trajectory: np.ndarray
    traj_fields: tuple
    finds: list
    verification_log: list
    mode_time: dict
    energy_log: dict
    safety_events: list
    detections_log: list
    decision_log: list = field(default_factory=list)
    recording: dict | None = None
    wall_s: float = 0.0
    planner_diag: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    people_final: list = field(default_factory=list)


TRAJ_FIELDS = ("t", "drone", "x", "y", "z", "V", "phi", "E", "mode", "netto")


# ----------------------------------------------------------------------------- simulation
class Sim:
    def __init__(self, world, controller: Controller, cfg: SimConfig | None = None, n_drones: int | None = None,
                 controller_params: ControllerParams | None = None):
        self.world = world
        self.cfg = cfg or SimConfig()
        self.controller = controller
        wcfg = world.cfg
        spec = world.spec
        self.spec = spec
        self.n = int(n_drones or self.cfg.n_drones or spec.n_drones)
        n = self.n
        self.gp = GliderParams.from_team(spec.team)
        self.table = PerfTable(self.gp)
        self.ap_params = wcfg.autopilot
        cp = controller_params or getattr(controller, "params", None) or wcfg.controller
        e_soar = getattr(cp, "e_soar", wcfg.controller.e_soar)
        w_enter = getattr(cp, "w_enter", wcfg.controller.w_enter)
        ter = world.terrain
        self.terrain = ter
        self.domain = wcfg.world.domain_m
        self.base = tuple(ter.base_xy)
        self.ceiling = spec.team.ceiling_agl
        self.autopilot = Autopilot(n, self.gp, self.table, self.ap_params, ter.surface, ter.z, ter.cell_size,
                                   self.domain, self.base, self.ceiling, world.weather.zi, e_soar=e_soar,
                                   w_enter=w_enter)
        self.maps = StaticMaps.from_terrain(ter, self.domain)
        rs = rngmod.streams(spec.seed, spec.attempt)
        self.rng = rs["sim"]
        self.sensor = Sensor(wcfg.sensor, ter, spec.camera_range_m, spec.lambda_fa, rs["sensor"])
        self.verify_rng = rngmod.child(spec.seed, spec.attempt, 991)
        perfect = self.cfg.perfect_comms
        self.radio = Radio(n, spec.signal_radius_km * 1000.0, self.base, perfect=perfect)
        self.people = copy.deepcopy(world.people)
        self.n_hat = float(len(self.people.x) * math.exp(wcfg.people.n_hat_sigma * self.rng.standard_normal()))
        self.wind_bias = self.rng.normal(0.0, wcfg.weather.wind_est_noise, size=(n, 2))
        self.netto_sd = wcfg.weather.netto_noise
        self.rain_noise = 1.0 + wcfg.people.rain_radar_noise * float(np.clip(self.rng.standard_normal(), -2.5, 2.5))
        bz = float(ter.surface[int(self.base[1] // ter.cell_size), int(self.base[0] // ter.cell_size)])
        self.gs = GliderState.create(n, self.base[0], self.base[1], bz, self.gp.capacity_wh)
        self.episode_s = wcfg.world.episode_s
        self.horizon_s = self.episode_s + wcfg.world.post_episode_s
        self.t = 0.0
        # agent-side buffers
        self.frames = {i: [] for i in range(n)}
        self.obs_cursor = {i: 0 for i in range(n)}
        self.det_pending = {i: [] for i in range(n)}
        self.verifs = {i: [] for i in range(n)}
        self.verif_pending = {i: [] for i in range(n)}
        self.inbox = {i: [] for i in range(n)}
        self.reasons = {i: set() for i in range(n)}
        self.last_decision = np.zeros(n)
        self.swap_until = np.full(n, np.inf)
        self.down = np.zeros(n, dtype=bool)
        self.verify_started = np.zeros(n)
        # truth-side accumulators
        nn = ter.n
        self.first_obs = np.full(nn * nn, np.inf)
        self.last_seen_t = np.full(nn * nn, -np.inf)
        self.last_seen_d = np.full(nn * nn, -1, dtype=np.int64)
        self.repeat = np.zeros(nn * nn, dtype=bool)
        self.observed = np.zeros(nn * nn, dtype=bool)
        self.mode_time = np.zeros((n, len(MODE_NAMES)))
        self.energy_used = np.zeros(n)
        self.motor_climb = np.zeros(n)
        self.lift_climb = np.zeros(n)
        self.stall_steps = np.zeros(n, dtype=int)
        self.safety = []
        self.counts = {"crash": 0, "forced_landing": 0, "geofence_violation": 0, "midair": 0, "sep_breach": 0,
                       "landing": 0, "relaunch": 0, "rtb": 0, "true_det": 0, "false_det": 0, "frames": 0}
        self.sep_breach_pairs = set()
        self._conflicts = {}
        self._prev_xy = (self.gs.x.copy(), self.gs.y.copy())
        self.fa_verify_time = 0.0
        self.verify_log = []
        self.det_log = []
        self.traj = []
        self.events = []      # for recording
        self.recording = None
        if self.cfg.record_grids:
            self.recording = {"frames": [], "people": [], "debug": [], "dt": self.cfg.grid_dt}

    # ------------------------------------------------------------------ helpers
    def wind_est10(self, i: int) -> tuple:
        u, v = self.world.weather.mean_wind(10.0)
        return (float(u + self.wind_bias[i, 0]), float(v + self.wind_bias[i, 1]))

    def wind_est_at(self, i: int, z_agl: float) -> tuple:
        from .controllers import wind_at_height
        u10, v10 = self.wind_est10(i)
        return wind_at_height(u10, v10, z_agl)

    def own_state(self, i: int) -> OwnState:
        g = self.gs
        d = self.autopilot.d[i]
        surf = self.autopilot.surf_at(g.x[i], g.y[i])
        return OwnState(x=float(g.x[i]), y=float(g.y[i]), z=float(g.z[i]), V=float(g.V[i]), psi=float(g.psi[i]),
                        phi=float(g.phi[i]), E=float(g.E[i]), mode=MODE_NAMES[d.mode], airborne=bool(g.airborne[i]),
                        down=bool(self.down[i]), goal_kind=(d.goal.kind if d.goal is not None else None),
                        wpi=(d.wpi if isinstance(d.goal, Search) else None), n_wps=len(d.wps), loiter=d.loiter,
                        e_return=float(d.e_return), height_above_surface=float(g.z[i] - surf))

    def make_view(self, i: int, reasons=(), for_observe: bool = False) -> PlannerView:
        fr = self.frames[i]
        if for_observe:
            new_frames = fr[self.obs_cursor[i]:]
            self.obs_cursor[i] = len(fr)
            new_dets, new_ver = [], []
            inbox = []
        else:
            new_frames = []
            new_dets = self.det_pending[i]
            self.det_pending[i] = []
            new_ver = self.verif_pending[i]
            self.verif_pending[i] = []
            inbox = self.inbox[i]
            self.inbox[i] = []
        rr = self._rain_radar
        return PlannerView(
            t=self.t, drone_id=i, n_drones=self.n, own=self.own_state(i), battery_frac=float(self.gs.E[i] / self.gp.capacity_wh),
            netto=float(self._netto[i]), wind_est=self.wind_est10(i), forecast=self.world.forecast, maps=self.maps,
            base_xy=self.base, inbox=inbox, frames=fr, new_frames=new_frames, new_detections=new_dets,
            verifications=list(self.verifs[i]), new_verifications=new_ver, rain_radar=rr, n_hat=self.n_hat,
            episode_end_s=self.episode_s, horizon_s=self.horizon_s, camera_range_m=self.spec.camera_range_m,
            signal_radius_m=self.spec.signal_radius_km * 1000.0, glider=self.gp, autopilot=self.ap_params,
            connected_to_base=bool(self._connected[i]), reasons=list(reasons))

    def _apply_goals(self, goals: dict) -> None:
        wd = math.atan2(*self.wind_est10(0)[::-1])
        for i, g in (goals or {}).items():
            if g is None or self.down[i]:
                continue
            if isinstance(g, Verify):
                self.verify_started[i] = self.t
            self.autopilot.set_goal(int(i), g, self.gs, wd, self.t)
            self.last_decision[i] = self.t

    def _update_rain_radar(self) -> None:
        if not self.spec.has_pluvial:
            self._rain_radar = None
            return
        c = self.world.cfg.world.flood_cell_m
        xs = (np.arange(int(self.domain // c)) + 0.5) * c
        X, Y = np.meshgrid(xs, xs)
        self._rain_radar = _readonly((self.world.weather.rain(X, Y, self.t) * self.rain_noise * 3.6e6).astype(np.float32))

    # ------------------------------------------------------------------ main loop
    def run(self) -> EpisodeResult:
        wall0 = time.perf_counter()
        cfg = self.cfg
        n = self.n
        gs = self.gs
        ap = self.autopilot
        weather = self.world.weather
        self._netto = np.zeros(n)
        self._connected = np.ones(n, dtype=bool)
        self._rain_radar = None
        self._est10 = np.array([self.wind_est10(i) for i in range(n)])
        self._update_rain_radar()
        views = {i: self.make_view(i, ["reset"]) for i in range(n)}
        goals = self.controller.reset(views)
        for i in range(n):
            if i in goals:
                ap.d[i].goal = goals[i]
            ap.launch(i, gs, 0.0, slot=i)
        self._apply_goals(goals)
        sensor_every = int(round(cfg.sensor_dt / cfg.dt))
        people_every = int(round(cfg.people_dt / cfg.dt))
        traj_every = int(round((cfg.traj_dt_fast if cfg.fast else cfg.traj_dt) / cfg.dt))
        grid_every = int(round(cfg.grid_dt / cfg.dt))
        radar_every = int(round(300.0 / cfg.dt))
        replan = getattr(getattr(self.controller, "params", None), "replan_s", 60.0)
        has_observe = type(self.controller).observe is not Controller.observe
        dt = cfg.dt
        step = 0
        max_steps = cfg.max_steps or int(self.episode_s / dt)
        app = self.ap_params
        L = self.domain
        while step < max_steps:
            t = self.t
            air = gs.airborne
            # --- truth wind at the drones and what they sense
            u, v, w = weather.wind(gs.x, gs.y, gs.z, t)
            self._netto = w + self.rng.normal(0.0, self.netto_sd, n)
            hg = gs.z - _terrain_at(self.terrain, gs.x, gs.y)
            zz = np.clip(hg, 2.0, 600.0)
            fac = np.log(zz / 0.1) / math.log(10.0 / 0.1)
            est = self._est10 * fac[:, None]
            # --- separation (vectorised over pairs, outside the base zone)
            sep_dz, sep_turn = self._separation(t)
            # --- autopilot and physics
            Vc, phic, tau, evs = ap.step(gs, self._netto, est, t, sep_dz, sep_turn)
            info = glider_step(gs, Vc, phic, tau, (u, v, w), self.gp, self.table, dt)
            self.energy_used += info.power_w * dt / 3600.0
            self.motor_climb += info.motor_climb
            self.lift_climb += info.lift_climb
            self.stall_steps += info.stall_now
            modes = ap.modes()
            self.mode_time[np.arange(n), modes] += np.where(air | (modes == GROUND), dt, 0.0)
            for i, kind, data in evs:
                self._on_event(i, kind, data)
            # --- termination per drone
            self._check_termination()
            self.t = t + dt
            step += 1
            # --- sensing
            if step % sensor_every == 0:
                self._sense()
                if has_observe:
                    self._connected = self.radio.connected_to_base(np.stack([gs.x, gs.y], 1), ~self.down)
                    self.controller.observe({i: self.make_view(i, for_observe=True) for i in range(n)})
            # --- radio
            for (snd, payload, rcp) in self.controller.messages_out():
                self.radio.send(self.t, snd, payload, rcp)
            if self.radio.queue:
                self.radio.deliver(np.stack([gs.x, gs.y], 1), self.t, ~self.down)
                for i in range(n):
                    self.inbox[i] += self.radio.take(i)
            # --- people
            if step % people_every == 0:
                self.people.step(self.t, cfg.people_dt, self.world.flood, self.terrain, L)
            if step % radar_every == 0:
                self._update_rain_radar()
            # --- relaunch after battery swap
            for i in np.flatnonzero(self.swap_until <= self.t):
                self.swap_until[i] = np.inf
                if self.episode_s - self.t >= cfg.relaunch_min_s:
                    gs.E[i] = self.gp.capacity_wh
                    gs.x[i], gs.y[i] = self.base
                    slot = int(sum(1 for j in range(n) if j != i and ap.d[j].launching))
                    ap.launch(int(i), gs, self.t, slot=slot)
                    self.counts["relaunch"] += 1
                    self.reasons[i].add("relaunch")
                    self._log_safety(i, "relaunch")
                else:
                    self._set_down(i, "done")
            # --- decisions
            due = {}
            for i in range(n):
                if self.down[i]:
                    continue
                r = self.reasons[i]
                if gs.airborne[i] and (r or self.t - self.last_decision[i] >= replan):
                    if not r:
                        r.add("timer")
                    due[i] = r
            if due:
                vd = {i: self.make_view(i, sorted(r)) for i, r in due.items()}
                for i in due:
                    self.reasons[i] = set()
                    self.last_decision[i] = self.t
                self._apply_goals(self.controller.decide(vd))
            # --- logging
            if step % traj_every == 0:
                self._log_traj()
            if self.recording is not None and step % grid_every == 0:
                self._record()
            # --- end conditions
            if self.down.all() or self.people.found.all():
                break
        if self.recording is not None:
            self._record()
        # unfound people keep accumulating danger until episode end + 2 h
        tt = self.t
        while tt < self.horizon_s - 1e-9:
            tt = min(tt + cfg.people_dt, self.horizon_s)
            self.people.step(tt, cfg.people_dt, self.world.flood, self.terrain, L)
        from .metrics import compute_metrics
        metrics = compute_metrics(self)
        traj = np.array(self.traj, dtype=np.float64) if self.traj else np.zeros((0, len(TRAJ_FIELDS)))
        finds = [{"person": int(k), "t": float(self.people.t_found[k]), "state": STATE_NAMES[int(self.people.state[k])],
                  "surv": float(self.people.surv_frozen[k]), "at_risk": bool(self.people.ever_affected[k]),
                  "cat": int(self.people.cat[k])}
                 for k in np.flatnonzero(self.people.found)]
        mode_time = {MODE_NAMES[m]: float(self.mode_time[:, m].sum()) for m in range(len(MODE_NAMES))}
        dlog = list(getattr(self.controller, "decision_log", []) or [])
        diag = self.controller.diagnostics() if hasattr(self.controller, "diagnostics") else {}
        return EpisodeResult(
            spec_hash=self.spec.spec_hash, seed=self.spec.seed, controller=getattr(self.controller, "name", "?"),
            controller_params=self.controller.params_dict(), n_drones=n, metrics=metrics, trajectory=traj,
            traj_fields=TRAJ_FIELDS, finds=finds, verification_log=self.verify_log, mode_time=mode_time,
            energy_log={"used_wh": self.energy_used.tolist(), "capacity_wh": self.gp.capacity_wh},
            safety_events=self.safety, detections_log=self.det_log, decision_log=dlog, recording=self.recording,
            wall_s=time.perf_counter() - wall0, planner_diag=diag, events=self.events,
            people_final=[{"id": int(k), "at_risk": bool(self.people.ever_affected[k]),
                           "surv": float(self.people.survival()[k]), "found": bool(self.people.found[k])}
                          for k in range(self.people.n)])

    # ------------------------------------------------------------------ events and checks
    def _log_safety(self, i, kind, **kw):
        self.safety.append({"t": self.t, "drone": int(i), "kind": kind, **kw})

    def _on_event(self, i, kind, data):
        self.events.append({"t": self.t, "drone": int(i), "kind": kind, **({} if not data else
                            {k: v for k, v in data.items() if isinstance(v, (int, float, str))})})
        if kind in ("arrived", "search_done", "thermal_exit", "cannot_progress"):
            self.reasons[i].add(kind)
        elif kind == "verify_done":
            self._verify_result(i, data["xy"])
            self.reasons[i].add("verify_done")
        elif kind == "rtb":
            self.counts["rtb"] += 1
            self.reasons[i].add("rtb")
            self._log_safety(i, "rtb", reason=data.get("reason", ""))
        elif kind == "terrain_warning":
            self._log_safety(i, "terrain_warning", **data)

    def _verify_result(self, i, xy):
        """Truth-side verification: confirm a true person near the cell with p = 0.95; false alarm rejected 0.97."""
        sp = self.world.cfg.sensor
        c = self.terrain.cell_size
        cell = int(xy[1] // c) * self.terrain.n + int(xy[0] // c)
        act = np.flatnonzero(self.people.active())
        person = -1
        if len(act):
            d = np.hypot(self.people.x[act] - xy[0], self.people.y[act] - xy[1])
            k = int(np.argmin(d))
            if d[k] <= sp.verify_match_m:
                person = int(act[k])
        r = self.verify_rng.uniform()
        if person >= 0:
            confirmed = r < sp.verify_confirm_p
            if confirmed:
                self.people.mark_found(person, self.t)
        else:
            confirmed = r >= sp.verify_reject_fa_p
            self.fa_verify_time += self.t - self.verify_started[i]
        rec = (self.t, cell, bool(confirmed))
        self.verifs[i].append(rec)
        self.verif_pending[i].append(rec)
        self.verify_log.append({"t": self.t, "drone": int(i), "cell": cell, "confirmed": bool(confirmed),
                                "true_person": person >= 0, "person": person})
        self.events.append({"t": self.t, "drone": int(i), "kind": "confirm" if confirmed else "reject", "cell": cell,
                            "true": person >= 0})

    def _set_down(self, i, why):
        self.down[i] = True
        self.gs.airborne[i] = False
        self.autopilot.d[i].mode = DOWN
        self._log_safety(i, "down", why=why)

    def _check_termination(self):
        gs, ap = self.gs, self.autopilot
        app = self.ap_params
        L = self.domain
        for i in np.flatnonzero(gs.airborne):
            x, y, z = float(gs.x[i]), float(gs.y[i]), float(gs.z[i])
            d = ap.d[i]
            dbase = math.hypot(x - self.base[0], y - self.base[1])
            outside = max(-x, x - L, -y, y - L, 0.0)
            if outside > app.geofence_limit:
                self.counts["geofence_violation"] += 1
                gs.crashed[i] = True
                self._set_down(i, "geofence")
                continue
            surf = ap.surf_at(min(max(x, 0), L - 1), min(max(y, 0), L - 1))
            if z <= surf:
                if d.mode == RTBM and dbase <= app.land_radius:
                    gs.airborne[i] = False
                    gs.landed[i] = True
                    gs.z[i] = surf
                    d.mode = GROUND
                    d.loiter = False
                    self.counts["landing"] += 1
                    self._log_safety(i, "landed", E=float(gs.E[i]), E_frac=float(gs.E[i] / self.gp.capacity_wh))
                    if self.cfg.swap:
                        self.swap_until[i] = self.t + self.spec.charge_time_min * 60.0
                    else:
                        self._set_down(i, "landed")
                    continue
                gs.crashed[i] = True
                self.counts["crash"] += 1
                self._set_down(i, "crash")
                continue
            if gs.E[i] <= 0.0 and z - ap.terrain_at(x, y) < app.forced_land_agl:
                self.counts["forced_landing"] += 1
                self._set_down(i, "forced_landing")

    def _separation(self, t):
        """Pairwise check (vectorised). Conflict: inside 300 m / 40 m now, or predicted inside 300 m within the
        CPA look-ahead; it lasts until the pair is 50 m apart vertically or 300 m horizontally. The higher-id
        drone climbs or descends away; both turn right."""
        gs = self.gs
        n = self.n
        app = self.ap_params
        sep_dz = np.full(n, np.nan)
        sep_turn = np.zeros(n, dtype=bool)
        vx = gs.x - self._prev_xy[0]
        vy = gs.y - self._prev_xy[1]
        self._prev_xy = (gs.x.copy(), gs.y.copy())
        act = gs.airborne
        if act.sum() < 2:
            self._conflicts = {}
            return sep_dz, sep_turn
        bz = self.autopilot.bz
        dbase = np.hypot(gs.x - self.base[0], gs.y - self.base[1])
        in_zone = (dbase < app.base_zone_r) & (gs.z - bz < app.base_zone_h)
        # avoidance runs everywhere except for drones holding before departure or on final approach;
        # breaches and collisions are only counted outside the base zone
        ap_d = self.autopilot.d
        exempt = np.array([(d.launching and t < d.hold_until) or (d.mode == RTBM and dbase[k] < 300.0)
                           for k, d in enumerate(ap_d)])
        ok = act & ~exempt
        idx = np.flatnonzero(ok)
        if len(idx) < 2:
            self._conflicts = {}
            return sep_dz, sep_turn
        x, y, z = gs.x[idx], gs.y[idx], gs.z[idx]
        iu, ju = np.triu_indices(len(idx), 1)
        rx, ry = x[ju] - x[iu], y[ju] - y[iu]
        dhp = np.hypot(rx, ry)
        dvp = np.abs(z[ju] - z[iu])
        wx, wy = vx[idx][ju] - vx[idx][iu], vy[idx][ju] - vy[idx][iu]
        w2 = wx * wx + wy * wy
        tc = np.clip(-(rx * wx + ry * wy) / np.maximum(w2, 1e-9), 0.0, app.sep_cpa_s)
        dcpa = np.hypot(rx + wx * tc, ry + wy * tc)
        new = ((dhp < app.sep_h) | (dcpa < app.sep_h)) & (dvp < app.sep_v)
        keys = [(int(idx[a]), int(idx[b])) for a, b in zip(iu, ju)]
        ongoing = np.array([k in self._conflicts for k in keys]) & (dhp < app.sep_exit_h)
        conflict = new | ongoing
        closing = dict(zip(keys, (tc > 0) & (dhp < app.sep_h)))
        rel = {}
        for k, c in zip(keys, conflict):
            if not c:
                continue
            rec = self._conflicts.get(k)
            if rec is None:      # latch who is above when the conflict starts
                a, b = k
                rec = (a, b) if gs.z[a] >= gs.z[b] else (b, a)
            rel[k] = rec
        self._conflicts = rel
        sep_zt = np.full(n, np.nan)
        eff = gs.z.copy()
        order = sorted(rel.values(), key=lambda r: gs.z[r[1]])
        for _ in range(3):       # propagate floors up through clusters
            for upper, lower in order:
                fl = eff[lower] + app.sep_dz
                if np.isnan(sep_zt[upper]) or fl > sep_zt[upper]:
                    sep_zt[upper] = fl
                eff[upper] = max(gs.z[upper], sep_zt[upper])
        for k, (upper, lower) in rel.items():
            if closing.get(k, False):
                dz_ = abs(gs.z[upper] - gs.z[lower])
                if dz_ < app.sep_v:
                    sep_turn[upper] = True
                if dz_ < app.breach_v:          # not yet vertically clear: both turn right
                    sep_turn[lower] = True
        sep_dz = sep_zt
        zone_pair = in_zone[idx][iu] | in_zone[idx][ju]
        breach = (dhp < app.breach_h) & (dvp < app.breach_v) & ~zone_pair
        cur = set()
        for k, b_ in zip(keys, breach):
            if not b_:
                continue
            cur.add(k)
            if k not in self.sep_breach_pairs:
                self.counts["sep_breach"] += 1
                ma, mb = self.autopilot.d[k[0]].mode, self.autopilot.d[k[1]].mode
                self._log_safety(k[0], "separation_breach", other=k[1], modes=(MODE_NAMES[ma], MODE_NAMES[mb]),
                                 z=(float(gs.z[k[0]]), float(gs.z[k[1]])))
        self.sep_breach_pairs = cur
        d3 = np.sqrt(dhp ** 2 + dvp ** 2)
        for k, c in zip(keys, (d3 < app.collision_m) & ~zone_pair):
            if not c:
                continue
            for j in k:
                if gs.airborne[j]:
                    gs.crashed[j] = True
                    self._set_down(j, "midair")
            self.counts["midair"] += 1
        return sep_dz, sep_turn

    def _sense(self):
        gs = self.gs
        n = self.n
        people = self.people
        elev = people.elevation(self.terrain)
        for i in range(n):
            if not gs.airborne[i]:
                continue
            fr, truth = self.sensor.frame(i, float(gs.x[i]), float(gs.y[i]), float(gs.z[i]), self.t, people, elev)
            if fr is None:
                continue
            self.counts["frames"] += 1
            self.frames[i].append(fr)
            cells = fr.cell_idx
            self.first_obs[cells] = np.minimum(self.first_obs[cells], self.t)
            rep = (self.last_seen_d[cells] >= 0) & (self.last_seen_d[cells] != i) & (self.t - self.last_seen_t[cells] <= 600.0)
            self.repeat[cells[rep]] = True
            self.last_seen_t[cells] = self.t
            self.last_seen_d[cells] = i
            self.observed[cells] = True
            for cell, pid in truth:
                self.counts["true_det" if pid >= 0 else "false_det"] += 1
                self.det_log.append((self.t, i, int(cell), int(pid)))
            for cell in fr.detections:
                self.det_pending[i].append((self.t, int(cell), fr.seq))
                self.events.append({"t": self.t, "drone": int(i), "kind": "detect", "cell": int(cell)})
            if fr.detections:
                self.reasons[i].add("detection")

    def _log_traj(self):
        gs = self.gs
        modes = self.autopilot.modes()
        for i in range(self.n):
            self.traj.append((self.t, i, gs.x[i], gs.y[i], gs.z[i], gs.V[i], gs.phi[i], gs.E[i], modes[i], self._netto[i]))

    def _record(self):
        """Recording for the visualisation (record_grids=True only)."""
        rec = self.recording
        gs = self.gs
        fl = self.world.flood
        ap = self.autopilot
        drones = []
        for i in range(self.n):
            d = ap.d[i]
            g = d.goal
            goal = None
            if g is not None:
                goal = {"type": g.kind}
                if isinstance(g, Search) and g.box is not None:
                    goal["box"] = [float(b) for b in g.box]
                    goal["h"] = float(g.agl)
                elif isinstance(g, Search):
                    goal["h"] = float(g.agl)
                if isinstance(g, Verify):
                    goal["cell"] = int(g.cell)
            drones.append([float(gs.x[i]), float(gs.y[i]), float(gs.z[i]), float(gs.E[i] / self.gp.capacity_wh),
                           MODE_NAMES[d.mode], goal, float(d.e_return / self.gp.capacity_wh)])
        p = self.people
        rec["frames"].append({
            "t": self.t, "depth": fl.depth_grid(self.t).astype(np.float16),
            "hazard": np.digitize(fl.hazard_grid(self.t), [0.75, 1.25, 2.0]).astype(np.uint8),
            "drones": drones, "seen": self.observed.copy(),
            "people": np.stack([p.x, p.y, p.state, p.found, self.people.survival(), p.ever_affected], 1).astype(np.float32),
        })
        snap = getattr(self.controller, "debug_snapshot", None)
        if callable(snap):
            rec["debug"].append(snap(self.t))


def _terrain_at(ter, x, y):
    c = ter.cell_size
    n = ter.n
    iy = np.clip((np.asarray(y) // c).astype(np.int64), 0, n - 1)
    ix = np.clip((np.asarray(x) // c).astype(np.int64), 0, n - 1)
    return ter.z[iy, ix]
