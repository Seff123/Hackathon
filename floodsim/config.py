"""All configuration dataclasses. Every number in the build plan lives here, never inline."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import tomllib
import warnings
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

G = 9.81


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Bounds:
    lo: float
    hi: float
    dist: str = "uniform"  # uniform | loguniform | int | normal_clipped

    def sample(self, rng) -> float:
        if self.dist == "uniform":
            return float(rng.uniform(self.lo, self.hi))
        if self.dist == "loguniform":
            return float(math.exp(rng.uniform(math.log(self.lo), math.log(self.hi))))
        if self.dist == "int":
            return int(rng.integers(int(self.lo), int(self.hi) + 1))
        if self.dist == "normal_clipped":
            mu, sd = 0.5 * (self.lo + self.hi), (self.hi - self.lo) / 4
            return float(min(max(rng.normal(mu, sd), self.lo), self.hi))
        raise ValueError(f"unknown dist {self.dist}")

    def contains(self, v: float) -> bool:
        return self.lo - 1e-9 <= v <= self.hi + 1e-9


def B(lo, hi, dist="uniform"):
    return field(default_factory=lambda: Bounds(lo, hi, dist))


# ---------------------------------------------------------------------------
# Team inputs
# ---------------------------------------------------------------------------
@dataclass
class TeamInputs:
    mass_kg: float = 5.0
    battery_life_min: float = 150.0
    battery_wh: float | None = None
    charge_time_min: float = 10.0
    signal_radius_km: float = 3.0
    camera_range_m: float = 140.0

    def validate(self):
        if not 1.0 <= self.mass_kg <= 25.0:
            raise ValueError(f"mass_kg {self.mass_kg} outside 1-25 kg")
        if self.battery_wh is None and not 20.0 <= self.battery_life_min <= 600.0:
            raise ValueError(f"battery_life_min {self.battery_life_min} outside 20-600 min")
        if self.camera_range_m <= 0 or self.signal_radius_km <= 0 or self.charge_time_min < 0:
            raise ValueError("camera_range_m, signal_radius_km must be > 0, charge_time_min >= 0")


# ---------------------------------------------------------------------------
# Scenario bounds and spec
# ---------------------------------------------------------------------------
FLOOD_TYPES = ("fluvial", "fluvial_pluvial", "pluvial", "breach")
OUTLET_EDGES = ("N", "E", "S", "W")


@dataclass
class ScenarioBounds:
    # terrain
    relief_amp: Bounds = B(20, 300, "loguniform")
    beta: Bounds = B(1.8, 2.4)
    regional_slope: Bounds = B(0.001, 0.008)
    valley_width: Bounds = B(300, 2000)
    valley_depth: Bounds = B(10, 60)
    sinuosity: Bounds = B(1.1, 1.8)
    channel_width: Bounds = B(20, 80)
    channel_depth: Bounds = B(2, 5)
    n_towns: Bounds = B(1, 3, "int")
    town_radius: Bounds = B(300, 1200)
    levee_height: Bounds = B(0, 3)
    levee_height_breach: Bounds = B(1, 3)
    forest_fraction: Bounds = B(0.05, 0.4)
    # flood
    flood_type_weights: dict = field(default_factory=lambda: {
        "fluvial": 0.45, "fluvial_pluvial": 0.25, "pluvial": 0.2, "breach": 0.1})
    q_peak: Bounds = B(100, 1500, "loguniform")
    base_frac: Bounds = B(0.05, 0.15)
    t_peak_h: Bounds = B(1, 6)
    hydro_k: Bounds = B(2, 5)
    start_offset_h: Bounds = B(-0.5, 4)
    rain_peak_mmh: Bounds = B(10, 60)
    climate_factor: Bounds = B(1.0, 1.4)
    storm_duration_h: Bounds = B(0.5, 4)
    storm_radius_km: Bounds = B(2, 8)
    storm_dir_offset_deg: Bounds = B(-30, 30)
    n_channel: Bounds = B(0.03, 0.04)
    n_floodplain: Bounds = B(0.04, 0.07)
    n_urban: Bounds = B(0.06, 0.12)
    n_forest: Bounds = B(0.08, 0.15)
    breach_time_h: Bounds = B(0, 2)
    breach_width: Bounds = B(20, 100)
    breach_drop: Bounds = B(1, 4)
    fc_sigma0: Bounds = B(0.1, 0.3)
    fc_growth: Bounds = B(0.1, 0.4)
    fc_bias: Bounds = B(-0.3, 0.3)
    # weather
    wind10: Bounds = B(1, 8)
    wind_from_deg: Bounds = B(0, 360)
    cloud_cover: Bounds = B(0.3, 1.0)
    w_star_raw: Bounds = B(0, 2.5)
    w_star_gain: float = 1.4
    zi: Bounds = B(400, 1500)
    thermal_density_km2: Bounds = B(0, 0.3)
    start_hour: Bounds = B(7, 15)
    ridge_k: Bounds = B(0.8, 1.2)
    # people
    n_targets: Bounds = B(3, 25, "int")
    category_mean: tuple = (0.55, 0.2, 0.25)
    category_concentration: float = 20.0
    immobile_frac: Bounds = B(0.2, 0.5)
    walk_speed: Bounds = B(0.5, 1.3)
    drift_frac: Bounds = B(0.3, 0.8)
    # swarm
    n_drones: Bounds = B(2, 8, "int")
    # rejection
    max_attempts: int = 50
    extent_frac_min: float = 0.03
    extent_frac_max: float = 0.40
    targets_near_flood_frac: float = 0.6
    targets_near_flood_m: float = 1000.0
    max_terrain_above_base_m: float = 300.0
    max_wind_300m: float = 15.0
    min_drain_to_outlet: float = 0.8


@dataclass
class ScenarioSpec:
    seed: int
    attempt: int
    # terrain
    relief_amp: float
    beta: float
    regional_slope: float
    outlet_edge: str
    valley_width: float
    valley_depth: float
    sinuosity: float
    channel_width: float
    channel_depth: float
    n_towns: int
    town_radius: list
    levee_height: float
    forest_fraction: float
    # flood
    flood_type: str
    q_peak: float
    base_frac: float
    t_peak_h: float
    hydro_k: float
    start_offset_h: float
    rain_peak_mmh: float
    climate_factor: float
    storm_duration_h: float
    storm_radius_km: float
    storm_dir_offset_deg: float
    n_channel: float
    n_floodplain: float
    n_urban: float
    n_forest: float
    breach_time_h: float
    breach_width: float
    breach_drop: float
    fc_sigma0: float
    fc_growth: float
    fc_bias: float
    # weather
    wind10: float
    wind_from_deg: float
    cloud_cover: float
    w_star: float
    zi: float
    thermal_density_km2: float
    start_hour: float
    ridge_k: float
    # people
    n_targets: int
    category_weights: list
    immobile_frac: float
    walk_speed: float
    drift_frac: float
    # swarm
    n_drones: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @property
    def spec_hash(self) -> str:
        return hashlib.sha1(self.to_json().encode()).hexdigest()

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioSpec":
        return cls(**{f.name: d[f.name] for f in fields(cls)})

    @property
    def wind_dir_to(self) -> float:
        """Direction the wind blows towards, radians anticlockwise from east."""
        return math.radians(270.0 - self.wind_from_deg)

    @property
    def has_river_flood(self) -> bool:
        return self.flood_type in ("fluvial", "fluvial_pluvial", "breach")

    @property
    def has_rain(self) -> bool:
        return self.flood_type in ("fluvial_pluvial", "pluvial")


# ---------------------------------------------------------------------------
# World build settings
# ---------------------------------------------------------------------------
@dataclass
class WorldConfig:
    domain_m: float = 10_000.0
    cell_m: float = 50.0
    flood_cell_m: float = 100.0
    episode_h: float = 2.5
    post_episode_h: float = 2.0
    spinup_s: float = 1800.0
    depth_snapshot_s: float = 60.0
    vel_snapshot_s: float = 300.0
    # flood solver
    cfl: float = 0.7
    dt_min: float = 0.5
    dt_max: float = 10.0
    wet_face_m: float = 1e-3
    vel_min_depth_m: float = 0.05
    flood_depth_m: float = 0.1  # "wet" for extent / affected
    runoff_urban: float = 0.9
    runoff_open: float = 0.5
    runoff_forest: float = 0.3
    # terrain
    drainage_area_km2: float = 1.0
    levee_town_dist_m: float = 1000.0
    town_river_dist_m: float = 1500.0
    base_town_dist_m: float = 2000.0
    forest_h: tuple = (15.0, 25.0)
    urban_h: tuple = (6.0, 20.0)
    floodplain_noise_m: tuple = (0.5, 2.0)
    # forecast
    fc_extent_period_s: float = 600.0
    fc_extent_accuracy: float = 0.95
    fc_issue_period_s: float = 300.0
    fc_lead_step_s: float = 300.0
    fc_lead_max_s: float = 3600.0
    fc_corr_len_m: float = 2000.0
    fc_vel_noise: float = 0.3
    fc_vel_period_s: float = 600.0
    fc_observed_wet_p: float = 0.95
    # weather
    z0: float = 0.1
    profile_top_m: float = 600.0
    ridge_smooth_m: float = 150.0
    ridge_relief_radius_m: float = 2000.0
    ridge_min_h: float = 50.0
    ridge_clip: float = -2.0
    thermal_core_mult: tuple = (0.8, 1.5)
    thermal_radius_frac: tuple = (0.1, 0.2)
    thermal_radius_clip: tuple = (60.0, 200.0)
    thermal_weights: dict = field(default_factory=lambda: {"urban": 1.5, "open": 1.0, "forest": 0.5, "water": 0.0})
    rho: float = 1.225

    @property
    def n(self) -> int:
        return int(round(self.domain_m / self.cell_m))

    @property
    def nf(self) -> int:
        return int(round(self.domain_m / self.flood_cell_m))


# ---------------------------------------------------------------------------
# Glider
# ---------------------------------------------------------------------------
@dataclass
class GliderParams:
    mass: float = 5.0
    wing_area: float = 0.90
    span: float = 3.67
    cd0: float = 0.018
    oswald: float = 0.85
    cl_max: float = 1.3
    v_min: float = 9.0
    v_max: float = 30.0
    bank_max_deg: float = 45.0
    bank_safety_deg: float = 60.0
    thrust_max: float = 14.0
    eta: float = 0.65
    payload_w: float = 25.0
    battery_life_min: float = 150.0
    battery_wh: float | None = None
    reserve_frac: float = 0.2
    v_ref: float = 13.0
    climb_speed: float = 12.0
    rho: float = 1.225
    table_v_step: float = 0.5
    table_bank_step_deg: float = 5.0

    @property
    def aspect_ratio(self) -> float:
        return self.span ** 2 / self.wing_area

    @property
    def k(self) -> float:
        return 1.0 / (math.pi * self.oswald * self.aspect_ratio)

    def drag(self, v, bank=0.0):
        import numpy as np
        v = np.asarray(v, dtype=float)
        q = 0.5 * self.rho * v ** 2 * self.wing_area
        cl = self.mass * G / (q * np.cos(bank))
        return q * (self.cd0 + self.k * cl ** 2)

    def stall_speed(self, bank=0.0) -> float:
        import numpy as np
        return np.sqrt(2 * self.mass * G / (self.rho * self.wing_area * self.cl_max * np.cos(bank)))

    def level_power_w(self, v) -> float:
        return self.drag(v) * v / self.eta + self.payload_w

    @property
    def capacity_wh(self) -> float:
        if self.battery_wh is not None:
            return float(self.battery_wh)
        return float(self.battery_life_min / 60.0 * self.level_power_w(self.v_ref))


# ---------------------------------------------------------------------------
# Sensor
# ---------------------------------------------------------------------------
@dataclass
class SensorParams:
    fov_deg: float = 60.0
    camera_range_m: float = 140.0
    d_ref: float = 0.5
    d_land: float = 0.5
    d_water: float = 0.3
    p_max: float = 0.95
    e0: float = 2.7
    e1: float = 0.7
    lambda_fa: float = 0.002
    frame_dt: float = 2.0
    skip_above_range_mult: float = 2.0
    confirm_p: float = 0.95
    fa_reject_p: float = 0.97
    cover_mod: dict = field(default_factory=lambda: {"open": 1.0, "water": 1.0, "urban": 0.6, "forest": 0.2})


# ---------------------------------------------------------------------------
# Autopilot
# ---------------------------------------------------------------------------
@dataclass
class AutopilotParams:
    transit_speed: float = 15.0
    cruise_cap: float = 25.0
    sink_extra_speed: float = 2.0
    sink_netto: float = -1.0
    fast_descent_speed: float = 18.0
    arrive_radius: float = 150.0
    band_hysteresis: float = 50.0
    transit_band_base: float = 150.0   # height above surface of the lowest transit layer
    transit_band_width: float = 100.0
    transit_layer_sep: float = 50.0
    transit_layer_cycle: int = 5
    heading_gain: float = 0.5
    bank_cmd_max_deg: float = 35.0
    search_bank_max_deg: float = 25.0
    turn_bank_deg: float = 35.0
    alt_gain: float = 0.02
    cannot_progress_gs: float = 1.0
    # thermal
    thermal_bank_deg: float = 35.0
    thermal_speed_mult: float = 1.2
    thermal_confirm_s: float = 4.0
    thermal_exit_climb: float = 0.3
    thermal_max_s: float = 900.0
    thermal_zi_margin: float = 50.0
    thermal_ceiling_margin: float = 100.0
    # verify
    verify_agl: float = 70.0
    verify_radius: float = 80.0
    verify_time: float = 60.0
    # RTB
    rtb_power_factor: float = 1.3
    rtb_min_gs: float = 2.0
    glide_margin: float = 0.7
    land_radius: float = 100.0
    land_orbit_radius: float = 80.0
    land_exempt_radius: float = 150.0
    land_speed: float = 12.0
    base_zone_radius: float = 500.0
    base_zone_height: float = 150.0
    # safety
    floor_agl: float = 30.0
    search_min_agl: float = 60.0
    lookahead_s: float = 10.0
    escape_bank_deg: float = 60.0
    geofence_margin: float = 300.0
    geofence_violation: float = 500.0
    stall_margin: float = 1.15
    ceiling_min: float = 400.0
    forced_landing_agl: float = 30.0
    # separation
    sep_h: float = 300.0
    sep_v: float = 40.0
    sep_climb: float = 50.0
    sep_turn_deg: float = 30.0
    breach_h: float = 100.0
    breach_v: float = 25.0
    collision_m: float = 10.0
    thermal_occupied_m: float = 300.0
    fixed_rtb_frac: float | None = None  # fallback: fixed RTB trigger


# ---------------------------------------------------------------------------
# Controller / swarm / sim
# ---------------------------------------------------------------------------
@dataclass
class ControllerParams:
    sweep_hand_max: float = 10.0
    search_agl_frac: float = 0.6
    search_min_agl: float = 60.0
    search_speed_min: float = 13.0
    search_speed_margin: float = 4.0
    search_speed_max: float = 25.0
    overlap: float = 0.15
    leg_angle_deg: float = 90.0
    soar_battery: float = 0.6
    w_enter: float = 0.8
    replan_s: float = 60.0
    allow_soaring: bool = True


@dataclass
class SwarmParams:
    signal_radius_km: float = 3.0
    perfect_comms: bool = False
    n_drones: int | None = None  # override the sampled swarm size


@dataclass
class SimConfig:
    fast: bool = False
    dt: float = 1.0
    sensor_dt: float = 2.0
    radio_dt: float = 1.0
    people_dt: float = 10.0
    log_dt: float = 2.0
    log_dt_fast: float = 10.0
    record_grids: bool = False
    grid_dt: float = 30.0
    swap_enabled: bool = True
    charge_time_min: float = 10.0
    relaunch_min_remaining_s: float = 1200.0
    t_crit_s: float = 2400.0
    hr_exposed: float = 0.75
    hr_sheltered: float = 2.0
    netto_noise: float = 0.1
    wind_est_noise: float = 1.0
    rain_radar_noise: float = 0.2
    coverage_window_s: float = 1800.0
    repeat_window_s: float = 600.0


@dataclass
class PeopleParams:
    roof_depth: float = 0.5
    walk_trigger_depth: float = 0.1
    walk_trigger_dist: float = 200.0
    in_water_depth: float = 0.5
    leave_water_depth: float = 0.3
    safe_hand_gain: float = 3.0
    home_hand_boost: float = 3.0
    home_hand_max: float = 5.0
    vehicle_hand_max: float = 3.0
    outdoor_river_m: float = 2000.0
    affected_depth: float = 0.1
    det_factor: dict = field(default_factory=lambda: {
        "HOME_INSIDE": 0.02, "ROOF": 1.0, "VEHICLE": 0.2, "OUTDOOR": 1.0,
        "SAFE_HIGH": 1.0, "IN_WATER": 0.8, "LOST": 0.0})


@dataclass
class Config:
    team: TeamInputs = field(default_factory=TeamInputs)
    bounds: ScenarioBounds = field(default_factory=ScenarioBounds)
    world: WorldConfig = field(default_factory=WorldConfig)
    glider: GliderParams = field(default_factory=GliderParams)
    sensor: SensorParams = field(default_factory=SensorParams)
    autopilot: AutopilotParams = field(default_factory=AutopilotParams)
    controller: ControllerParams = field(default_factory=ControllerParams)
    swarm: SwarmParams = field(default_factory=SwarmParams)
    sim: SimConfig = field(default_factory=SimConfig)
    people: PeopleParams = field(default_factory=PeopleParams)

    def apply_team(self) -> "Config":
        """Push team inputs into the parameter blocks that use them."""
        t = self.team
        t.validate()
        self.glider.mass = t.mass_kg
        self.glider.battery_life_min = t.battery_life_min
        self.glider.battery_wh = t.battery_wh
        self.sim.charge_time_min = t.charge_time_min
        self.swarm.signal_radius_km = t.signal_radius_km
        self.sensor.camera_range_m = t.camera_range_m
        if self.glider.stall_speed() > 12.0:
            warnings.warn(f"level stall speed {self.glider.stall_speed():.1f} m/s exceeds 12 m/s")
        return self

    @property
    def ceiling_agl(self) -> float:
        return max(self.autopilot.ceiling_min, 2.0 * self.sensor.camera_range_m)


def _update_dc(obj, d: dict):
    for k, v in d.items():
        if not hasattr(obj, k):
            raise KeyError(f"unknown config key {type(obj).__name__}.{k}")
        cur = getattr(obj, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _update_dc(cur, v)
        elif isinstance(cur, Bounds) and isinstance(v, (list, tuple)):
            setattr(obj, k, Bounds(v[0], v[1], v[2] if len(v) > 2 else cur.dist))
        elif isinstance(cur, tuple) and isinstance(v, list):
            setattr(obj, k, tuple(v))
        else:
            setattr(obj, k, v)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    cfg = Config()
    if path is not None and Path(path).exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)
        for section, vals in data.items():
            _update_dc(getattr(cfg, section), vals)
    for k, v in (overrides or {}).items():
        if v is not None:
            section, key = k.split(".")
            setattr(getattr(cfg, section), key, v)
    return cfg.apply_team()
