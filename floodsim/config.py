"""Every number in the plan lives here as a dataclass default.

Sections map 1:1 to TOML tables in ``configs/default.toml``: ``[team]``,
``[world]``, ``[glider]``, ``[autopilot]`` and so on. Scenario bounds live in
``[bounds]`` with entries like ``relief_m = [20, 300, "loguniform"]``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import tomllib
import warnings
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Bounds and sampling
# --------------------------------------------------------------------------

DISTS = ("uniform", "loguniform", "int", "normal_clipped", "fixed")


@dataclass(frozen=True)
class Bounds:
    lo: float
    hi: float
    dist: str = "uniform"

    def __post_init__(self):
        if self.dist not in DISTS:
            raise ValueError(f"unknown distribution {self.dist!r}")
        if self.hi < self.lo:
            raise ValueError(f"Bounds hi < lo: {self}")

    def sample(self, rng: np.random.Generator) -> float:
        lo, hi = self.lo, self.hi
        if self.dist == "fixed" or lo == hi:
            return int(lo) if self.dist == "int" else float(lo)
        if self.dist == "uniform":
            return float(rng.uniform(lo, hi))
        if self.dist == "loguniform":
            return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        if self.dist == "int":
            return int(rng.integers(int(lo), int(hi) + 1))
        if self.dist == "normal_clipped":
            mu, sd = 0.5 * (lo + hi), (hi - lo) / 4.0
            return float(np.clip(rng.normal(mu, sd), lo, hi))
        raise AssertionError

    def contains(self, v: float) -> bool:
        return self.lo - 1e-9 <= v <= self.hi + 1e-9


def B(lo, hi, dist="uniform") -> Any:
    return field(default_factory=lambda: Bounds(lo, hi, dist))


def _weights(d: dict[str, float]):
    return field(default_factory=lambda: dict(d))


def sample_categorical(rng: np.random.Generator, weights: dict[str, float]) -> str:
    keys = list(weights)
    p = np.array([weights[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


FLOOD_TYPES = ("fluvial", "fluvial_pluvial", "pluvial", "breach")
OUTLET_EDGES = ("N", "E", "S", "W")


@dataclass
class ScenarioBounds:
    # terrain
    relief_m: Bounds = B(20, 300, "loguniform")
    spectral_beta: Bounds = B(1.8, 2.4)
    regional_slope: Bounds = B(0.001, 0.008)
    outlet_edge: dict = _weights({e: 1.0 for e in OUTLET_EDGES})
    valley_width_m: Bounds = B(300, 2000)
    valley_depth_m: Bounds = B(10, 60)
    sinuosity: Bounds = B(1.1, 1.8)
    channel_width_m: Bounds = B(20, 80)
    channel_depth_m: Bounds = B(2, 5)
    n_towns: Bounds = B(1, 3, "int")
    town_radius_m: Bounds = B(300, 1200)
    levee_height_m: Bounds = B(0, 3)
    forest_fraction: Bounds = B(0.05, 0.4)
    # flood
    flood_type: dict = _weights(
        {"fluvial": 0.45, "fluvial_pluvial": 0.25, "pluvial": 0.2, "breach": 0.1}
    )
    peak_inflow_m3s: Bounds = B(100, 1500, "loguniform")
    base_flow_frac: Bounds = B(0.05, 0.15)
    time_to_peak_h: Bounds = B(1, 6)
    hydro_shape_k: Bounds = B(2, 5)
    start_offset_h: Bounds = B(-0.5, 4.0)
    rain_peak_mmh: Bounds = B(10, 60)
    climate_factor: Bounds = B(1.0, 1.4)
    storm_duration_h: Bounds = B(0.5, 4)
    storm_radius_km: Bounds = B(2, 8)
    n_channel: Bounds = B(0.03, 0.04)
    n_floodplain: Bounds = B(0.04, 0.07)
    n_urban: Bounds = B(0.06, 0.12)
    n_forest: Bounds = B(0.08, 0.15)
    breach_time_h: Bounds = B(0, 2)
    breach_width_m: Bounds = B(20, 100)
    breach_drop_m: Bounds = B(1, 4)
    fc_sigma0_m: Bounds = B(0.1, 0.3)
    fc_growth_mph: Bounds = B(0.1, 0.4)
    fc_bias_m: Bounds = B(-0.3, 0.3)
    # weather
    wind10_ms: Bounds = B(1, 8)
    wind_from_deg: Bounds = B(0, 360)
    cloud_cover: Bounds = B(0.3, 1.0)
    w_star_base: Bounds = B(0, 2.5)  # x (1 - cloud) x 1.4
    zi_m: Bounds = B(400, 1500)
    thermal_density_km2: Bounds = B(0, 0.3)
    start_hour: Bounds = B(7, 15)
    k_ridge: Bounds = B(0.8, 1.2)
    storm_track_offset_deg: Bounds = B(-30, 30)
    # people
    n_targets: Bounds = B(3, 25, "int")
    category_means: tuple = (0.55, 0.20, 0.25)  # home, vehicle, outdoor
    category_concentration: float = 20.0
    immobile_frac: Bounds = B(0.2, 0.5)
    walk_speed_ms: Bounds = B(0.5, 1.3)
    drift_frac: Bounds = B(0.3, 0.8)
    # swarm / episode
    n_drones: Bounds = B(2, 8, "int")
    episode_h: Bounds = B(2.5, 2.5, "fixed")


@dataclass
class ScenarioSpec:
    """Flat, JSON-serialisable description of one scenario."""

    seed: int
    attempt: int
    # terrain
    relief_m: float
    spectral_beta: float
    regional_slope: float
    outlet_edge: str
    valley_width_m: float
    valley_depth_m: float
    sinuosity: float
    channel_width_m: float
    channel_depth_m: float
    n_towns: int
    town_radii_m: tuple
    levee_height_m: float
    forest_fraction: float
    # flood
    flood_type: str
    peak_inflow_m3s: float
    base_flow_frac: float
    time_to_peak_h: float
    hydro_shape_k: float
    start_offset_h: float
    rain_peak_mmh: float
    climate_factor: float
    storm_duration_h: float
    storm_radius_km: float
    n_channel: float
    n_floodplain: float
    n_urban: float
    n_forest: float
    breach_time_h: float
    breach_width_m: float
    breach_drop_m: float
    fc_sigma0_m: float
    fc_growth_mph: float
    fc_bias_m: float
    # weather
    wind10_ms: float
    wind_from_deg: float
    cloud_cover: float
    w_star_ms: float
    zi_m: float
    thermal_density_km2: float
    start_hour: float
    k_ridge: float
    storm_track_offset_deg: float
    # people
    n_targets: int
    category_weights: tuple
    immobile_frac: float
    walk_speed_ms: float
    drift_frac: float
    # swarm / episode
    n_drones: int
    episode_h: float

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @property
    def spec_hash(self) -> str:
        return hashlib.sha1(self.to_json().encode()).hexdigest()

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioSpec":
        d = dict(d)
        for k in ("town_radii_m", "category_weights"):
            d[k] = tuple(d[k])
        return cls(**d)

    # convenience
    @property
    def wind_to_rad(self) -> float:
        """Direction the wind blows towards, radians anticlockwise from east."""
        return math.radians(270.0 - self.wind_from_deg)

    @property
    def episode_s(self) -> float:
        return self.episode_h * 3600.0

    @property
    def has_rain(self) -> bool:
        return self.flood_type in ("fluvial_pluvial", "pluvial")

    @property
    def has_river_peak(self) -> bool:
        return self.flood_type != "pluvial"

    @property
    def has_breach(self) -> bool:
        return self.flood_type == "breach"


def sample_raw_spec(seed: int, bounds: ScenarioBounds, attempt: int = 0) -> ScenarioSpec:
    """Sample every field from its bounds (no rejection checks)."""
    from .rng import stream

    rng = stream(seed, "spec", attempt)
    b = bounds
    s: dict[str, Any] = {"seed": int(seed), "attempt": int(attempt)}
    for name in (
        "relief_m", "spectral_beta", "regional_slope", "valley_width_m", "valley_depth_m",
        "sinuosity", "channel_width_m", "channel_depth_m",
    ):
        s[name] = getattr(b, name).sample(rng)
    s["outlet_edge"] = sample_categorical(rng, b.outlet_edge)
    s["n_towns"] = int(b.n_towns.sample(rng))
    s["town_radii_m"] = tuple(float(b.town_radius_m.sample(rng)) for _ in range(s["n_towns"]))
    s["levee_height_m"] = b.levee_height_m.sample(rng)
    s["forest_fraction"] = b.forest_fraction.sample(rng)
    # flood
    s["flood_type"] = sample_categorical(rng, b.flood_type)
    for name in (
        "peak_inflow_m3s", "base_flow_frac", "time_to_peak_h", "hydro_shape_k", "start_offset_h",
        "climate_factor", "storm_duration_h", "storm_radius_km", "n_channel", "n_floodplain",
        "n_urban", "n_forest", "breach_time_h", "breach_width_m", "breach_drop_m",
        "fc_sigma0_m", "fc_growth_mph", "fc_bias_m",
    ):
        s[name] = getattr(b, name).sample(rng)
    if s["flood_type"] == "breach":
        # a breach needs a levee to cut: never let the levee height be ~0
        s["levee_height_m"] = max(s["levee_height_m"], 1.0)
    # weather (cloud cover first; rain up and thermals down with it)
    s["cloud_cover"] = b.cloud_cover.sample(rng)
    c_n = (s["cloud_cover"] - b.cloud_cover.lo) / max(b.cloud_cover.hi - b.cloud_cover.lo, 1e-9)
    u = rng.uniform()
    rain_lo, rain_hi = b.rain_peak_mmh.lo, b.rain_peak_mmh.hi
    s["rain_peak_mmh"] = (rain_lo + (rain_hi - rain_lo) * (0.5 * u + 0.5 * c_n)) * s["climate_factor"]
    s["w_star_ms"] = b.w_star_base.sample(rng) * (1.0 - s["cloud_cover"]) * 1.4
    for name in ("wind10_ms", "wind_from_deg", "zi_m", "thermal_density_km2", "start_hour",
                 "k_ridge", "storm_track_offset_deg"):
        s[name] = getattr(b, name).sample(rng)
    # people
    s["n_targets"] = int(b.n_targets.sample(rng))
    alpha = np.asarray(b.category_means, float) * b.category_concentration
    s["category_weights"] = tuple(float(x) for x in rng.dirichlet(alpha))
    for name in ("immobile_frac", "walk_speed_ms", "drift_frac"):
        s[name] = getattr(b, name).sample(rng)
    s["n_drones"] = int(b.n_drones.sample(rng))
    s["episode_h"] = float(b.episode_h.sample(rng))
    return ScenarioSpec(**s)


# --------------------------------------------------------------------------
# Team inputs and physics/agent parameters
# --------------------------------------------------------------------------


@dataclass
class TeamInputs:
    mass_kg: float = 5.0
    battery_life_min: float = 150.0
    battery_wh: float | None = None  # overrides battery_life_min when set
    charge_time_min: float = 10.0
    signal_radius_km: float = 3.0
    camera_range_m: float = 140.0

    def validate(self) -> "TeamInputs":
        if not 1.0 <= self.mass_kg <= 25.0:
            raise ValueError(f"mass_kg must be 1-25 kg, got {self.mass_kg}")
        if self.battery_wh is None and not 20.0 <= self.battery_life_min <= 600.0:
            raise ValueError(f"battery_life_min must be 20-600 min, got {self.battery_life_min}")
        if self.battery_wh is not None and self.battery_wh <= 0:
            raise ValueError("battery_wh must be positive")
        if self.charge_time_min < 0:
            raise ValueError("charge_time_min must be >= 0")
        if self.signal_radius_km <= 0:
            raise ValueError("signal_radius_km must be positive")
        if not 10.0 <= self.camera_range_m <= 1000.0:
            raise ValueError("camera_range_m must be 10-1000 m")
        return self

    @property
    def ceiling_agl_m(self) -> float:
        return max(400.0, 2.0 * self.camera_range_m)


@dataclass
class WorldConfig:
    domain_m: float = 10_000.0
    cell_m: float = 50.0
    flood_cell_m: float = 100.0
    # terrain
    edge_rim_frac: float = 0.35  # rim raised on non-outlet edges, x relief
    edge_rim_width_m: float = 2500.0
    floodplain_noise_m: tuple = (0.5, 2.0)
    floodplain_frac: float = 0.6  # inner share of the valley half-width blended flat
    centreline_noise_m: float = 0.3
    min_river_slope: float = 0.0005
    max_pond_depth_m: float = 0.5  # closed depressions deeper than this are partly filled
    prefill_eps_m: float = 0.025  # per-cell gradient given to filled basins (slope 0.0005)
    river_start_jitter: float = 0.3
    river_margin_m: float = 400.0
    drainage_area_km2: float = 1.0
    levee_town_dist_m: float = 1000.0
    town_river_max_m: float = 1500.0
    town_min_sep_m: float = 2000.0
    forest_h_m: tuple = (15.0, 25.0)
    urban_h_m: tuple = (6.0, 20.0)
    base_town_dist_m: float = 2000.0
    base_clear_radius_m: float = 150.0
    roads_per_town: int = 2
    road_len_m: tuple = (2000.0, 4000.0)
    # flood solver
    cfl: float = 0.7
    dt_min_s: float = 0.5
    dt_max_s: float = 10.0
    wet_tol_m: float = 1e-3
    vel_min_depth_m: float = 0.05
    froude_limit: float = 1.0
    spinup_s: float = 1800.0
    post_episode_s: float = 7200.0
    depth_snapshot_s: float = 60.0
    velocity_snapshot_s: float = 300.0
    inflow_spread_cells: int = 3
    runoff_coef: dict = _weights({"urban": 0.9, "open": 0.5, "forest": 0.3, "water": 1.0})
    wet_depth_m: float = 0.1  # "flooded" threshold used for extents and forecasts

    @property
    def n(self) -> int:
        return int(round(self.domain_m / self.cell_m))

    @property
    def nf(self) -> int:
        return int(round(self.domain_m / self.flood_cell_m))


@dataclass
class ForecastParams:
    obs_dt_s: float = 600.0
    p_obs_correct: float = 0.95
    issue_dt_s: float = 300.0
    max_lead_s: float = 3600.0
    lead_step_s: float = 300.0
    corr_length_m: float = 2000.0
    velocity_noise: float = 0.3
    velocity_dt_s: float = 600.0
    observed_wet_floor: float = 0.95
    rain_noise: float = 0.2
    pluvial_accum_cells: float = 25.0
    pluvial_min_depression_m: float = 0.1
    floodplain_hand_m: float = 10.0  # Brier is scored over this zone


@dataclass
class WeatherParams:
    z0_m: float = 0.1
    profile_top_m: float = 600.0
    ridge_smooth_m: float = 150.0
    relief_window_m: float = 2000.0
    ridge_h_factor: float = 0.8
    ridge_h_min_m: float = 50.0
    ridge_sink_clip: float = -2.0
    thermal_core_factor: tuple = (0.8, 1.5)
    thermal_radius_frac: tuple = (0.1, 0.2)
    thermal_radius_clamp: tuple = (60.0, 200.0)
    thermal_landcover_weight: dict = _weights({"urban": 1.5, "open": 1.0, "forest": 0.5, "water": 0.0})
    rho: float = 1.225
    storm_steer_height_m: float = 600.0


@dataclass
class GliderParams:
    mass_kg: float = 5.0
    wing_area_m2: float = 0.90
    span_m: float = 3.67
    cd0: float = 0.018
    oswald: float = 0.85
    cl_max: float = 1.3
    v_min: float = 9.0
    v_max: float = 30.0
    bank_limit_deg: float = 45.0
    bank_limit_safety_deg: float = 60.0
    thrust_max_n: float = 14.0
    prop_eff: float = 0.65
    payload_w: float = 25.0
    battery_life_min: float = 150.0
    battery_wh: float | None = None
    reserve_frac: float = 0.20
    v_ref: float = 13.0
    climb_speed: float = 12.0
    rho: float = 1.225
    g: float = 9.81
    table_v_step: float = 0.5
    table_bank_step_deg: float = 5.0
    table_bank_max_deg: float = 60.0

    @classmethod
    def from_team(cls, team: TeamInputs, **kw) -> "GliderParams":
        return cls(mass_kg=team.mass_kg, battery_life_min=team.battery_life_min,
                   battery_wh=team.battery_wh, **kw)

    @property
    def aspect_ratio(self) -> float:
        return self.span_m**2 / self.wing_area_m2

    @property
    def k_induced(self) -> float:
        return 1.0 / (math.pi * self.oswald * self.aspect_ratio)

    @property
    def weight_n(self) -> float:
        return self.mass_kg * self.g

    def drag(self, v, bank_rad=0.0):
        v = np.asarray(v, float)
        q = 0.5 * self.rho * v**2 * self.wing_area_m2
        cl = self.weight_n / (q * np.cos(bank_rad))
        return q * (self.cd0 + self.k_induced * cl**2)

    def stall_speed(self, bank_rad=0.0):
        return np.sqrt(2 * self.weight_n / (self.rho * self.wing_area_m2 * self.cl_max * np.cos(bank_rad)))

    @property
    def battery_capacity_wh(self) -> float:
        if self.battery_wh is not None:
            return float(self.battery_wh)
        p = float(self.drag(self.v_ref)) * self.v_ref / self.prop_eff + self.payload_w
        return self.battery_life_min / 60.0 * p


@dataclass
class AutopilotParams:
    transit_speed: float = 15.0
    cruise_cap: float = 25.0
    sink_speed_add: float = 2.0
    sink_netto: float = -1.0
    band_fast_speed: float = 18.0
    band_hysteresis_m: float = 50.0
    transit_arrive_m: float = 150.0
    heading_gain: float = 0.5
    bank_cmd_limit_deg: float = 35.0
    search_bank_deg: float = 25.0
    turn_bank_deg: float = 35.0
    alt_gain: float = 0.02
    xtrack_gain: float = 0.01  # rad per metre of cross-track error
    xtrack_max_deg: float = 60.0
    search_wp_m: float = 40.0
    verify_agl_m: float = 70.0
    verify_radius_m: float = 80.0
    verify_s: float = 60.0
    verify_arrive_m: float = 120.0
    rtb_orbit_radius_m: float = 60.0
    rtb_orbit_speed: float = 16.0
    rtb_orbit_bank_deg: float = 40.0
    thermal_bank_deg: float = 35.0
    thermal_speed_factor: float = 1.2
    thermal_exit_climb: float = 0.3
    thermal_max_s: float = 900.0
    thermal_zi_margin_m: float = 50.0
    thermal_confirm_s: float = 4.0
    thermal_exclusion_m: float = 300.0
    ceiling_margin_m: float = 100.0
    floor_m: float = 30.0
    lookahead_s: float = 10.0
    geofence_margin_m: float = 300.0
    geofence_violation_m: float = 500.0
    stall_margin: float = 1.15
    rtb_factor: float = 1.3
    glide_margin: float = 0.7
    min_ground_speed: float = 2.0
    cannot_progress_ms: float = 1.0
    sep_h_m: float = 300.0
    sep_v_m: float = 40.0
    sep_dz_m: float = 50.0
    sep_turn_deg: float = 30.0
    sep_hold_s: float = 15.0
    breach_h_m: float = 100.0
    breach_v_m: float = 25.0
    collision_m: float = 10.0
    base_zone_r_m: float = 500.0
    base_zone_h_m: float = 150.0
    rtb_floor_exempt_m: float = 150.0
    land_radius_m: float = 100.0
    transit_layer_base_m: float = 150.0
    transit_layer_step_m: float = 50.0
    transit_layer_cycle: int = 5
    transit_band_half_m: float = 25.0
    forced_landing_agl_m: float = 30.0


@dataclass
class SensorParams:
    fov_deg: float = 60.0
    p_max: float = 0.95
    target_land_m: float = 0.5
    target_water_m: float = 0.3
    ref_target_m: float = 0.5
    exp_a: float = 2.7
    exp_b: float = 0.7
    cover_mod: dict = _weights({"open": 1.0, "water": 1.0, "urban": 0.6, "forest": 0.2})
    false_alarm_rate: float = 0.002
    false_alarm_bounds: tuple = (0.0005, 0.005)
    skip_above_range_factor: float = 2.0
    verify_p_confirm: float = 0.95
    verify_p_reject_fa: float = 0.97
    vario_noise_ms: float = 0.1
    wind_est_noise_ms: float = 1.0
    rain_radar_noise: float = 0.2


@dataclass
class PeopleParams:
    t_crit_s: float = 2400.0
    hr_exposed: float = 0.75
    hr_sheltered: float = 2.0
    affected_depth_m: float = 0.1
    roof_depth_m: float = 0.5
    to_water_depth_m: float = 0.5
    from_water_depth_m: float = 0.3
    walk_trigger_depth_m: float = 0.1
    walk_trigger_dist_m: float = 200.0
    high_ground_hand_m: float = 3.0
    dt_s: float = 10.0
    horizon_after_s: float = 7200.0
    home_hand_m: float = 5.0
    home_hand_boost: float = 3.0
    vehicle_hand_m: float = 3.0
    vehicle_boost: float = 3.0
    outdoor_river_m: float = 2000.0
    detect_factor: dict = _weights({
        "HOME_INSIDE": 0.02, "ROOF": 1.0, "VEHICLE": 0.2, "OUTDOOR": 1.0,
        "SAFE_HIGH": 1.0, "IN_WATER": 0.8,
    })


@dataclass
class ControllerParams:
    sweep_agl_frac: float = 0.6
    sweep_min_agl_m: float = 60.0
    sweep_speed_min: float = 13.0
    sweep_wind_margin: float = 4.0
    sweep_speed_cap: float = 25.0
    floodplain_hand_m: float = 10.0
    overlap: float = 0.15
    leg_angle_deg: float = 90.0
    w_enter: float = 0.8
    e_soar: float = 0.6
    replan_s: float = 60.0
    verify_all: bool = True


@dataclass
class SwarmParams:
    n_drones: int | None = None  # None: use the scenario's sampled swarm size
    perfect_comms: bool = False


@dataclass
class SimConfig:
    dt_s: float = 1.0
    sensor_dt_s: float = 2.0
    radio_dt_s: float = 1.0
    people_dt_s: float = 10.0
    log_dt_s: float = 2.0
    fast_log_dt_s: float = 10.0
    snapshot_dt_s: float = 30.0
    fast: bool = False
    record_grids: bool = False
    battery_swap: bool = True
    relaunch_min_remaining_s: float = 1200.0
    episode_s: float | None = None  # None: use the scenario's episode length
    repeat_window_s: float = 600.0
    coverage_window_s: float = 1800.0


@dataclass
class Config:
    team: TeamInputs = field(default_factory=TeamInputs)
    bounds: ScenarioBounds = field(default_factory=ScenarioBounds)
    world: WorldConfig = field(default_factory=WorldConfig)
    forecast: ForecastParams = field(default_factory=ForecastParams)
    weather: WeatherParams = field(default_factory=WeatherParams)
    glider: GliderParams | None = None  # derived from team when None
    autopilot: AutopilotParams = field(default_factory=AutopilotParams)
    sensor: SensorParams = field(default_factory=SensorParams)
    people: PeopleParams = field(default_factory=PeopleParams)
    controller: ControllerParams = field(default_factory=ControllerParams)
    swarm: SwarmParams = field(default_factory=SwarmParams)
    sim: SimConfig = field(default_factory=SimConfig)

    def __post_init__(self):
        self.team.validate()
        if self.glider is None:
            self.glider = GliderParams.from_team(self.team)
        vs = float(self.glider.stall_speed(0.0))
        if vs > 12.0:
            warnings.warn(f"level stall speed {vs:.1f} m/s exceeds 12 m/s for mass {self.team.mass_kg} kg")

    def world_key(self) -> str:
        """Hash of everything besides the spec that changes a built world."""
        d = {"world": dataclasses.asdict(self.world), "forecast": dataclasses.asdict(self.forecast),
             "weather": dataclasses.asdict(self.weather), "people": dataclasses.asdict(self.people)}
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------
# TOML loading
# --------------------------------------------------------------------------

_SECTIONS = {
    "team": TeamInputs, "world": WorldConfig, "forecast": ForecastParams,
    "weather": WeatherParams, "glider": GliderParams, "autopilot": AutopilotParams,
    "sensor": SensorParams, "people": PeopleParams, "controller": ControllerParams,
    "swarm": SwarmParams, "sim": SimConfig,
}


def _apply(obj, table: dict, where: str):
    names = {f.name: f for f in fields(obj)}
    out = {}
    for k, v in table.items():
        if k not in names:
            raise KeyError(f"unknown config key [{where}] {k}")
        cur = getattr(obj, k)
        if isinstance(cur, tuple) and isinstance(v, list):
            v = tuple(v)
        out[k] = v
    return replace(obj, **out)


def _apply_bounds(b: ScenarioBounds, table: dict) -> ScenarioBounds:
    out = {}
    names = {f.name for f in fields(b)}
    for k, v in table.items():
        if k not in names:
            raise KeyError(f"unknown config key [bounds] {k}")
        cur = getattr(b, k)
        if isinstance(cur, Bounds):
            if isinstance(v, (int, float)):
                v = Bounds(float(v), float(v), "int" if cur.dist == "int" else "fixed")
            else:
                v = Bounds(float(v[0]), float(v[1]), v[2] if len(v) > 2 else cur.dist)
        elif isinstance(cur, tuple):
            v = tuple(v)
        out[k] = v
    return replace(b, **out)


def load_config(path: str | Path | None = None, team: dict | None = None, **sections) -> Config:
    """Load defaults, then a TOML file, then keyword overrides (dicts per section)."""
    tables: dict[str, dict] = {}
    if path is not None:
        with open(path, "rb") as f:
            tables = tomllib.load(f)
    if team:
        tables.setdefault("team", {}).update({k: v for k, v in team.items() if v is not None})
    for k, v in sections.items():
        tables.setdefault(k, {}).update(v)
    kw: dict[str, Any] = {}
    for name, table in tables.items():
        if name == "bounds":
            kw["bounds"] = _apply_bounds(ScenarioBounds(), table)
        elif name in _SECTIONS:
            kw[name] = (table, _SECTIONS[name])
        else:
            raise KeyError(f"unknown config section [{name}]")
    team_obj = _apply(TeamInputs(), kw.pop("team")[0], "team") if "team" in kw else TeamInputs()
    team_obj.validate()
    built: dict[str, Any] = {"team": team_obj}
    if "bounds" in kw:
        built["bounds"] = kw.pop("bounds")
    for name, (table, cls) in kw.items():
        base = GliderParams.from_team(team_obj) if name == "glider" else cls()
        built[name] = _apply(base, table, name)
    return Config(**built)
