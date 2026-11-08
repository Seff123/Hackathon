"""All configuration dataclasses. Every number in the build plan lives here as a default.

Team inputs live in `TeamInputs` and can be overridden from `configs/default.toml` ([team]) or the CLI.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

G = 9.81
RHO = 1.225


# ----------------------------------------------------------------------------- bounds
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
            mu, sd = 0.5 * (self.lo + self.hi), (self.hi - self.lo) / 4.0
            return float(min(self.hi, max(self.lo, rng.normal(mu, sd))))
        raise ValueError(self.dist)

    def contains(self, v: float, tol: float = 1e-9) -> bool:
        return self.lo - tol <= v <= self.hi + tol


# ----------------------------------------------------------------------------- team inputs
@dataclass
class TeamInputs:
    mass_kg: float = 5.0
    battery_life_min: float = 150.0
    battery_wh: float | None = None
    charge_time_min: float = 10.0
    signal_radius_km: float = 3.0
    camera_range_m: float = 140.0

    def validate(self) -> None:
        if not (1.0 <= self.mass_kg <= 25.0):
            raise ValueError(f"mass_kg {self.mass_kg} outside 1-25 kg")
        if self.battery_wh is None and not (20.0 <= self.battery_life_min <= 600.0):
            raise ValueError(f"battery_life_min {self.battery_life_min} outside 20-600 min")

    @property
    def ceiling_agl(self) -> float:
        return max(400.0, 2.0 * self.camera_range_m)


# ----------------------------------------------------------------------------- scenario bounds
@dataclass
class ScenarioBounds:
    # terrain
    relief_m: Bounds = Bounds(20.0, 300.0, "loguniform")
    spectral_beta: Bounds = Bounds(1.8, 2.4)
    regional_slope: Bounds = Bounds(0.001, 0.008)
    outlet_edge: dict = field(default_factory=lambda: {"N": 1.0, "E": 1.0, "S": 1.0, "W": 1.0})
    valley_width_m: Bounds = Bounds(300.0, 2000.0)
    valley_depth_m: Bounds = Bounds(10.0, 60.0)
    sinuosity: Bounds = Bounds(1.1, 1.8)
    channel_width_m: Bounds = Bounds(20.0, 80.0)
    channel_depth_m: Bounds = Bounds(2.0, 5.0)
    n_towns: Bounds = Bounds(1, 3, "int")
    town_radius_m: Bounds = Bounds(300.0, 1200.0)
    levee_height_m: Bounds = Bounds(0.0, 3.0)
    forest_fraction: Bounds = Bounds(0.05, 0.4)
    floodplain_noise_m: Bounds = Bounds(0.5, 2.0)
    # flood
    flood_type: dict = field(default_factory=lambda: {
        "fluvial": 0.45, "fluvial+pluvial": 0.25, "pluvial": 0.2, "breach": 0.1})
    peak_inflow_m3s: Bounds = Bounds(100.0, 1500.0, "loguniform")
    base_flow_frac: Bounds = Bounds(0.05, 0.15)
    time_to_peak_h: Bounds = Bounds(1.0, 6.0)
    hydrograph_k: Bounds = Bounds(2.0, 5.0)
    start_offset_h: Bounds = Bounds(-0.5, 4.0)
    peak_rain_mmh: Bounds = Bounds(10.0, 60.0)
    climate_factor: Bounds = Bounds(1.0, 1.4)
    storm_duration_h: Bounds = Bounds(0.5, 4.0)
    storm_radius_km: Bounds = Bounds(2.0, 8.0)
    manning_channel: Bounds = Bounds(0.03, 0.04)
    manning_floodplain: Bounds = Bounds(0.04, 0.07)
    manning_urban: Bounds = Bounds(0.06, 0.12)
    manning_forest: Bounds = Bounds(0.08, 0.15)
    breach_time_h: Bounds = Bounds(0.0, 2.0)
    breach_width_m: Bounds = Bounds(20.0, 100.0)
    breach_head_m: Bounds = Bounds(1.0, 4.0)
    fc_sigma0_m: Bounds = Bounds(0.1, 0.3)
    fc_growth_mph: Bounds = Bounds(0.1, 0.4)
    fc_bias_m: Bounds = Bounds(-0.3, 0.3)
    # weather
    wind10_ms: Bounds = Bounds(1.0, 8.0)
    wind_from_deg: Bounds = Bounds(0.0, 360.0)
    cloud_cover: Bounds = Bounds(0.3, 1.0)
    wstar_base_ms: Bounds = Bounds(0.0, 2.5)
    wstar_factor: float = 1.4
    zi_m: Bounds = Bounds(400.0, 1500.0)
    thermal_density_km2: Bounds = Bounds(0.0, 0.3)
    start_hour: Bounds = Bounds(7.0, 15.0)
    ridge_k: Bounds = Bounds(0.8, 1.2)
    storm_track_dev_deg: Bounds = Bounds(-30.0, 30.0)
    # people / aircraft
    n_targets: Bounds = Bounds(3, 25, "int")
    category_mean: tuple = (0.55, 0.2, 0.25)  # home, vehicle, outdoor
    category_concentration: float = 20.0
    immobile_fraction: Bounds = Bounds(0.2, 0.5)
    walk_speed_ms: Bounds = Bounds(0.5, 1.3)
    drift_fraction: Bounds = Bounds(0.3, 0.8)
    n_drones: Bounds = Bounds(2, 8, "int")
    lambda_fa: Bounds = Bounds(0.0005, 0.005)


# ----------------------------------------------------------------------------- world-level constants
@dataclass
class WorldConfig:
    domain_m: float = 10_000.0
    cell_m: float = 50.0          # terrain / sensor grid
    flood_cell_m: float = 100.0   # flood solver grid
    episode_s: float = 2.5 * 3600
    post_episode_s: float = 2 * 3600   # hazard-dose horizon after episode end
    spinup_s: float = 1800.0
    depth_snapshot_s: float = 60.0
    velocity_snapshot_s: float = 300.0
    drainage_area_km2: float = 1.0
    town_max_river_dist_m: float = 1500.0
    town_min_separation_m: float = 2000.0
    levee_town_dist_m: float = 1000.0
    base_town_dist_m: float = 2000.0
    base_clear_radius_m: float = 150.0
    forest_h: tuple = (15.0, 25.0)
    urban_h: tuple = (6.0, 20.0)
    max_attempts: int = 50
    # rejection checks
    extent_frac: tuple = (0.03, 0.40)
    extent_check_margin: float = 1.45  # the cheap estimate must clear the band by this factor
    targets_near_flood_frac: float = 0.6
    targets_near_flood_m: float = 1000.0
    max_terrain_above_base_m: float = 300.0
    max_wind_300_ms: float = 15.0
    min_drain_to_outlet: float = 0.8
    est_channel_factor: float = 0.7      # cheap-estimate conveyance calibration vs the full solver
    est_floodplain_factor: float = 0.3

    @property
    def n(self) -> int:
        return int(round(self.domain_m / self.cell_m))

    @property
    def nf(self) -> int:
        return int(round(self.domain_m / self.flood_cell_m))


@dataclass
class FloodParams:
    g: float = G
    cfl: float = 0.7
    dt_min: float = 0.5
    dt_max: float = 10.0
    wet_tol: float = 1e-3
    vel_min_depth: float = 0.05
    runoff_urban: float = 0.9
    runoff_open: float = 0.5
    runoff_forest: float = 0.3
    wet_depth: float = 0.1         # depth that counts as "flooded"
    storm_speed_factor: float = 0.5
    storm_speed_bounds: tuple = (1.0, 5.0)


@dataclass
class ForecastParams:
    extent_refresh_s: float = 600.0
    extent_correct_p: float = 0.95
    issue_every_s: float = 300.0
    max_lead_s: float = 3600.0
    lead_step_s: float = 300.0
    ar_corr_m: float = 2000.0
    velocity_refresh_s: float = 600.0
    velocity_noise: float = 0.3
    rain_noise: float = 0.2
    observed_wet_p: float = 0.95
    trib_ref_area_km2: float = 25.0   # forecast error scales with sqrt(upstream area / this) on tributaries
    sigma_floor_m: float = 0.05
    pluvial_sigma_m: float = 0.1
    pluvial_area_cap: float = 20.0


@dataclass
class WeatherParams:
    z0: float = 0.1
    profile_top_m: float = 600.0
    ridge_smooth_m: float = 150.0
    ridge_relief_radius_m: float = 2000.0
    ridge_min_sink: float = -2.0
    thermal_core: tuple = (0.8, 1.5)
    thermal_radius_frac: tuple = (0.1, 0.2)
    thermal_radius_clamp: tuple = (60.0, 200.0)
    thermal_cover_weight: dict = field(default_factory=lambda: {"urban": 1.5, "open": 1.0, "forest": 0.5, "water": 0.0})
    netto_noise: float = 0.1
    wind_est_noise: float = 1.0


# ----------------------------------------------------------------------------- glider
@dataclass
class GliderParams:
    mass_kg: float = 5.0
    wing_area: float = 0.90
    span: float = 3.67
    cd0: float = 0.018
    oswald: float = 0.85
    cl_max: float = 1.3
    v_min: float = 9.0
    v_max: float = 30.0
    bank_limit_deg: float = 45.0
    bank_limit_safety_deg: float = 60.0
    thrust_max: float = 14.0
    prop_eff: float = 0.65
    payload_w: float = 25.0
    battery_life_min: float = 150.0
    battery_wh: float | None = None
    reserve_frac: float = 0.2
    v_ref: float = 13.0
    climb_speed: float = 12.0
    # performance-table grid
    table_v: tuple = (9.0, 30.0, 0.5)
    table_bank_deg: tuple = (0.0, 60.0, 5.0)

    @staticmethod
    def from_team(team: TeamInputs, **kw) -> "GliderParams":
        return GliderParams(mass_kg=team.mass_kg, battery_life_min=team.battery_life_min,
                            battery_wh=team.battery_wh, **kw)

    @property
    def aspect(self) -> float:
        return self.span ** 2 / self.wing_area

    @property
    def k(self) -> float:
        return 1.0 / (math.pi * self.oswald * self.aspect)

    @property
    def weight(self) -> float:
        return self.mass_kg * G

    def drag(self, v, bank=0.0):
        """Drag (N) from the parabolic polar, lift coefficient clamped at cl_max."""
        import numpy as np
        v = np.asarray(v, dtype=float)
        q = 0.5 * RHO * v * v * self.wing_area
        cl = np.minimum(self.weight / (q * np.cos(bank)), self.cl_max)
        return q * (self.cd0 + self.k * cl * cl)

    def sink(self, v, bank=0.0):
        import numpy as np
        return self.drag(v, bank) * np.asarray(v, dtype=float) / self.weight

    def level_power_w(self, v):
        return self.drag(v, 0.0) * v / self.prop_eff + self.payload_w

    def stall_speed(self, bank=0.0):
        import numpy as np
        return np.sqrt(2 * self.weight / (RHO * self.wing_area * self.cl_max * np.cos(bank)))

    @property
    def capacity_wh(self) -> float:
        if self.battery_wh is not None:
            return float(self.battery_wh)
        return float(self.battery_life_min / 60.0 * self.level_power_w(self.v_ref))

    @property
    def reserve_wh(self) -> float:
        return self.reserve_frac * self.capacity_wh


# ----------------------------------------------------------------------------- autopilot / controllers
@dataclass
class AutopilotParams:
    transit_speed: float = 15.0
    transit_speed_cap: float = 25.0
    sink_bonus_speed: float = 2.0
    fast_descent_speed: float = 18.0
    search_descent_speed: float = 20.0
    arrive_radius: float = 150.0
    bank_cmd_limit_deg: float = 35.0
    search_leg_bank_deg: float = 25.0
    thermal_bank_deg: float = 35.0
    thermal_speed_factor: float = 1.2
    thermal_orbit_r: float = 40.0
    thermal_enter_s: float = 4.0
    thermal_min_climb: float = 0.3
    thermal_max_s: float = 900.0
    thermal_occupied_m: float = 300.0
    verify_agl: float = 70.0
    verify_radius: float = 80.0
    verify_s: float = 60.0
    verify_stack_m: float = 500.0
    rtb_orbit_r: float = 70.0
    rtb_approach_agl: float = 80.0
    land_radius: float = 100.0
    rtb_exempt_radius: float = 150.0
    rtb_energy_factor: float = 1.3
    rtb_min_gs: float = 2.0
    glide_home_eff: float = 0.7
    rtb_fixed_trigger: float | None = None   # fallback: fixed energy fraction
    search_tau_gain: float = 0.02
    band_hysteresis: float = 50.0
    layer_base_agl: float = 120.0
    layer_step: float = 50.0
    layer_cycle: int = 5
    layer_half_band: float = 25.0
    carrot_m: float = 100.0
    waypoint_tol: float = 40.0
    safety_floor: float = 30.0
    lookahead_s: float = 10.0
    geofence_margin: float = 300.0
    geofence_limit: float = 500.0
    stall_margin: float = 1.15
    cannot_progress_gs: float = 1.0
    base_zone_r: float = 500.0
    base_zone_h: float = 150.0
    sep_h: float = 300.0
    sep_v: float = 40.0
    sep_exit_h: float = 400.0         # a conflict lasts until the pair is this far apart horizontally
    sep_cpa_s: float = 20.0           # look-ahead for closest point of approach
    sep_dz: float = 50.0
    sep_turn_deg: float = 30.0
    breach_h: float = 100.0
    breach_v: float = 25.0
    collision_m: float = 10.0
    forced_land_agl: float = 30.0
    launch_stagger_s: float = 20.0     # departure spacing between drones launched together
    launch_hold_r: float = 150.0
    launch_hold_agl: float = 80.0
    sep_descend_min_agl: float = 110.0


@dataclass
class ControllerParams:
    search_height_frac: float = 0.6
    search_height_min: float = 60.0
    search_speed_min: float = 13.0
    search_speed_margin: float = 4.0
    search_speed_cap: float = 25.0
    overlap: float = 0.15
    leg_angle_deg: float = 90.0
    floodplain_hand_m: float = 10.0
    e_soar: float = 0.6
    w_enter: float = 0.8
    replan_s: float = 60.0
    verify_all: bool = True


@dataclass
class SwarmParams:
    n_drones: int | None = None   # None = use the scenario's sampled swarm size
    perfect_comms: bool = False


@dataclass
class SensorParams:
    fov_deg: float = 60.0
    camera_range_m: float = 140.0
    land_size: float = 0.5
    water_size: float = 0.3
    cover_mod: dict = field(default_factory=lambda: {"open": 1.0, "water": 1.0, "urban": 0.6, "forest": 0.2})
    lambda_fa: float = 0.002
    verify_confirm_p: float = 0.95
    verify_reject_fa_p: float = 0.97
    verify_match_m: float = 75.0

    @property
    def tan_half(self) -> float:
        return math.tan(math.radians(self.fov_deg / 2))


@dataclass
class PeopleParams:
    t_crit_s: float = 40 * 60.0
    hr_exposed: float = 0.75
    hr_sheltered: float = 2.0
    roof_depth: float = 0.5
    affected_depth: float = 0.1
    walk_trigger_depth: float = 0.1
    walk_trigger_flood_m: float = 200.0
    in_water_depth: float = 0.5
    leave_water_depth: float = 0.3
    high_ground_hand_m: float = 3.0
    home_low_hand: float = 5.0
    home_low_boost: float = 3.0
    vehicle_low_hand: float = 3.0
    outdoor_river_m: float = 2000.0
    n_hat_sigma: float = 0.3
    rain_radar_noise: float = 0.2


@dataclass
class SimConfig:
    dt: float = 1.0
    sensor_dt: float = 2.0
    radio_dt: float = 1.0
    people_dt: float = 10.0
    traj_dt: float = 2.0
    traj_dt_fast: float = 10.0
    grid_dt: float = 30.0
    fast: bool = False
    record_grids: bool = False
    swap: bool = True
    relaunch_min_s: float = 20 * 60.0
    perfect_comms: bool = False
    n_drones: int | None = None
    max_steps: int | None = None


# ----------------------------------------------------------------------------- scenario spec
@dataclass
class ScenarioSpec:
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
    town_radius_m: list
    levee_height_m: float
    forest_fraction: float
    floodplain_noise_m: float
    river_start_frac: float
    river_end_frac: float
    # flood
    flood_type: str
    peak_inflow_m3s: float
    base_flow_frac: float
    time_to_peak_h: float
    hydrograph_k: float
    start_offset_h: float
    peak_rain_mmh: float
    climate_factor: float
    storm_duration_h: float
    storm_radius_km: float
    storm_track_deg: float
    manning_channel: float
    manning_floodplain: float
    manning_urban: float
    manning_forest: float
    breach_time_h: float
    breach_width_m: float
    breach_head_m: float
    fc_sigma0_m: float
    fc_growth_mph: float
    fc_bias_m: float
    # weather
    wind10_ms: float
    wind_from_deg: float
    cloud_cover: float
    wstar_ms: float
    zi_m: float
    thermal_density_km2: float
    start_hour: float
    ridge_k: float
    # people / aircraft
    n_targets: int
    category_weights: list
    immobile_fraction: float
    walk_speed_ms: float
    drift_fraction: float
    n_drones: int
    lambda_fa: float
    # team inputs (copied so the hash covers them)
    mass_kg: float
    battery_life_min: float
    battery_wh: float | None
    charge_time_min: float
    signal_radius_km: float
    camera_range_m: float
    # named scenario preset and its generator knobs (floodsim/presets.py); None for ordinary scenarios
    preset: dict | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("preset") is None:
            d.pop("preset", None)   # keeps the hash (and the flood cache) of ordinary scenarios unchanged
        return d

    @staticmethod
    def from_dict(d: dict) -> "ScenarioSpec":
        names = {f.name for f in fields(ScenarioSpec)}
        return ScenarioSpec(**{k: v for k, v in d.items() if k in names})

    @property
    def spec_hash(self) -> str:
        s = json.dumps(self.to_dict(), sort_keys=True, default=float)
        return hashlib.sha1(s.encode()).hexdigest()

    @property
    def wind_to_rad(self) -> float:
        """Direction the wind blows towards, radians anticlockwise from east."""
        return math.radians(270.0 - self.wind_from_deg)

    @property
    def team(self) -> TeamInputs:
        return TeamInputs(self.mass_kg, self.battery_life_min, self.battery_wh, self.charge_time_min,
                          self.signal_radius_km, self.camera_range_m)

    @property
    def has_fluvial(self) -> bool:
        return self.flood_type in ("fluvial", "fluvial+pluvial", "breach")

    @property
    def has_pluvial(self) -> bool:
        return self.flood_type in ("fluvial+pluvial", "pluvial")

    @property
    def has_breach(self) -> bool:
        return self.flood_type == "breach"


# ----------------------------------------------------------------------------- toml loading
def _update_dc(obj, d: dict):
    for k, v in d.items():
        if hasattr(obj, k):
            cur = getattr(obj, k)
            if isinstance(cur, Bounds) and isinstance(v, (list, tuple)):
                v = Bounds(float(v[0]), float(v[1]), v[2] if len(v) > 2 else cur.dist)
            setattr(obj, k, v)
        else:
            raise KeyError(f"unknown config key {k!r} for {type(obj).__name__}")
    return obj


@dataclass
class AirframeParams:
    """Airframe and power-train settings applied on top of the team inputs (defaults = GliderParams defaults)."""
    cd0: float = 0.018
    oswald: float = 0.85
    prop_eff: float = 0.65
    payload_w: float = 25.0


@dataclass
class Config:
    airframe: AirframeParams = field(default_factory=AirframeParams)
    team: TeamInputs = field(default_factory=TeamInputs)
    bounds: ScenarioBounds = field(default_factory=ScenarioBounds)
    world: WorldConfig = field(default_factory=WorldConfig)
    flood: FloodParams = field(default_factory=FloodParams)
    forecast: ForecastParams = field(default_factory=ForecastParams)
    weather: WeatherParams = field(default_factory=WeatherParams)
    autopilot: AutopilotParams = field(default_factory=AutopilotParams)
    controller: ControllerParams = field(default_factory=ControllerParams)
    sensor: SensorParams = field(default_factory=SensorParams)
    people: PeopleParams = field(default_factory=PeopleParams)
    sim: SimConfig = field(default_factory=SimConfig)


DEFAULT_TOML = Path(__file__).resolve().parent.parent / "configs" / "default.toml"


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    p = Path(path) if path else DEFAULT_TOML
    if p.exists() and tomllib is not None:
        with open(p, "rb") as fh:
            data = tomllib.load(fh)
        for section, values in data.items():
            if hasattr(cfg, section):
                _update_dc(getattr(cfg, section), values)
    cfg.team.validate()
    return cfg


def replace(obj, **kw):
    return dataclasses.replace(obj, **kw)


def to_jsonable(obj: Any):
    if dataclasses.is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:  # pragma: no cover
        pass
    return obj
