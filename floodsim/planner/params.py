"""PlannerParams and SwarmParams: every default from PLANNER_PLAN section 12."""
from __future__ import annotations

from dataclasses import dataclass, field

COMPONENTS = ("home", "home_low", "road", "open", "high_ground", "anywhere")
SHELTERED = (0, 1, 2)
EXPOSED = (3, 4, 5)


@dataclass
class PlannerParams:
    # survival value
    t_crit_hat_s: float = 40 * 60.0
    sigma_hr: float = 0.5
    tail_factor: float = 1.0
    eps_welfare: float = 0.02
    # heuristic baseline
    lambda_f: float = 2.0
    lambda_h: float = 1.0
    lead_L_s: float = 20 * 60.0
    # scoring
    beta_travel: float = 1.0
    mu_energy: float = 0.5
    overlap: float = 0.15
    hysteresis: float = 0.15
    replan_s: float = 60.0
    nu_verify: float = 0.3
    nu_value_scaled: bool = False
    mobile_fraction: float = 0.3
    scenario_concentration: float = 6.0
    e_soar: float = 0.6
    w_enter: float = 0.8
    block_m: float = 500.0
    heights_frac: tuple = (None, 0.6, 0.8, 1.0)    # None = 60 m
    search_speed_min: float = 13.0
    search_speed_margin: float = 4.0
    search_speed_cap: float = 25.0
    leg_turn_s: float = 20.0
    # belief
    min_revisit_s: float = 20.0
    kappa: float = 0.002
    prior_mean: tuple = (0.30, 0.20, 0.20, 0.15, 0.10, 0.05)
    scenario_step: float = 0.25
    scenario_scales: tuple = (0.5, 0.7, 1.0, 1.4, 2.0)
    scale_sigma: float = 0.3
    predict_every_s: float = 60.0
    walk_speed: float = 0.9
    water_drift_rate_per_min: float = 0.5
    water_drift_depth: float = 0.6
    water_drift_speed: float = 1.0
    drift_fraction: float = 0.5
    verify_pd: float = 0.95
    # value tables
    value_every_s: float = 300.0
    table_every_s: float = 60.0
    # optional pieces / ablations
    state_pd: bool = True
    scenarios: bool = True
    bias_learning: bool = True
    beta_scen: float = 0.0
    prediction: bool = True
    # bias learning
    bias_grid: tuple = (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    bias_prior_sd: float = 0.2
    bias_tau: float = 0.1
    bias_leads_min: tuple = (15, 30, 60)
    bias_max_cells: int = 2000


@dataclass
class SwarmParams:
    claim_expiry_s: float = 300.0
    spread_radius: float = 1500.0
    spread_penalty: float = 0.5
    comms_mode: str = "base_relay"   # or "perfect"
