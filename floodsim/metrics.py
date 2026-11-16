"""Episode scoring from truth (survival of at-risk people, coverage, safety)."""

from __future__ import annotations

import math

import numpy as np

from .autopilot import MODE_NAMES, THERMAL, VERIFY


def compute_metrics(sim, t_end: float) -> dict:
    p = sim.people
    fl = sim.flight
    world = sim.world
    n = sim.n
    surv = p.survival()
    affected = ~np.isnan(p.t_affected)
    at_risk = surv[affected] if affected.any() else surv
    found_t = p.t_found[p.found]
    land = world.terrain.land.ravel()
    win = sim.sc.coverage_window_s
    flooded = (world.flood.max_depth(0.0, win).ravel() > world.cfg.world.wet_depth_m) & land
    seen_early = sim.first_seen <= win
    hr_cov = float((flooded & seen_early).sum() / flooded.sum()) if flooded.any() else 1.0
    observed = sim.seen_count > 0
    airborne = float(fl.airborne_time.sum())
    counts = fl.auto.counts
    crashed = int(fl.s.crashed.sum())
    log = fl.log
    m = {
        "survival_at_risk": float(np.mean(at_risk)) if len(at_risk) else 1.0,
        "n_people": int(p.n),
        "n_affected": int(affected.sum()),
        "found_fraction": float(p.found.mean()) if p.n else 0.0,
        "n_found": int(p.found.sum()),
        # censored at the episode length when nobody is found (keeps every metric finite)
        "time_to_first_find_s": float(found_t.min()) if len(found_t) else float(sim.episode_s),
        "mean_time_to_find_s": float(found_t.mean()) if len(found_t) else float(sim.episode_s),
        "high_risk_coverage": hr_cov,
        "area_observed_km2": float(observed.sum() * world.terrain.cell_size**2 / 1e6),
        "repeat_coverage": float(sim.repeat[observed].mean()) if observed.any() else 0.0,
        "verify_false_alarm_s": float(sim.verify_fa_time),
        "n_detections": len(sim.detections),
        "n_true_detections": int(sum(1 for d in sim.detections if d[3] is not None)),
        "n_verifications": len(sim.verifs),
        "energy_used_wh": float(fl.energy_used.sum()),
        "soaring_share": float(fl.mode_time[:, THERMAL].sum() / airborne) if airborne else 0.0,
        "climb_lift_m": float(fl.lift_climb.sum()),
        "climb_motor_m": float(fl.motor_climb.sum()),
        "crashes": int(log.crashes),
        "crash_fraction": crashed / n,
        "forced_landings": int(log.forced_landings),
        "geofence_violations": int(log.geofence_violations),
        "stalls": int(log.stalls),
        "terrain_warnings": int(counts["terrain_warning"].sum()),
        "stall_protections": int(counts["stall_protect"].sum()),
        "mid_air_collisions": int(log.collisions),
        "separation_breaches": int(log.separation_breaches),
        "separation_manoeuvres": int(counts["separation_manoeuvre"].sum()),
        "rtb_energy_triggers": int(counts["rtb_energy"].sum()),
        "n_drones": int(n),
        "episode_end_s": float(t_end),
        "airborne_s": airborne,
        "people_lost": int(p.lost.sum()),
        "relaunches": len(sim.relaunches),
    }
    return m
