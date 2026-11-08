"""Episode scoring from the truth: survival of at-risk people and safety."""
from __future__ import annotations

import math

import numpy as np

from .autopilot import MODE_NAMES, THERMAL


def survival_at_risk(people) -> float:
    s = people.survival()
    risk = people.ever_affected
    if risk.any():
        return float(s[risk].mean())
    return float(s.mean()) if len(s) else 1.0


def secured_curve(finds: list, n_at_risk: int, horizon_s: float, step_s: float = 300.0):
    """Cumulative frozen survival of confirmed at-risk people (per at-risk person) every step_s."""
    ts = np.arange(0.0, horizon_s + 1e-9, step_s)
    if n_at_risk == 0:
        return ts, np.zeros_like(ts)
    out = np.zeros_like(ts)
    for f in finds:
        if f["at_risk"]:
            out[ts >= f["t"]] += f["surv"]
    return ts, out / n_at_risk


def compute_metrics(sim) -> dict:
    p = sim.people
    n = sim.n
    ep = sim.episode_s
    world = sim.world
    found = p.found
    tf = p.t_found[found]
    m = {}
    m["survival_at_risk"] = survival_at_risk(p)
    m["n_people"] = int(p.n)
    m["n_at_risk"] = int(p.ever_affected.sum())
    m["found"] = int(found.sum())
    m["found_fraction"] = float(found.mean()) if p.n else 0.0
    m["at_risk_found"] = int((found & p.ever_affected).sum())
    m["time_to_first_find_s"] = float(tf.min()) if len(tf) else float(ep)
    m["mean_time_to_find_s"] = float(tf.mean()) if len(tf) else float(ep)
    m["lost_people"] = int(p.lost.sum())
    # high-risk coverage: cells truly flooded in the first 30 min observed in that time
    fl = world.flood
    wet = np.zeros(world.terrain.z.shape, dtype=bool)
    for t in np.arange(0.0, 1800.0 + 1, 300.0):
        wet |= fl.depth_grid50(t) > world.cfg.flood.wet_depth
    wet &= ~world.terrain.river_mask
    wf = wet.reshape(-1)
    m["high_risk_coverage"] = float((sim.first_obs[wf] <= 1800.0).mean()) if wf.any() else 0.0
    m["area_observed_km2"] = float(sim.observed.sum() * world.terrain.cell_size ** 2 / 1e6)
    m["repeat_coverage"] = float(sim.repeat.sum() / max(sim.observed.sum(), 1))
    m["fa_verify_time_s"] = float(sim.fa_verify_time)
    m["energy_used_wh"] = float(sim.energy_used.sum())
    air_time = float(sim.mode_time[:, 1:6].sum())
    m["airborne_s"] = air_time
    m["soaring_share"] = float(sim.mode_time[:, THERMAL].sum() / max(air_time, 1.0))
    m["climb_lift_m"] = float(sim.lift_climb.sum())
    m["climb_motor_m"] = float(sim.motor_climb.sum())
    c = sim.counts
    m["crashes"] = int(c["crash"])
    m["midair_collisions"] = int(c["midair"])
    m["forced_landings"] = int(c["forced_landing"])
    m["geofence_violations"] = int(c["geofence_violation"])
    m["separation_breaches"] = int(c["sep_breach"])
    m["stall_count"] = int(sim.stall_steps.sum())
    m["stall_protect"] = int(sim.autopilot.counters["stall_protect"].sum())
    m["terrain_warnings"] = int(sim.autopilot.counters["terrain_warning"].sum())
    m["landings"] = int(c["landing"])
    m["relaunches"] = int(c["relaunch"])
    m["rtb_triggers"] = int(c["rtb"])
    m["crash_fraction"] = float(sim.gs.crashed.sum() / n)
    m["n_drones"] = int(n)
    m["true_detections"] = int(c["true_det"])
    m["false_detections"] = int(c["false_det"])
    m["frames"] = int(c["frames"])
    m["verifications"] = len(sim.verify_log)
    m["confirmed"] = int(sum(v["confirmed"] for v in sim.verify_log))
    m["episode_s"] = float(sim.t)
    # scenario statistics
    spec = world.spec
    land = ~world.terrain.river_mask
    m["flood_type"] = spec.flood_type
    m["peak_extent_frac"] = float(((fl.max_depth50() > world.cfg.flood.wet_depth) & land).sum() / land.sum())
    m["wind10_ms"] = float(spec.wind10_ms)
    m["n_thermals"] = len(world.weather.thermals())
    m["wstar_ms"] = float(spec.wstar_ms)
    m["brier_30"] = float(world.forecast.brier.get(30, float("nan")))
    m["start_offset_h"] = float(spec.start_offset_h)
    secured_t, secured = secured_curve(
        [{"t": float(p.t_found[k]), "surv": float(p.surv_frozen[k]), "at_risk": bool(p.ever_affected[k])}
         for k in np.flatnonzero(found)], int(p.ever_affected.sum()), ep)
    m["secured_curve"] = secured.round(5).tolist()
    return m


SCALAR_KEYS = None


def episode_report(result) -> str:
    m = result.metrics
    lines = [f"seed {result.seed}  controller {result.controller}  drones {result.n_drones}  ({m['flood_type']})",
             f"  survival_at_risk {m['survival_at_risk']:.3f}  found {m['found']}/{m['n_people']}  "
             f"at-risk found {m['at_risk_found']}/{m['n_at_risk']}  first find {m['time_to_first_find_s'] / 60:.1f} min",
             f"  high-risk coverage {m['high_risk_coverage']:.2f}  area {m['area_observed_km2']:.1f} km2  "
             f"repeat {m['repeat_coverage']:.2f}  FA verify {m['fa_verify_time_s']:.0f} s",
             f"  energy {m['energy_used_wh']:.0f} Wh  soaring {m['soaring_share']:.3f}  lift climb {m['climb_lift_m']:.0f} m  "
             f"motor climb {m['climb_motor_m']:.0f} m",
             f"  safety: crashes {m['crashes']}  midair {m['midair_collisions']}  forced {m['forced_landings']}  "
             f"geofence {m['geofence_violations']}  breaches {m['separation_breaches']}  stall {m['stall_count']}  "
             f"terrain warnings {m['terrain_warnings']}  landings {m['landings']}  relaunches {m['relaunches']}",
             f"  scenario: extent {m['peak_extent_frac']:.3f}  wind {m['wind10_ms']:.1f} m/s  thermals {m['n_thermals']}  "
             f"brier30 {m['brier_30']:.3f}  wall {result.wall_s:.1f} s"]
    return "\n".join(lines)
