"""Batch evaluation: run a controller over many seeds, collect metrics and scenario stats."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import Config

CONTROLLERS = {"scripted": "floodsim.controllers:ScriptedSweep"}


def make_controller(name: str, cfg: Config):
    mod, cls = CONTROLLERS[name].split(":")
    import importlib

    return getattr(importlib.import_module(mod), cls)(cfg.controller)


def scenario_stats(world, result) -> dict:
    spec = world.spec
    th = world.weather.thermal_list()
    z0 = world.cfg.weather.z0_m
    det = result.metrics
    return {
        "seed": spec.seed, "attempt": spec.attempt, "flood_type": spec.flood_type,
        "peak_extent": round(world.peak_extent(), 4), "peak_inflow_m3s": round(spec.peak_inflow_m3s, 1),
        "rain_peak_mmh": round(spec.rain_peak_mmh, 1) if spec.has_rain else 0.0,
        "start_offset_h": round(spec.start_offset_h, 2), "wind10_ms": round(spec.wind10_ms, 2),
        "wind300_ms": round(spec.wind10_ms * math.log(300 / z0) / math.log(10 / z0), 2),
        "w_star_ms": round(spec.w_star_ms, 3), "n_thermals": len(th),
        "mean_thermal_core_ms": round(float(th[:, 2].mean()), 3) if len(th) else 0.0,
        "relief_m": round(spec.relief_m, 1), "n_targets": spec.n_targets, "n_drones": result.n_drones,
        "true_detection_rate": round(det["n_true_detections"] / max(det["n_detections"], 1), 3),
        "detections_per_drone_hour": round(det["n_detections"] / max(det["airborne_s"] / 3600, 1e-9), 2),
    }


def _one(args):
    seed, cfg, controller, n_drones, cache_dir = args
    from . import scenario as S
    from . import world as W
    from .sim import Sim

    spec = S.sample_spec(seed, cfg=cfg)
    w = W.build_world(spec, cfg, cache_dir=cache_dir)
    sc = replace(cfg.sim, fast=True)
    res = Sim(w, make_controller(controller, cfg), sc, n_drones=n_drones, cfg=cfg).run()
    row = {"seed": seed, "controller": controller, **res.metrics, "wall_s": round(res.wall_s, 2)}
    stats = scenario_stats(w, res)
    W._MEM.clear()
    W._FLOOD.clear()
    S._TERRAIN_CACHE.clear()
    return row, stats


def evaluate(seeds, cfg: Config, controller="scripted", n_drones=None, out_dir="results", workers=0,
             cache_dir="cache", progress=True):
    workers = workers or min(os.cpu_count() or 1, 4)
    out = Path(out_dir) / f"eval_{controller}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, cfg, controller, n_drones, cache_dir) for s in seeds]
    rows, stats = [], []
    t0 = time.perf_counter()
    it = map(_one, jobs) if workers <= 1 else ProcessPoolExecutor(workers).map(_one, jobs)
    for k, (row, st) in enumerate(it):
        rows.append(row)
        stats.append(st)
        if progress:
            print(f"[{k + 1}/{len(seeds)}] seed {row['seed']}: {row['n_drones']} drones, "
                  f"survival_at_risk {row['survival_at_risk']:.3f}, found {row['n_found']}/{row['n_people']}, "
                  f"crashes {row['crashes']}", flush=True)
    _write_csv(out / "episodes.csv", rows)
    _write_csv(out / "scenarios.csv", stats)
    summary = summarise(rows, stats)
    summary["wall_s"] = round(time.perf_counter() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return out, summary


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _ms(vals):
    a = np.asarray(vals, float)
    if len(a) == 0:
        return {"mean": float("nan"), "se": float("nan")}
    return {"mean": round(float(a.mean()), 4), "se": round(float(a.std(ddof=1) / math.sqrt(len(a))), 4) if len(a) > 1 else 0.0,
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}


def summarise(rows, stats) -> dict:
    out = {"n_episodes": len(rows)}
    for k in ("survival_at_risk", "found_fraction", "high_risk_coverage", "area_observed_km2", "repeat_coverage",
              "energy_used_wh", "soaring_share", "crash_fraction", "separation_breaches", "terrain_warnings",
              "time_to_first_find_s", "n_detections", "wall_s"):
        out[k] = _ms([r[k] for r in rows])
    out["total_crashes"] = int(sum(r["crashes"] for r in rows))
    out["total_collisions"] = int(sum(r["mid_air_collisions"] for r in rows))
    out["total_forced_landings"] = int(sum(r["forced_landings"] for r in rows))
    for k in ("peak_extent", "wind10_ms", "wind300_ms", "w_star_ms", "n_thermals", "true_detection_rate",
              "detections_per_drone_hour"):
        out["scenario_" + k] = _ms([s[k] for s in stats])
    types = {}
    for s in stats:
        types[s["flood_type"]] = types.get(s["flood_type"], 0) + 1
    out["flood_types"] = types
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n_drones"], []).append(r["survival_at_risk"])
    out["survival_by_swarm_size"] = {str(k): round(float(np.mean(v)), 4) for k, v in sorted(by_n.items())}
    return out
