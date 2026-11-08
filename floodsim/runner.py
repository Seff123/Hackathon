"""Running episodes in parallel and writing results."""
from __future__ import annotations

import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .config import Config, SimConfig, TeamInputs, load_config, to_jsonable


def parse_seeds(specs) -> list[int]:
    if isinstance(specs, str):
        specs = [specs]
    out = []
    for s in specs:
        for part in str(s).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part[1:]:
                a, b = part.split("-", 1) if not part.startswith("-") else (part, part)
                out += list(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
    return out


def make_config(config_path=None, team_overrides: dict | None = None) -> Config:
    cfg = load_config(config_path)
    for k, v in (team_overrides or {}).items():
        if v is not None:
            setattr(cfg.team, k, v)
    cfg.team.validate()
    return cfg


CONTROLLERS = {}


def register(name):
    def deco(fn):
        CONTROLLERS[name] = fn
        return fn
    return deco


@register("scripted")
def _scripted(cfg: Config, **kw):
    from .controllers import ScriptedSweep
    return ScriptedSweep(cfg.controller)


def get_controller(name: str, cfg: Config, **kw):
    if name not in CONTROLLERS:
        try:
            from .planner import controllers as _pc  # noqa: F401  (registers planner controllers)
        except ImportError:
            pass
    if name not in CONTROLLERS:
        raise KeyError(f"unknown controller {name!r}; known: {sorted(CONTROLLERS)}")
    return CONTROLLERS[name](cfg, **kw)


def run_episode(seed: int, controller: str = "scripted", n_drones: int | None = None, config_path=None,
                team: dict | None = None, cache_dir="cache/", fast: bool = True, record: bool = False,
                controller_kw: dict | None = None, sim_kw: dict | None = None, attempt: int | None = None):
    from .scenario import build_world, sample_spec
    from .sim import Sim
    cfg = make_config(config_path, team)
    spec = sample_spec(seed, cfg=cfg, attempt=attempt)
    world = build_world(spec, cache_dir=cache_dir, cfg=cfg)
    ctrl = get_controller(controller, cfg, **(controller_kw or {}))
    sc = SimConfig(fast=fast and not record, record_grids=record, **(sim_kw or {}))
    sim = Sim(world, ctrl, sc, n_drones=n_drones)
    return sim.run(), world


def _episode_task(args):
    seed, controller, n_drones, config_path, team, cache_dir, controller_kw, sim_kw, tag = args
    t0 = time.perf_counter()
    try:
        res, world = run_episode(seed, controller, n_drones, config_path, team, cache_dir, True, False,
                                 controller_kw, sim_kw)
    except Exception as e:  # keep the batch going; report the failure
        return {"seed": seed, "controller": controller, "tag": tag, "error": repr(e)}
    row = {"seed": seed, "controller": controller, "tag": tag, "spec_hash": res.spec_hash, "n_drones": res.n_drones,
           "wall_s": round(time.perf_counter() - t0, 2)}
    for k, v in res.metrics.items():
        if isinstance(v, (int, float, str, bool)):
            row[k] = v
    row["secured_curve"] = res.metrics.get("secured_curve")
    row["finds"] = res.finds
    row["planner_diag"] = getattr(res, "planner_diag", None) or {}
    return row


def run_batch(seeds, controller="scripted", n_drones=None, config_path=None, team=None, cache_dir="cache/",
              workers=None, controller_kw=None, sim_kw=None, tag="", progress=True, drones_by_seed=None):
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    tasks = [(s, controller, (drones_by_seed or {}).get(s, n_drones), config_path, team, cache_dir, controller_kw,
              sim_kw, tag) for s in seeds]
    rows = []
    t0 = time.perf_counter()
    if workers == 1:
        it = map(_episode_task, tasks)
        for k, r in enumerate(it):
            rows.append(r)
            if progress:
                _progress(k + 1, len(tasks), t0, controller, tag)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, r in enumerate(ex.map(_episode_task, tasks, chunksize=1)):
                rows.append(r)
                if progress:
                    _progress(k + 1, len(tasks), t0, controller, tag)
    return rows


def _progress(k, n, t0, controller, tag):
    if k == n or k % max(1, n // 10) == 0:
        print(f"  [{controller}{(' ' + tag) if tag else ''}] {k}/{n} episodes, {time.perf_counter() - t0:.0f} s", flush=True)


SKIP_CSV = {"secured_curve", "finds", "planner_diag"}


def write_rows(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys and k not in SKIP_CSV:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    with open(path.with_suffix(".jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(to_jsonable(r)) + "\n")


def read_rows(path: Path) -> list[dict]:
    p = Path(path)
    jl = p.with_suffix(".jsonl") if p.suffix != ".jsonl" else p
    rows = []
    with open(jl) as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def scenario_statistics(rows) -> dict:
    ok = [r for r in rows if "error" not in r]

    def stat(key):
        v = np.array([float(r[key]) for r in ok if key in r and r[key] != "" and r[key] is not None], float)
        v = v[np.isfinite(v)]
        if not len(v):
            return None
        return {"mean": float(v.mean()), "sd": float(v.std()), "min": float(v.min()), "max": float(v.max()),
                "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}

    keys = ["peak_extent_frac", "wind10_ms", "n_thermals", "wstar_ms", "brier_30", "n_people", "n_at_risk",
            "survival_at_risk", "found_fraction", "time_to_first_find_s", "high_risk_coverage", "true_detections",
            "false_detections", "energy_used_wh", "soaring_share", "crashes", "forced_landings",
            "separation_breaches", "midair_collisions", "terrain_warnings", "n_drones", "wall_s"]
    out = {k: stat(k) for k in keys}
    td = sum(r.get("true_detections", 0) for r in ok)
    fd = sum(r.get("false_detections", 0) for r in ok)
    fr = sum(r.get("frames", 0) for r in ok)
    out["detection_rates"] = {"true_per_frame": td / max(fr, 1), "false_per_frame": fd / max(fr, 1),
                              "share_false": fd / max(td + fd, 1)}
    types = {}
    for r in ok:
        types[r.get("flood_type", "?")] = types.get(r.get("flood_type", "?"), 0) + 1
    out["flood_types"] = types
    out["n_episodes"] = len(ok)
    out["errors"] = [r for r in rows if "error" in r]
    return out
