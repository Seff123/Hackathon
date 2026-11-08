"""Experiments: compare, ablation, lambda sweep, swarm size, paired statistics (PLANNER_PLAN section 13)."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ..config import to_jsonable
from ..runner import parse_seeds, read_rows, run_batch, write_rows

OURS = "survival"
METRICS = ["survival_at_risk", "found_fraction", "time_to_first_find_s", "at_risk_found", "crashes", "forced_landings",
           "separation_breaches", "high_risk_coverage", "fa_verify_time_s", "n_at_risk"]


def _ok(rows):
    return [r for r in rows if "error" not in r]


def mean_se(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return {"mean": None, "se": None, "n": 0}
    se = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
    return {"mean": float(v.mean()), "se": se, "n": int(len(v))}


def paired(rows_a, rows_b, key="survival_at_risk"):
    a = {r["seed"]: r[key] for r in _ok(rows_a)}
    b = {r["seed"]: r[key] for r in _ok(rows_b)}
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds], float)
    if len(d) == 0:
        return {"n": 0}
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0
    return {"mean": float(d.mean()), "se": se, "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
            "wins": int((d > 1e-9).sum()), "losses": int((d < -1e-9).sum()), "ties": int((np.abs(d) <= 1e-9).sum()),
            "n": int(len(d)), "seeds": seeds, "diffs": d.round(6).tolist()}


def secured_curves(rows):
    curves = [r["secured_curve"] for r in _ok(rows) if r.get("secured_curve") and r.get("n_at_risk", 0) > 0]
    if not curves:
        return None
    L = min(len(c) for c in curves)
    C = np.array([c[:L] for c in curves])
    return {"t_min": (np.arange(L) * 5).tolist(), "mean": C.mean(0).round(5).tolist(),
            "se": (C.std(0, ddof=1) / math.sqrt(len(C))).round(5).tolist() if len(C) > 1 else [0.0] * L, "n": len(C)}


def summarize(by_ctrl: dict, ours: str = OURS) -> dict:
    out = {"controllers": {}, "paired": {}, "by_flood_type": {}, "by_swarm_size": {}, "secured_curves": {}}
    for c, rows in by_ctrl.items():
        ok = _ok(rows)
        out["controllers"][c] = {m: mean_se([r.get(m) for r in ok]) for m in METRICS}
        out["controllers"][c]["n_errors"] = len(rows) - len(ok)
        out["secured_curves"][c] = secured_curves(rows)
    if ours in by_ctrl:
        for c in by_ctrl:
            if c != ours:
                out["paired"][f"{ours}-{c}"] = paired(by_ctrl[ours], by_ctrl[c])
    for split, key in (("by_flood_type", "flood_type"), ("by_swarm_size", "n_drones")):
        groups = sorted({str(r.get(key)) for rows in by_ctrl.values() for r in _ok(rows)})
        for g in groups:
            sub = {c: [r for r in rows if str(r.get(key)) == g] for c, rows in by_ctrl.items()}
            out[split][g] = {c: mean_se([r["survival_at_risk"] for r in _ok(rs)]) for c, rs in sub.items()}
            if ours in sub:
                out[split][g]["paired"] = {f"{ours}-{c}": {k: v for k, v in paired(sub[ours], sub[c]).items()
                                                            if k not in ("seeds", "diffs")}
                                           for c in sub if c != ours}
    # demo seed: median ours - bayes difference among seeds where the two differ (median seed, not best);
    # most seeds tie exactly, and a tied seed shows nothing about the decision

    ref = "bayes" if "bayes" in by_ctrl else next((c for c in by_ctrl if c != ours), None)
    if ours in by_ctrl and ref:
        pr = paired(by_ctrl[ours], by_ctrl[ref])
        if pr.get("n"):
            d = np.array(pr["diffs"])
            seeds = np.array(pr["seeds"])
            nz = np.abs(d) > 1e-9
            if nz.any():
                d, seeds = d[nz], seeds[nz]
            med = float(np.median(d))
            k = int(np.argmin(np.abs(d - med)))
            out["demo_seed"] = {"seed": int(seeds[k]), "diff": float(d[k]), "median_diff": med, "vs": ref,
                                "among": "non-tied seeds" if nz.any() else "all seeds", "n_non_tied": int(nz.sum())}
    return out


def _save(out_dir: Path, by_ctrl: dict, summary: dict, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for rs in by_ctrl.values() for r in rs]
    write_rows(rows, out_dir / "episodes.csv")
    summary["meta"] = meta
    (out_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=1, default=float))


def run_compare(controllers, seeds, out_root="results/", workers=None, cache="cache/", team=None, config=None,
                tag="compare", controller_kw=None, n_drones=None):
    t0 = time.perf_counter()
    by = {}
    for c in controllers:
        name, kw = c if isinstance(c, tuple) else (c, (controller_kw or {}).get(c, {}))
        base = name.split("+")[0]
        by[name] = run_batch(seeds, base, n_drones, config, team, cache, workers, controller_kw=kw, tag=name)
        for r in by[name]:
            r["controller"] = name
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(out_root) / f"{tag}_{ts}"
    summ = summarize(by, OURS if OURS in by else next(iter(by)))
    _save(out, by, summ, {"seeds": [int(s) for s in seeds], "controllers": list(by), "wall_s": time.perf_counter() - t0})
    return out, summ


def run_ablation(seeds, out_root="results/", workers=None, cache="cache/", **kw):
    from .controllers import ABLATIONS
    ctrls = [(OURS, {})] + [(f"{OURS}+{a}", {"ablations": [a]}) for a in ABLATIONS] + \
            [(f"{OURS}+comms_perfect", {"comms": "perfect"})]
    return run_compare(ctrls, seeds, out_root, workers, cache, tag="ablation", **kw)


def run_lambda_sweep(seeds, lambdas=(0, 0.5, 1, 2, 4, 8), out_root="results/", workers=None, cache="cache/", **kw):
    ctrls = [(OURS, {})] + [(f"heuristic+lf{lf:g}", {"params": {"lambda_f": float(lf)}}) for lf in lambdas]
    return run_compare(ctrls, seeds, out_root, workers, cache, tag="lambda_sweep", **kw)


def run_swarm_size(seeds, sizes=(2, 4, 6, 8), out_root="results/", workers=None, cache="cache/", **kw):
    t0 = time.perf_counter()
    by = {}
    for n in sizes:
        for c in ("bayes", OURS):
            rows = run_batch(seeds, c, n, kw.get("config"), kw.get("team"), cache, workers, tag=f"n{n}")
            for r in rows:
                r["controller"] = c
                r["swarm"] = n
            by[f"{c}@{n}"] = rows
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(out_root) / f"swarm_size_{ts}"
    summ = {"sizes": list(sizes), "bayes": {}, OURS: {}, "paired": {}}
    for n in sizes:
        for c in ("bayes", OURS):
            summ[c][str(n)] = mean_se([r["survival_at_risk"] for r in _ok(by[f"{c}@{n}"])])
        summ["paired"][str(n)] = {k: v for k, v in paired(by[f"{OURS}@{n}"], by[f"bayes@{n}"]).items()
                                  if k not in ("seeds", "diffs")}
    _save(out, by, summ, {"seeds": [int(s) for s in seeds], "sizes": list(sizes), "wall_s": time.perf_counter() - t0})
    return out, summ


def print_summary(summ):
    for c, m in summ.get("controllers", {}).items():
        s = m["survival_at_risk"]
        f = m["found_fraction"]
        print(f"  {c:28s} survival_at_risk {s['mean']:.3f} ± {s['se']:.3f}   found {f['mean']:.3f}   "
              f"crashes {m['crashes']['mean']:.2f}  n={s['n']}")
    for k, p in summ.get("paired", {}).items():
        if p.get("n"):
            print(f"  {k:28s} diff {p['mean']:+.3f} ± {p['se']:.3f}  CI [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  "
                  f"wins {p['wins']} losses {p['losses']} ties {p['ties']}")
    if "demo_seed" in summ:
        print("  demo seed (median, not best):", summ["demo_seed"])


# ----------------------------------------------------------------------------- CLI
def add_cli(sub, team_args):
    p = sub.add_parser("compare", help="compare controllers on the same seeds (headline experiment)")
    p.add_argument("--controllers", default="scripted,bayes,heuristic,survival")
    p.add_argument("--seeds", nargs="+", default=["20000-20099"])
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", default="results/")
    team_args(p)
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("ablation", help="survival planner with each piece removed")
    p.add_argument("--seeds", nargs="+", default=["20000-20099"])
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", default="results/")
    team_args(p)
    p.set_defaults(func=_cmd_ablation)

    p = sub.add_parser("lambda-sweep", help="heuristic baseline over lambda_f, ours as reference")
    p.add_argument("--seeds", nargs="+", default=["20000-20049"])
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", default="results/")
    team_args(p)
    p.set_defaults(func=_cmd_lambda)

    p = sub.add_parser("swarm-size", help="bayes vs survival at 2/4/6/8 drones")
    p.add_argument("--seeds", nargs="+", default=["20000-20049"])
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", default="results/")
    team_args(p)
    p.set_defaults(func=_cmd_swarm)

    p = sub.add_parser("tune", help="equal-budget tuning of a planner (Sobol; optuna if installed)")
    p.add_argument("--planner", default="survival")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--train-seeds", nargs="+", default=["0-29"])
    p.add_argument("--val-seeds", nargs="+", default=["10000-10049"])
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", default="results/")
    team_args(p)
    p.set_defaults(func=_cmd_tune)


def _team(args):
    return {k: getattr(args, k, None) for k in ("mass_kg", "battery_life_min", "battery_wh", "charge_time_min",
                                                 "signal_radius_km", "camera_range_m")}


def _cmd_compare(args):
    out, summ = run_compare(args.controllers.split(","), parse_seeds(args.seeds), args.out, args.workers, args.cache,
                            _team(args), args.config)
    print_summary(summ)
    print("wrote", out)
    return 0


def _cmd_ablation(args):
    out, summ = run_ablation(parse_seeds(args.seeds), args.out, args.workers, args.cache, team=_team(args),
                             config=args.config)
    print_summary(summ)
    print("wrote", out)
    return 0


def _cmd_lambda(args):
    out, summ = run_lambda_sweep(parse_seeds(args.seeds), out_root=args.out, workers=args.workers, cache=args.cache,
                                 team=_team(args), config=args.config)
    print_summary(summ)
    print("wrote", out)
    return 0


def _cmd_swarm(args):
    out, summ = run_swarm_size(parse_seeds(args.seeds), out_root=args.out, workers=args.workers, cache=args.cache,
                               team=_team(args), config=args.config)
    print(json.dumps({k: v for k, v in summ.items() if k != "meta"}, indent=1, default=float)[:2000])
    print("wrote", out)
    return 0


def _cmd_tune(args):
    from .tune import tune
    out, best = tune(args.planner, parse_seeds(args.train_seeds), parse_seeds(args.val_seeds), args.trials, args.top,
                     args.out, args.workers, args.cache, team=_team(args), config=args.config)
    print("best", best)
    print("wrote", out)
    return 0
