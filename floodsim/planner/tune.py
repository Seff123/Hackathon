"""Equal-budget tuning for each Bayesian planner (PLANNER_PLAN section 13).

Optuna TPE if installed, otherwise a scrambled Sobol design; best `top` trials re-scored on validation seeds.
Objective J = mean(survival_at_risk - 1.0 [crash] - 0.2 [forced landing]) over the training seeds."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from ..config import to_jsonable
from ..runner import run_batch

SPACES = {
    "survival": {"t_crit_hat_s": (1200.0, 4800.0), "sigma_hr": (0.2, 1.0), "tail_factor": (0.3, 1.5),
                 "eps_welfare": (0.0, 0.1)},
    "heuristic": {"lambda_f": (0.0, 10.0), "lambda_h": (0.0, 5.0), "lead_L_s": (0.0, 3600.0)},
    "bayes": {},
}
COMMON = {"beta_travel": (0.5, 2.0), "mu_energy": (0.0, 2.0), "overlap": (0.0, 0.4), "hysteresis": (0.0, 0.5),
          "replan_s": (30.0, 180.0), "nu_verify": (0.1, 0.9), "mobile_fraction": (0.0, 0.6),
          "scenario_concentration": (2.0, 20.0)}


def space_for(planner: str) -> dict:
    return {**COMMON, **SPACES.get(planner, {})}


def objective(rows) -> float:
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return -1e9
    j = [r["survival_at_risk"] - 1.0 * (r.get("crashes", 0) > 0) - 0.2 * (r.get("forced_landings", 0) > 0) for r in ok]
    return float(np.mean(j))


def _sobol(space, n, seed=0):
    from scipy.stats import qmc
    keys = list(space)
    s = qmc.Sobol(len(keys), scramble=True, seed=seed).random(max(n, 1))
    out = []
    for row in s[:n]:
        out.append({k: float(space[k][0] + u * (space[k][1] - space[k][0])) for k, u in zip(keys, row)})
    return out


def tune(planner, train_seeds, val_seeds, n_trials=30, top=3, out_root="results/", workers=None, cache="cache/",
         team=None, config=None):
    space = space_for(planner)
    t0 = time.perf_counter()
    trials = []
    try:
        import optuna  # noqa: F401
        use_optuna = True
    except ImportError:
        use_optuna = False
    if use_optuna:  # pragma: no cover
        import optuna
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        for k in range(n_trials):
            tr = study.ask()
            params = {key: tr.suggest_float(key, lo, hi) for key, (lo, hi) in space.items()}
            score = objective(run_batch(train_seeds, planner, None, config, team, cache, workers,
                                        controller_kw={"params": params}, progress=False))
            study.tell(tr, score)
            trials.append({"trial": k, "score": score, **params})
            print(f"  trial {k + 1}/{n_trials}: J = {score:.4f}", flush=True)
    else:
        designs = [{}] + _sobol(space, n_trials - 1)        # trial 0 = defaults
        for k, params in enumerate(designs):
            score = objective(run_batch(train_seeds, planner, None, config, team, cache, workers,
                                        controller_kw={"params": params}, progress=False))
            trials.append({"trial": k, "score": score, **params})
            print(f"  trial {k + 1}/{len(designs)}: J = {score:.4f}", flush=True)
    best_train = sorted(trials, key=lambda r: -r["score"])[:top]
    val = []
    for tr in best_train:
        params = {k: v for k, v in tr.items() if k in space}
        s = objective(run_batch(val_seeds, planner, None, config, team, cache, workers,
                                controller_kw={"params": params}, progress=False))
        val.append({"trial": tr["trial"], "train": tr["score"], "val": s, "params": params})
    best = max(val, key=lambda r: r["val"])
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(out_root) / f"tune_{planner}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    keys = ["trial", "score"] + list(space)
    with open(out / "trials.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in trials:
            w.writerow({k: r.get(k, "") for k in keys})
    (out / "best_params.json").write_text(json.dumps(to_jsonable({"planner": planner, "best": best, "validation": val,
                                                                    "sampler": "optuna" if use_optuna else "sobol",
                                                                    "train_seeds": list(map(int, train_seeds)),
                                                                    "val_seeds": list(map(int, val_seeds)),
                                                                    "wall_s": time.perf_counter() - t0}), indent=1))
    return out, best
