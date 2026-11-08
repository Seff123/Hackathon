"""Full experiment pipeline (PLANNER_PLAN section 13), reproducible:

1. build caches for train / validation / test seeds
2. equal-budget tuning of bayes, heuristic, survival on train seeds, best 3 re-scored on validation
3. compare (scripted, bayes, heuristic, survival) on the test seeds once, with tuned parameters
4. ablation, lambda sweep and swarm size on test seeds

Usage: python scripts/run_experiments.py [--trials 10] [--train 0-19] [--val 10000-10019] [--test 20000-20099]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from floodsim.planner.experiments import (print_summary, run_ablation, run_compare, run_lambda_sweep,
                                          run_swarm_size)
from floodsim.planner.tune import tune
from floodsim.runner import parse_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--train", default="0-19")
    ap.add_argument("--val", default="10000-10019")
    ap.add_argument("--test", default="20000-20099")
    ap.add_argument("--sweep-test", default="20000-20049")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", default="results/")
    ap.add_argument("--skip-tune", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    for split, seeds in (("train", args.train), ("val", args.val), ("test", args.test)):
        subprocess.run([sys.executable, "-m", "floodsim", "build-cache", "--seeds", seeds, "--split", split,
                        "--workers", str(args.workers)], check=True)
    train, val, test = parse_seeds(args.train), parse_seeds(args.val), parse_seeds(args.test)
    tuned = {}
    if not args.skip_tune:
        for planner in ("bayes", "heuristic", "survival"):
            print(f"== tuning {planner}: {args.trials} trials x {len(train)} train seeds", flush=True)
            out, best = tune(planner, train, val, args.trials, 3, args.out, args.workers)
            tuned[planner] = best["params"]
            print(f"   best {planner}: val J {best['val']:.4f} train J {best['train']:.4f} -> {out}", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "tuned_params.json").write_text(json.dumps(tuned, indent=1))
    kw = {c: {"params": p} for c, p in tuned.items()}
    print("== compare on test seeds (run once)", flush=True)
    out, summ = run_compare(["scripted", "bayes", "heuristic", "survival"], test, args.out, args.workers, controller_kw=kw)
    print_summary(summ)
    print("   ->", out, flush=True)
    (Path(args.out) / "latest_compare.txt").write_text(str(out))
    print("== ablation", flush=True)
    out_a, summ_a = run_ablation(test, args.out, args.workers)
    print_summary(summ_a)
    print("   ->", out_a, flush=True)
    print("== lambda sweep", flush=True)
    out_l, summ_l = run_lambda_sweep(parse_seeds(args.sweep_test), out_root=args.out, workers=args.workers)
    print_summary(summ_l)
    print("   ->", out_l, flush=True)
    print("== swarm size", flush=True)
    out_s, summ_s = run_swarm_size(parse_seeds(args.sweep_test), out_root=args.out, workers=args.workers)
    print(json.dumps(summ_s["paired"], indent=1, default=float))
    print("   ->", out_s, flush=True)
    (Path(args.out) / "latest_runs.json").write_text(json.dumps({"compare": str(out), "ablation": str(out_a),
                                                                 "lambda_sweep": str(out_l), "swarm_size": str(out_s)}))
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
