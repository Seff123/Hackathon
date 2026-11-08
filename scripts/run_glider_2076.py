"""Today's glider (configs/default.toml) vs a very good 2076 glider (configs/glider_2076.toml) on the
"big town far from the flood, crowded" scenario (preset town_far+crowded), with Bayesian search and the
survival-aware planner, then GIFs of the survival-aware planner flying the median flood with each glider.

    python scripts/run_glider_2076.py                 # runs (~35 min) + GIFs
    python scripts/run_glider_2076.py --gifs-only     # GIFs from the latest runs

Results: results/glider_today15/ and results/glider2076/; GIFs and MP4s in site/glider2076/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PRESET = "town_far+crowded"
GLIDERS = {"today": ("configs/default.toml", "results/glider_today15/"),
           "2076": ("configs/glider_2076.toml", "results/glider2076/")}


def _rows(root):
    d = Path((ROOT / root / "latest_scenarios.txt").read_text().split()[-1])
    return [json.loads(line) for line in open(ROOT / d / "episodes.jsonl")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-80")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--gifs-only", action="store_true")
    args = ap.parse_args()
    from floodsim.planner.experiments import paired
    from floodsim.runner import parse_seeds
    from floodsim.viz import animate_episode
    from floodsim.vizpage.cli import tuned_kw
    from floodsim.vizpage.pitch import make_pitch
    from floodsim.vizpage.record import record_episode
    from floodsim.vizpage.showcase import run_scenarios

    tuned = tuned_kw(str(ROOT / "results" / "tuned_params.json"))
    if not args.gifs_only:
        for name, (cfg, out) in GLIDERS.items():
            print(f"== {name} glider", flush=True)
            run_scenarios([PRESET], parse_seeds(args.seeds), ("bayes", "survival"), out_root=str(ROOT / out),
                          workers=args.workers, cache=str(ROOT / "cache"), config=str(ROOT / cfg), tuned=tuned)
    rows = {k: [r for r in _rows(out) if "error" not in r] for k, (_, out) in GLIDERS.items()}
    g = lambda k, c: [r for r in rows[k] if r["controller"] == c]
    for c in ("bayes", "survival"):
        p = paired(g("2076", c), g("today", c))
        print(f"{c:9s} 2076 - today {p['mean']:+.3f} CI [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] "
              f"wins {p['wins']} losses {p['losses']} ties {p['ties']}", flush=True)
    # GIF flood: median 2076 - today difference for the survival-aware planner among floods where they differ
    p = paired(g("2076", "survival"), g("today", "survival"))
    d, s = np.array(p["diffs"]), np.array(p["seeds"])
    nz = np.abs(d) > 1e-9
    d, s = (d[nz], s[nz]) if nz.any() else (d, s)
    seed = int(s[int(np.argmin(np.abs(d - np.median(d))))])
    print(f"GIF flood: seed {seed} (median of {len(d)} non-tied floods)", flush=True)
    site = ROOT / "site" / "glider2076"
    site.mkdir(parents=True, exist_ok=True)
    runs = [record_episode(seed, "survival", None, str(ROOT / "cache"), None, str(ROOT / cfg), tuned.get("survival"),
                           preset=PRESET) for cfg, _ in GLIDERS.values()]
    for (r, w), name in zip(runs, GLIDERS):
        print("wrote", animate_episode(r, w, site / f"episode_{name}_seed{seed}.gif"),
              f"(survival_at_risk {r.metrics['survival_at_risk']:.2f})", flush=True)
    make_pitch(runs, site, labels=["Today's glider", "2076 glider"], name=f"glider_today_vs_2076_seed{seed}",
               suptitle="Same flood, same planner (survival-aware), today's vs 2076 glider")
    print("wrote", site, flush=True)


if __name__ == "__main__":
    main()
