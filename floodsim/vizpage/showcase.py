"""Scenario showcase: run Bayesian search and the survival-aware planner on every scenario preset
(floodsim/presets.py), export one compare page per preset and a selector page to switch between them.

Each preset is a small paired experiment: the same base seeds give the same worlds to both planners, so each
card reports the paired difference over all of the preset's floods, not only the one shown. The flood shown is
the median ours − Bayesian difference among floods where the two differ (median, not best), the same rule as
the main comparison. Truth features of each world (walk to dry ground, town distance from the flood, people in
the flood) are measured offline to check that the preset produced the situation it names.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

WET_M = 0.3      # a cell counts as flooded when its maximum depth exceeds this
DRY_M = 0.1      # a cell is a refuge when its maximum depth stays below this
FEATURE_DT_S = 600.0
LABELS = {"bayes": "Standard Bayesian search", "survival": "Survival-aware (ours)",
          "heuristic": "Flood-bonus heuristic", "scripted": "Scripted lawnmower"}


def flood_features(world) -> dict:
    """Truth features of one flood (offline only, never given to a controller)."""
    from scipy.ndimage import distance_transform_edt as edt

    ter = world.terrain
    c, n = ter.cell_size, ter.n
    times = np.arange(0.0, world.cfg.world.episode_s + 1.0, FEATURE_DT_S)
    D = np.stack([world.flood.depth_grid50(t) for t in times])
    maxd = D.max(axis=0)
    wet, dry = maxd > WET_M, maxd < DRY_M
    dist_wet = edt(~wet) * c if wet.any() else np.full(maxd.shape, 1e9)
    dist_dry = edt(~dry) * c

    towns = sorted(ter.towns, key=lambda t: -t["r"])
    big = towns[0] if towns else None
    big_gap, big_wet = float("nan"), 0.0
    if big is not None:
        bi = min(int(big["y"] // c), n - 1), min(int(big["x"] // c), n - 1)
        big_gap = max(0.0, float(dist_wet[bi]) - big["r"])
        yy, xx = np.mgrid[0:n, 0:n]
        in_big = np.hypot((xx + 0.5) * c - big["x"], (yy + 0.5) * c - big["y"]) <= big["r"]
        big_wet = float(wet[in_big].mean()) if in_big.any() else 0.0
    pop = ter.population
    pop_wet = float((pop * wet).sum() / max(pop.sum(), 1e-12))

    P = world.people
    ix = np.clip((P.start_x // c).astype(int), 0, n - 1)
    iy = np.clip((P.start_y // c).astype(int), 0, n - 1)
    risk = maxd[iy, ix] > WET_M
    escape_m = trapped = onset_min = climb_m = float("nan")
    if risk.any():
        esc = dist_dry[iy[risk], ix[risk]]
        series = D[:, iy[risk], ix[risk]] > WET_M
        first = np.where(series.any(axis=0), series.argmax(axis=0), len(times) - 1)
        climb = []
        for k in np.flatnonzero(risk):
            y0, x0, r = iy[k], ix[k], 12
            ys, xs = slice(max(0, y0 - r), y0 + r + 1), slice(max(0, x0 - r), x0 + r + 1)
            sub = dry[ys, xs]
            if sub.any():   # height of the nearest-lying refuge cells above the person
                climb.append(float(np.percentile(ter.z[ys, xs][sub], 10) - ter.z[y0, x0]))
        escape_m = float(np.median(esc))
        trapped = float((esc > 500).mean())
        onset_min = float(np.median(times[first] / 60.0))
        climb_m = float(np.median(climb)) if climb else float("nan")
    return {"seed": int(world.spec.seed), "flood_type": world.spec.flood_type, "n_people": int(P.n),
            "n_start_wet": int(risk.sum()), "extent_frac": float(wet.mean()),
            "big_town": big["name"] if big else "", "big_town_r_m": float(big["r"]) if big else 0.0,
            "big_town_gap_m": big_gap, "big_town_wet_frac": big_wet, "pop_wet_frac": pop_wet,
            "escape_m": escape_m, "trapped_frac": trapped, "climb_m": climb_m, "onset_min": onset_min}


FEATURE_TEXT = {   # which measured features each card shows (median over the preset's floods)
    "town_in_flood": [("big_town_wet_frac", "of the big town flooded", "pct"), ("pop_wet_frac", "of all homes in the flood", "pct")],
    "town_far": [("big_town_gap_m", "from the big town to the flood edge", "km"), ("pop_wet_frac", "of homes in the flood", "pct")],
    "escape_hills": [("escape_m", "walk to dry ground", "m"), ("climb_m", "climb to it", "m")],
    "trapped_valley": [("escape_m", "walk to dry ground", "m"), ("trapped_frac", "of people in the flood over 500 m from it", "pct")],
    "sudden_breach": [("onset_min", "until water reaches people", "min"), ("pop_wet_frac", "of homes in the flood", "pct")],
    "slow_rise": [("onset_min", "until water reaches people", "min"), ("pop_wet_frac", "of homes in the flood", "pct")],
    "few_gliders": [("n_at_risk", "people reached by the flood", "n"), ("n_drones", "gliders", "n")],
}


def _fmt(v, unit):
    if v is None or not np.isfinite(v):
        return "–"
    return {"pct": f"{100 * v:.0f}%", "km": f"{v / 1000:.1f} km", "m": f"{v:.0f} m", "min": f"{v:.0f} min",
            "n": f"{v:.0f}"}[unit]


def run_scenarios(presets, seeds, controllers=("bayes", "survival"), out_root="results/", workers=None,
                  cache="cache/", config=None, tuned=None) -> Path:
    """Run every controller on every preset x seed; measure the worlds; write episodes.jsonl + summary.json."""
    from ..config import load_config
    from ..planner.experiments import paired
    from ..runner import run_batch
    from ..scenario import ScenarioRejected, build_world, sample_spec
    t0 = time.perf_counter()
    out = Path(out_root) / f"scenarios_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config) if config else load_config()
    rows, summary = [], {"presets": {}, "controllers": list(controllers), "seeds": list(seeds)}
    ours, ref = controllers[-1], controllers[0]
    for key in presets:
        print(f"== {key}", flush=True)
        by = {}
        for c in controllers:
            by[c] = [r for r in run_batch(seeds, c, None, config, None, cache, workers,
                                          controller_kw=(tuned or {}).get(c), tag=key, preset=key)]
            rows += by[c]
        feats = {}
        for s in seeds:
            try:
                w = build_world(sample_spec(s, cfg=cfg, preset=key), cache_dir=cache, cfg=cfg)
            except ScenarioRejected:
                continue
            feats[s] = flood_features(w)
        ok = {c: {r["seed"]: r for r in by[c] if "error" not in r} for c in controllers}
        for s, f in feats.items():
            r = ok[ours].get(s) or ok[ref].get(s)
            if r:
                f["n_at_risk"], f["n_drones"] = r["n_at_risk"], r["n_drones"]
        pr = paired(list(ok[ours].values()), list(ok[ref].values()))
        med = {}
        for k in next(iter(feats.values()), {}):
            v = np.array([f[k] for f in feats.values() if isinstance(f.get(k), (int, float))], float)
            v = v[np.isfinite(v)]
            if len(v):
                med[k] = float(np.median(v))
        demo = None
        if pr.get("n"):
            d, ss = np.array(pr["diffs"]), np.array(pr["seeds"])
            nz = np.abs(d) > 1e-9
            if nz.any():
                d, ss = d[nz], ss[nz]
                k = int(np.argmin(np.abs(d - np.median(d))))
            else:   # all tied: show the flood with the most people at risk
                k = int(np.argmax([ok[ours][int(s)]["n_at_risk"] for s in ss]))
            demo = {"seed": int(ss[k]), "diff": float(d[k]), "n_non_tied": int(nz.sum())}
        summary["presets"][key] = {
            "paired": {k: v for k, v in pr.items() if k not in ("diffs",)} if pr.get("n") else {"n": 0},
            "means": {c: float(np.mean([r["survival_at_risk"] for r in ok[c].values()])) if ok[c] else None
                      for c in controllers},
            "found": {c: float(np.mean([r["found_fraction"] for r in ok[c].values()])) if ok[c] else None
                      for c in controllers},
            "n_at_risk": float(np.mean([r["n_at_risk"] for r in ok[ref].values()])) if ok[ref] else 0,
            "features_median": med, "features": {str(s): f for s, f in feats.items()}, "demo": demo,
            "errors": sum(1 for c in controllers for r in by[c] if "error" in r)}
        p = summary["presets"][key]
        if pr.get("n"):
            print(f"  {ours} - {ref}: {pr['mean']:+.3f} CI [{pr['ci95'][0]:+.3f}, {pr['ci95'][1]:+.3f}] "
                  f"wins {pr['wins']} losses {pr['losses']} ties {pr['ties']}; demo {demo}", flush=True)
        print(f"  features: " + ", ".join(f"{k} {v:.2f}" for k, v in med.items() if k in
              ("big_town_gap_m", "big_town_wet_frac", "escape_m", "climb_m", "trapped_frac", "onset_min",
               "n_at_risk")), flush=True)
    with open(out / "episodes.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=float) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    Path(out_root, "latest_scenarios.txt").write_text(str(out))
    print(f"wrote {out} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return out


def export_pages(results_dir, site="site/", cache="cache/", config=None, tuned=None, rec_out=None) -> Path:
    """One compare page per preset (its demo flood) and site/scenarios.html to switch between them."""
    from ..presets import PRESETS
    from .export import build_bundle, callouts, write_page
    from .record import record_episode
    summ = json.loads((Path(results_dir) / "summary.json").read_text())
    ref, ours = summ["controllers"][0], summ["controllers"][-1]
    site = Path(site)
    site.mkdir(parents=True, exist_ok=True)
    cards = []
    for key, ps in summ["presets"].items():
        demo = ps.get("demo")
        if not demo:
            continue
        pre = PRESETS[key]
        runs = [record_episode(demo["seed"], c, None, cache, None, config, (tuned or {}).get(c), preset=key)
                for c in (ref, ours)]
        bundles = [build_bundle(r, w) for r, w in runs]
        calls = callouts(bundles[0], bundles[1], runs[1][1])
        n = ps["paired"].get("n", 0)
        page = f"compare_{key}_{demo['seed']}.html"
        write_page(bundles, site / page, f"{pre.title}: {LABELS.get(ref, ref)} vs {LABELS.get(ours, ours)}",
                   f"scenario preset '{key}' · median flood of {n}, not best", calls)
        res = {c: r.metrics["survival_at_risk"] for c, (r, _) in zip((ref, ours), runs)}
        cards.append({"key": key, "page": page, "seed": demo["seed"], "this": res, **ps})
        print(f"  {key}: seed {demo['seed']} {ref} {res[ref]:.2f} {ours} {res[ours]:.2f} -> {page}", flush=True)
    p = write_selector(cards, site / "scenarios.html", ref, ours)
    print("wrote", p, flush=True)
    return p


def write_selector(cards: list, path, ref="bayes", ours="survival") -> Path:
    from ..presets import PRESETS
    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html_cards = []
    for i, c in enumerate(cards):
        pre = PRESETS[c["key"]]
        pr = c["paired"]
        if pr.get("n"):
            sig = pr["ci95"][0] > 0 or pr["ci95"][1] < 0
            verdict = (("ours ahead" if pr["mean"] > 0 else "Bayesian ahead") if sig else "no clear difference")
            cls = ("ahead" if pr["mean"] > 0 else "behind") if sig else "level"
            gtxt = (f"Over {pr['n']} floods: ours − Bayesian <b>{pr['mean']:+.3f}</b> "
                    f"(95% CI {pr['ci95'][0]:+.3f} to {pr['ci95'][1]:+.3f}) · {pr['wins']} wins, {pr['losses']} "
                    f"losses, {pr['ties']} ties")
        else:
            verdict, cls, gtxt = "no results", "level", ""
        m = c["means"]
        mtxt = (f"Mean survival of people at risk: ours {m[ours]:.2f}, Bayesian {m[ref]:.2f} · "
                f"{c['n_at_risk']:.1f} people at risk per flood")
        fm = c["features_median"]
        ftxt = " · ".join(f"{_fmt(fm.get(k), u)} {lab}" for k, lab, u in FEATURE_TEXT.get(c["key"], []))
        this = f"This flood (seed {c['seed']}): ours {c['this'][ours]:.2f}, Bayesian {c['this'][ref]:.2f}"
        html_cards.append(f"""<button class="card{' on' if i == 0 else ''}" data-src="{c['page']}">
<div class="t">{esc(pre.title)}<span class="v {cls}">{verdict}</span></div>
<div class="b">{esc(pre.blurb)}</div>
<div class="f">Typical flood: {esc(ftxt)}</div>
<div class="e">Expected: {esc(pre.expect)}</div>
<div class="g">{gtxt}</div>
<div class="g">{mtxt}</div>
<div class="r">{this}</div></button>""")
    first = cards[0]["page"] if cards else ""
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>FloodSim scenarios: survival-aware vs Bayesian search</title>
<style>
:root {{ color-scheme: dark; }}
body {{ margin:0; font:14px/1.4 -apple-system, system-ui, sans-serif; background:#0f1419; color:#e6edf3;
       display:grid; grid-template-columns: 400px 1fr; height:100vh; }}
aside {{ overflow-y:auto; padding:16px; border-right:1px solid #2a3440; }}
h1 {{ font-size:18px; margin:0 0 6px; }}
p.note {{ color:#9aa7b4; font-size:12px; margin:0 0 14px; }}
.card {{ display:block; width:100%; text-align:left; background:#161d26; color:inherit; border:1px solid #2a3440;
        border-radius:8px; padding:10px 12px; margin:0 0 10px; cursor:pointer; font:inherit; }}
.card:hover {{ border-color:#4a5868; }}
.card.on {{ border-color:#d9861a; box-shadow: inset 3px 0 0 #d9861a; }}
.t {{ font-weight:600; font-size:15px; display:flex; justify-content:space-between; gap:8px; }}
.v {{ font-weight:500; font-size:11px; padding:1px 7px; border-radius:9px; white-space:nowrap; align-self:center; }}
.v.ahead {{ background:#23845a33; color:#5fd39a; }} .v.behind {{ background:#5a6f8f44; color:#a9bad3; }}
.v.level {{ background:#2a3440; color:#9aa7b4; }}
.b {{ color:#c4cdd6; margin:4px 0; font-size:13px; }}
.f {{ color:#e0b46a; font-size:12px; }}
.e {{ color:#9aa7b4; font-size:12px; font-style:italic; margin-top:3px; }}
.g {{ color:#c4cdd6; font-size:12px; margin-top:3px; }}
.r {{ color:#9aa7b4; font-size:12px; margin-top:3px; }}
iframe {{ width:100%; height:100vh; border:0; background:#0f1419; }}
</style></head><body>
<aside>
<h1>Scenarios</h1>
<p class="note">Each scenario is a preset: random floods drawn from narrowed ranges (town placement, valley shape,
flood type, fleet size). Both planners fly the same floods. The badge and the grey lines are the result over all of
the scenario's floods; the replay on the right is its median flood, not the best one. The "expected" line was
written before the runs.</p>
{''.join(html_cards)}
</aside>
<iframe id="view" src="{first}" title="compare replay"></iframe>
<script>
document.querySelectorAll('.card').forEach(b => b.addEventListener('click', () => {{
  document.querySelectorAll('.card').forEach(x => x.classList.remove('on'));
  b.classList.add('on');
  document.getElementById('view').src = b.dataset.src;
}}));
</script></body></html>"""
    Path(path).write_text(html)
    return Path(path)
