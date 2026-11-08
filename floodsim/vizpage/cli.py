"""CLI for the visualisation plan: export-replay, charts, demo3d, pitch."""
from __future__ import annotations

import json
import time
from pathlib import Path


def _team(args):
    return {k: getattr(args, k, None) for k in ("mass_kg", "battery_life_min", "battery_wh", "charge_time_min",
                                                 "signal_radius_km", "camera_range_m")}


def tuned_kw(path="results/tuned_params.json") -> dict:
    p = Path(path)
    if p.exists():
        return {c: {"params": v} for c, v in json.loads(p.read_text()).items()}
    return {}


def add_cli(sub, team_args):
    p = sub.add_parser("export-replay", help="record runs and write the offline replay / compare page")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--controllers", default="bayes,survival")
    p.add_argument("--drones", type=int, default=None)
    p.add_argument("--out", default="site/")
    p.add_argument("--rec-out", default="out/")
    p.add_argument("--tuned", default="results/tuned_params.json")
    p.add_argument("--preset", default=None, help="scenario preset (see `scenarios --list`)")
    team_args(p)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("charts", help="results charts (PNGs + results.html) from a compare run")
    p.add_argument("--results", required=False, help="results/compare_<ts>/ (default: latest)")
    p.add_argument("--ablation", default=None)
    p.add_argument("--lambda-sweep", default=None)
    p.add_argument("--swarm", default=None)
    p.add_argument("--out", default="site/charts/")
    p.set_defaults(func=cmd_charts)

    p = sub.add_parser("demo3d", help="interactive 3D flight view (plotly HTML)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--controller", default="survival")
    p.add_argument("--drones", type=int, default=None)
    p.add_argument("--out", default="site/")
    p.add_argument("--tuned", default="results/tuned_params.json")
    p.add_argument("--preset", default=None)
    team_args(p)
    p.set_defaults(func=cmd_demo3d)

    p = sub.add_parser("pitch", help="backup assets: compare-view MP4/GIF and key-frame PNGs in site/pitch/")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--controllers", default="bayes,survival")
    p.add_argument("--out", default="site/pitch/")
    p.add_argument("--tuned", default="results/tuned_params.json")
    team_args(p)
    p.set_defaults(func=cmd_pitch)

    p = sub.add_parser("scenarios", help="run both planners on the scenario presets; compare pages + site/scenarios.html")
    p.add_argument("--presets", default="all", help="comma list or 'all'")
    p.add_argument("--seeds", default="1-20", help="base seeds per preset")
    p.add_argument("--controllers", default="bayes,survival", help="reference first, ours last")
    p.add_argument("--out", default="site/")
    p.add_argument("--results-root", default="results/")
    p.add_argument("--results", default=None,
                   help="reuse results/scenarios_<ts>/ runs, comma-separated, later ones win (skip the runs)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--tuned", default="results/tuned_params.json")
    p.add_argument("--list", action="store_true", help="list the presets and exit")
    team_args(p)
    p.set_defaults(func=cmd_scenarios)


def _latest(kind):
    p = Path("results/latest_runs.json")
    if p.exists():
        return json.loads(p.read_text()).get(kind)
    cands = sorted(Path("results").glob(f"{kind}_*"))
    return str(cands[-1]) if cands else None


def record_runs(seed, controllers, drones, team, cache, config, tuned, preset=None):
    from .record import record_episode
    out = []
    for c in controllers:
        res, world = record_episode(seed, c, drones, cache, team, config, controller_kw=tuned.get(c), preset=preset)
        out.append((res, world))
    return out


def cmd_export(args):
    from .export import build_bundle, bundle_json, callouts, write_page
    from .record import save_recording
    t0 = time.perf_counter()
    ctrls = args.controllers.split(",")
    runs = record_runs(args.seed, ctrls, args.drones, _team(args), args.cache, args.config, tuned_kw(args.tuned),
                       args.preset)
    bundles = []
    for res, world in runs:
        save_recording(res, world, args.rec_out)
        b = build_bundle(res, world)
        bundles.append(b)
        write_page([b], Path(args.out) / f"replay_{args.seed}_{res.controller}.html", b["meta"]["label"],
                   "single run replay")
        print(f"  {res.controller}: bundle {len(bundle_json(b)) / 1e6:.2f} MB, survival_at_risk {res.metrics['survival_at_risk']:.3f}")
    if len(bundles) == 2:
        world = runs[1][1]
        calls = callouts(bundles[0], bundles[1], world)
        sub = "median seed, not best" if _is_demo_seed(args.seed) else ""
        p = write_page(bundles, Path(args.out) / f"compare_{args.seed}.html",
                       f"{bundles[0]['meta']['label']} vs {bundles[1]['meta']['label']}: same flood, same drones, same sensors",
                       sub, calls)
        (Path(args.out) / f"callouts_{args.seed}.json").write_text(json.dumps(calls, indent=1, default=float))
        print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB) with {len(calls)} callouts in {time.perf_counter() - t0:.0f} s")
    return 0


def _is_demo_seed(seed):
    rd = _latest("compare")
    if rd and (Path(rd) / "summary.json").exists():
        return json.loads((Path(rd) / "summary.json").read_text()).get("demo_seed", {}).get("seed") == seed
    return False


def cmd_charts(args):
    from .charts import make_charts
    rd = args.results or _latest("compare")
    out = make_charts(rd, args.out, args.ablation or _latest("ablation"), args.lambda_sweep or _latest("lambda_sweep"),
                      args.swarm or _latest("swarm_size"))
    print("wrote", out, "and", Path(out).parent / "results.html")
    return 0


def cmd_demo3d(args):
    from .demo3d import demo3d
    from .record import record_episode
    res, world = record_episode(args.seed, args.controller, args.drones, args.cache, _team(args), args.config,
                                tuned_kw(args.tuned).get(args.controller), preset=args.preset)
    drape = None
    dbg = [d for d in res.recording.get("debug", []) if d and "stake" in d]
    if dbg:
        import numpy as np
        drape = np.asarray(dbg[len(dbg) // 3]["stake"], dtype=float)
        drape = np.sqrt(drape / max(drape.max(), 1e-9))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    p = demo3d(res, world, Path(args.out) / f"flight3d_{args.seed}.html", drape=drape)
    print("wrote", p, f"({Path(p).stat().st_size / 1e6:.1f} MB)")
    return 0


def cmd_pitch(args):
    from .pitch import make_pitch
    ctrls = args.controllers.split(",")
    runs = record_runs(args.seed, ctrls, None, _team(args), args.cache, args.config, tuned_kw(args.tuned))
    out = make_pitch(runs, args.out)
    print("wrote", out)
    return 0


def cmd_scenarios(args):
    from ..presets import PRESETS
    from ..runner import parse_seeds
    from .showcase import export_pages, run_scenarios
    if args.list:
        for k, p in PRESETS.items():
            print(f"{k:16s} {p.title}: {p.blurb}")
        return 0
    tuned = tuned_kw(args.tuned)
    if args.results:
        dirs = args.results.split(",")
    else:
        keys = list(PRESETS) if args.presets == "all" else args.presets.split(",")
        latest = Path(args.results_root) / "latest_scenarios.txt"
        # a run of some presets is shown together with every earlier run (one directory per line, later ones win)
        prev = latest.read_text().split() if args.presets != "all" and latest.exists() else []
        dirs = prev + [str(run_scenarios(keys, parse_seeds(args.seeds), tuple(args.controllers.split(",")),
                                         args.results_root, args.workers, args.cache, args.config, tuned))]
        latest.write_text("\n".join(dirs) + "\n")
    export_pages(dirs, args.out, args.cache, args.config, tuned)
    return 0
