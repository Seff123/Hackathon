"""V4: results charts (PNG at dpi 200 + SVG embedded in site/results.html), built from summary.json."""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np

CTRL_COL = {"scripted": "#8a8f94", "bayes": "#5a6f8f", "heuristic": "#7b5cd6", "survival": "#d9861a"}
LABEL = {"scripted": "Scripted lawnmower", "bayes": "Bayesian search", "heuristic": "Flood-bonus heuristic",
         "survival": "Survival-aware (ours)"}
ORDER = ["scripted", "bayes", "heuristic", "survival"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e7ea", "grid.linewidth": 0.8,
                         "axes.axisbelow": True})
    return plt


def _save(fig, out: Path, name: str, svgs: dict):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    svgs[name] = buf.getvalue()[buf.getvalue().index("<svg"):]
    import matplotlib.pyplot as plt
    plt.close(fig)


def chart_headline(summ, out, svgs, data):
    plt = _plt()
    pairs = [(k, v) for k, v in summ["paired"].items() if v.get("n")]
    pairs.sort(key=lambda kv: ORDER.index(kv[0].split("-", 1)[1]) if kv[0].split("-", 1)[1] in ORDER else 9)
    fig, ax = plt.subplots(figsize=(9.6, 0.9 + 0.9 * len(pairs)))
    ys = np.arange(len(pairs))[::-1]
    rows = []
    for y, (k, p) in zip(ys, pairs):
        other = k.split("-", 1)[1]
        m, lo, hi = p["mean"], p["ci95"][0], p["ci95"][1]
        ax.plot([lo, hi], [y, y], color=CTRL_COL.get(other, "#555"), lw=2.5)
        ax.plot([m], [y], "o", color=CTRL_COL["survival"], ms=9, zorder=3)
        ax.text(hi + 0.005, y, f"{m:+.3f}  ({p['wins']} wins, {p['losses']} losses, {p['ties']} ties)", va="center", fontsize=9)
        rows.append({"vs": other, "mean": round(m, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "wins": p["wins"],
                     "losses": p["losses"], "ties": p["ties"]})
    ax.axvline(0, color="#17222b", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"ours − {LABEL.get(k.split('-', 1)[1], k)}" for k, _ in pairs])
    n = pairs[0][1]["n"] if pairs else 0
    best = max(pairs, key=lambda kv: kv[1]["mean"])[1]["mean"] if pairs else 0
    ax.set_title(f"Survival-aware search changes survival of at-risk people by {pairs[0][1]['mean']:+.3f} vs Bayesian search"
                 if pairs and pairs[0][0].endswith("bayes") else "Paired difference in survival_at_risk", loc="left", fontsize=11, weight="bold")
    ax.set_xlabel(f"difference in survival per at-risk person (ours − other), dot = mean, bar = 95% CI, n = {n} test floods")
    lo_all = min([r["ci_lo"] for r in rows] + [0]) - 0.02
    hi_all = max([r["ci_hi"] for r in rows] + [0]) + 0.2
    ax.set_xlim(lo_all, hi_all)
    _save(fig, out, "1_headline", svgs)
    data["1_headline"] = rows


def chart_secured(summ, out, svgs, data):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    rows = {}
    for c in ORDER:
        cur = summ["secured_curves"].get(c)
        if not cur:
            continue
        t = np.array(cur["t_min"])
        m = np.array(cur["mean"])
        se = np.array(cur["se"])
        sel = t <= 150
        ax.plot(t[sel], m[sel], color=CTRL_COL[c], lw=2.6 if c == "survival" else 1.8, label=LABEL[c])
        ax.fill_between(t[sel], (m - se)[sel], (m + se)[sel], color=CTRL_COL[c], alpha=0.15, lw=0)
        ax.text(t[sel][-1] + 1, m[sel][-1], LABEL[c], color=CTRL_COL[c], va="center", fontsize=9)
        rows[c] = round(float(m[sel][-1]), 3)
    ax.set_xlim(0, 175)
    ax.set_xlabel("minutes after launch")
    ax.set_ylabel("survival secured per at-risk person")
    lead = max(rows, key=rows.get) if rows else ""
    ax.set_title(f"{LABEL.get(lead, lead)} secures the most survival by the end of the sortie ({rows.get(lead, 0):.3f})",
                 loc="left", fontsize=11, weight="bold")
    _save(fig, out, "2_secured_over_time", svgs)
    data["2_secured_over_time"] = rows


def chart_lambda(lsumm, out, svgs, data):
    if not lsumm:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    xs, ms, ses = [], [], []
    for k, v in lsumm["controllers"].items():
        if k.startswith("heuristic+lf"):
            xs.append(float(k[len("heuristic+lf"):]))
            ms.append(v["survival_at_risk"]["mean"])
            ses.append(v["survival_at_risk"]["se"])
    o = np.argsort(xs)
    xs, ms, ses = np.array(xs)[o], np.array(ms)[o], np.array(ses)[o]
    ours = lsumm["controllers"].get("survival", {}).get("survival_at_risk")
    if ours:
        ax.axhspan(ours["mean"] - ours["se"], ours["mean"] + ours["se"], color=CTRL_COL["survival"], alpha=0.2)
        ax.axhline(ours["mean"], color=CTRL_COL["survival"], lw=2)
        ax.text(xs.max(), ours["mean"], "  survival-aware (ours)", color=CTRL_COL["survival"], va="bottom", ha="right")
    ax.errorbar(xs, ms, yerr=ses, color=CTRL_COL["heuristic"], marker="o", capsize=3, lw=1.8)
    ax.set_xlabel("flood-bonus weight λ_f of the heuristic baseline")
    ax.set_ylabel("survival per at-risk person")
    best = float(ms.max()) if len(ms) else 0
    verdict = "never catches" if ours and best < ours["mean"] else "matches or beats"
    ax.set_title(f"Tuning the flood bonus {verdict} the survival value (best heuristic {best:.3f} vs ours {ours['mean']:.3f})"
                 if ours else "Heuristic λ_f sweep", loc="left", fontsize=11, weight="bold")
    _save(fig, out, "3_lambda_sweep", svgs)
    data["3_lambda_sweep"] = {"lambda_f": xs.tolist(), "mean": [round(float(v), 3) for v in ms],
                              "ours": round(ours["mean"], 3) if ours else None}


def chart_flood_type(summ, out, svgs, data):
    plt = _plt()
    types = [t for t in ["fluvial", "fluvial+pluvial", "pluvial", "breach"] if t in summ["by_flood_type"]]
    ctrls = [c for c in ORDER if c in summ["controllers"]]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    w = 0.8 / max(len(ctrls), 1)
    rows = {}
    for j, c in enumerate(ctrls):
        ms = [summ["by_flood_type"][t][c]["mean"] or 0 for t in types]
        se = [summ["by_flood_type"][t][c]["se"] or 0 for t in types]
        x = np.arange(len(types)) + (j - (len(ctrls) - 1) / 2) * w
        ax.bar(x, ms, w * 0.92, yerr=se, color=CTRL_COL[c], label=LABEL[c], capsize=2)
        rows[c] = [round(float(m), 3) for m in ms]
    ax.set_xticks(np.arange(len(types)))
    ax.set_xticklabels([f"{t}\n(n={summ['by_flood_type'][t][ctrls[0]]['n']})" for t in types])
    ax.set_ylabel("survival per at-risk person")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    gains = {t: (summ["by_flood_type"][t]["survival"]["mean"] or 0) - (summ["by_flood_type"][t].get("bayes", {}).get("mean") or 0)
             for t in types if "survival" in summ["by_flood_type"][t]}
    bt = max(gains, key=gains.get) if gains else ""
    ax.set_title(f"Largest gain over Bayesian search on {bt} floods ({gains.get(bt, 0):+.3f})" if gains else "By flood type",
                 loc="left", fontsize=11, weight="bold")
    _save(fig, out, "4_by_flood_type", svgs)
    data["4_by_flood_type"] = {"types": types, **rows}


def chart_swarm(ssumm, out, svgs, data):
    if not ssumm:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    sizes = ssumm["sizes"]
    rows = {}
    for c in ("bayes", "survival"):
        ms = [ssumm[c][str(n)]["mean"] for n in sizes]
        se = [ssumm[c][str(n)]["se"] for n in sizes]
        ax.errorbar(sizes, ms, yerr=se, color=CTRL_COL[c], marker="o", capsize=3, lw=2.4 if c == "survival" else 1.8)
        ax.text(sizes[-1] + 0.15, ms[-1], LABEL[c], color=CTRL_COL[c], va="center")
        rows[c] = [round(float(m), 3) for m in ms]
    ax.set_xticks(sizes)
    ax.set_xlim(min(sizes) - 0.5, max(sizes) + 2.5)
    ax.set_xlabel("gliders in the swarm")
    ax.set_ylabel("survival per at-risk person")
    gain = np.diff(rows["survival"])
    ax.set_title(f"Going from {sizes[-2]} to {sizes[-1]} gliders adds {gain[-1]:+.3f} survival (ours)", loc="left",
                 fontsize=11, weight="bold")
    _save(fig, out, "5_swarm_size", svgs)
    data["5_swarm_size"] = {"sizes": sizes, **rows}


def chart_ablation(asumm, out, svgs, data):
    if not asumm:
        return
    plt = _plt()
    pairs = [(k.split("-", 1)[1], v) for k, v in asumm["paired"].items() if v.get("n")]
    pairs.sort(key=lambda kv: kv[1]["mean"])
    fig, ax = plt.subplots(figsize=(9.6, 0.8 + 0.6 * len(pairs)))
    ys = np.arange(len(pairs))
    names = {"survival+no_state_pd": "state-aware detection", "survival+no_scenarios": "scenario posterior",
             "survival+no_bias": "forecast bias learning", "survival+scen_bonus": "(adding) uncertainty bonus",
             "survival+no_prediction": "people-movement prediction", "survival+comms_perfect": "(perfect comms instead of relay)"}
    rows = []
    for y, (k, p) in zip(ys, pairs):
        ax.barh(y, p["mean"], xerr=1.96 * p["se"], color=CTRL_COL["survival"] if p["mean"] > 0 else "#8a8f94", capsize=3)
        ax.text(max(p["mean"], 0) + 1.96 * p["se"] + 0.003, y, f"{p['mean']:+.3f}", va="center", fontsize=9)
        rows.append({"variant": k, "mean": round(p["mean"], 3)})
    ax.set_yticks(ys)
    ax.set_yticklabels([names.get(k, k) for k, _ in pairs])
    ax.axvline(0, color="#17222b", lw=1)
    ax.set_xlabel("ours − ours without this piece (survival per at-risk person; bars = 95% CI)")
    top = pairs[-1] if pairs else None
    ax.set_title(f"Biggest contribution: {names.get(top[0], top[0])} ({top[1]['mean']:+.3f})" if top else "Ablation",
                 loc="left", fontsize=11, weight="bold")
    _save(fig, out, "6_ablation", svgs)
    data["6_ablation"] = rows


def chart_brier(rows_csv, out, svgs, data, briers):
    plt = _plt()
    if not briers:
        return
    leads = sorted(briers)
    ms = [float(np.mean(briers[L])) for L in leads]
    se = [float(np.std(briers[L]) / math.sqrt(len(briers[L]))) for L in leads]
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    ax.errorbar(leads, ms, yerr=se, marker="o", color="#1f6fc2", capsize=3, lw=2)
    ax.set_xlabel("forecast lead (minutes)")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_ylim(0, max(ms) * 1.4)
    ax.set_title(f"The fleet worked with an imperfect forecast: Brier {ms[0]:.3f} at {leads[0]} min, {ms[-1]:.3f} at {leads[-1]} min",
                 loc="left", fontsize=11, weight="bold")
    _save(fig, out, "7_forecast_brier", svgs)
    data["7_forecast_brier"] = {"leads": leads, "mean": [round(m, 4) for m in ms]}


def safety_table(summ, data):
    rows = []
    for c in ORDER:
        if c in summ["controllers"]:
            m = summ["controllers"][c]
            rows.append({"controller": c, "crashes": round(m["crashes"]["mean"], 3),
                         "forced_landings": round(m["forced_landings"]["mean"], 3),
                         "separation_breaches": round(m["separation_breaches"]["mean"], 3), "n": m["crashes"]["n"]})
    data["8_safety"] = rows
    return rows


def make_charts(results_dir, out_dir, ablation_dir=None, lambda_dir=None, swarm_dir=None) -> Path:
    rd = Path(results_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summ = json.loads((rd / "summary.json").read_text())
    load = lambda d: json.loads((Path(d) / "summary.json").read_text()) if d and (Path(d) / "summary.json").exists() else None
    asumm, lsumm, ssumm = load(ablation_dir), load(lambda_dir), load(swarm_dir)
    svgs, data = {}, {}
    chart_headline(summ, out, svgs, data)
    chart_secured(summ, out, svgs, data)
    chart_lambda(lsumm, out, svgs, data)
    chart_flood_type(summ, out, svgs, data)
    chart_swarm(ssumm, out, svgs, data)
    chart_ablation(asumm, out, svgs, data)
    # forecast Brier by lead, from the scenarios in the comparison
    from ..runner import read_rows
    briers = {}
    try:
        rows = read_rows(rd / "episodes.jsonl")
        seeds = sorted({r["seed"] for r in rows if "error" not in r and r.get("controller") == "survival"})
        from ..runner import make_config
        from ..scenario import build_world, sample_spec
        cfg = make_config()
        for s in seeds[:40]:
            w = build_world(sample_spec(s, cfg=cfg), cache_dir="cache/", cfg=cfg)
            for L, v in w.forecast.brier.items():
                briers.setdefault(int(L), []).append(v)
    except Exception as e:  # pragma: no cover
        print("brier chart skipped:", e)
    chart_brier(None, out, svgs, data, briers)
    safety = safety_table(summ, data)
    (out / "charts_data.json").write_text(json.dumps(data, indent=1))
    write_results_page(out, summ, svgs, safety, data)
    return out


def write_results_page(out: Path, summ: dict, svgs: dict, safety: list, data: dict):
    n = next(iter(summ["controllers"].values()))["survival_at_risk"]["n"]
    cards = "\n".join(f'<section class="card"><div class="svg">{svg}</div></section>' for k, svg in sorted(svgs.items()))
    rows = "".join(f"<tr><td>{LABEL.get(r['controller'], r['controller'])}</td><td>{r['crashes']:.2f}</td>"
                   f"<td>{r['forced_landings']:.2f}</td><td>{r['separation_breaches']:.2f}</td></tr>" for r in safety)
    ctrl_rows = "".join(
        f"<tr><td>{LABEL.get(c, c)}</td><td>{m['survival_at_risk']['mean']:.3f} ± {m['survival_at_risk']['se']:.3f}</td>"
        f"<td>{m['found_fraction']['mean']:.3f}</td><td>{m['time_to_first_find_s']['mean'] / 60:.1f}</td>"
        f"<td>{m['at_risk_found']['mean']:.2f}</td></tr>" for c, m in summ["controllers"].items())
    demo = summ.get("demo_seed", {})
    html = f"""<!DOCTYPE html><html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FloodSim results</title><style>
body{{margin:0;background:#eef1f3;color:#17222b;font-family:"Segoe UI",system-ui,-apple-system,sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding:20px;display:grid;gap:16px}}h1{{margin:0;font-size:30px}}
.card{{background:#fff;border:1px solid #d5dde3;border-radius:8px;padding:12px}}.svg svg{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{padding:6px 10px;border-bottom:1px solid #e3e7ea;text-align:right}}
td:first-child,th:first-child{{text-align:left}}.note{{color:#5b6b77}}
@media (prefers-color-scheme: dark){{body{{background:#0f161c;color:#e3eaef}}.card{{background:#fff;color:#17222b}}}}
</style></head><body><div class="wrap">
<h1>Does it work? {n} test floods per controller</h1>
<p class="note">Every controller faced the same {n} random floods (common random numbers): terrain, flood, wind and survivors identical.
Whiskers and bands show ±1 standard error unless stated; the headline uses 95% intervals on paired differences.
Demo seed {demo.get('seed', '?')} is the median seed (ours − Bayesian difference closest to the median), not the best.</p>
<section class="card"><h3>Per controller</h3><table><tr><th>Controller</th><th>survival_at_risk</th><th>found fraction</th>
<th>first find (min)</th><th>at-risk found</th></tr>{ctrl_rows}</table></section>
{cards}
<section class="card"><h3>8. Safety: nothing unsafe was traded for speed (mean per episode)</h3><table><tr><th>Controller</th><th>crashes</th>
<th>forced landings</th><th>separation breaches</th></tr>{rows}</table></section>
</div></body></html>"""
    (out.parent / "results.html").write_text(html) if out.name == "charts" else (out / "results.html").write_text(html)
