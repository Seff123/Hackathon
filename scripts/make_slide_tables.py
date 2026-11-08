"""Slide-ready PNG tables (16:9, 200 dpi) in site/slides/:
1_scenarios.png      survival-aware vs Bayesian search across the scenario presets
2_glider_specs.png   today's glider vs the 2076 glider: what changes
3_glider_results.png today's vs 2076 glider on the crowded far-town scenario
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "site" / "slides"

INK, MUTED, RULE, HEAD_BG = "#1b2430", "#5b6673", "#d5dae0", "#eef1f4"
GOOD, GOOD_BG, OURS = "#1d7a4f", "#e3f3ea", "#c46f0f"


def table_png(path, title, subtitle, header, rows, col_w, align=None, highlight=(), bold_cols=(), note="",
              row_h=0.62, size=(13.33, 7.5)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=size, dpi=200)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, size[0]); ax.set_ylim(0, size[1]); ax.axis("off")
    x0, top = 0.55, size[1] - 0.5
    ax.text(x0, top, title, fontsize=24, weight="bold", color=INK, va="top")
    ax.text(x0, top - 0.62, subtitle, fontsize=13, color=MUTED, va="top")
    total = sum(col_w)
    scale = (size[0] - 2 * x0) / total
    xs = np.cumsum([0] + [w * scale for w in col_w]) + x0
    align = align or ["left"] + ["right"] * (len(header) - 1)
    y = top - 1.35
    head_h = row_h * (1.0 if not any("\n" in h for h in header) else 1.45)
    bottom = 0.95 if note else 0.4          # keep the footnote clear of the table
    row_h = min(row_h, (y - head_h - bottom) / max(len(rows), 1))
    ax.add_patch(plt.Rectangle((x0, y - head_h), xs[-1] - x0, head_h, color=HEAD_BG, lw=0))
    for j, h in enumerate(header):
        tx = xs[j] + 0.12 if align[j] == "left" else xs[j + 1] - 0.12 if align[j] == "right" else (xs[j] + xs[j + 1]) / 2
        ax.text(tx, y - head_h / 2, h, fontsize=12.5, weight="bold", color=INK, va="center", ha=align[j],
                multialignment=align[j] if align[j] != "center" else "center", linespacing=1.3)
    y -= head_h
    for i, r in enumerate(rows):
        if i in highlight:
            ax.add_patch(plt.Rectangle((x0, y - row_h), xs[-1] - x0, row_h, color=GOOD_BG, lw=0))
        for j, cell in enumerate(r):
            colour = INK
            if isinstance(cell, tuple):
                cell, colour = cell
            tx = xs[j] + 0.12 if align[j] == "left" else xs[j + 1] - 0.12 if align[j] == "right" else (xs[j] + xs[j + 1]) / 2
            ax.text(tx, y - row_h / 2, cell, fontsize=13, color=colour, va="center", ha=align[j],
                    weight="bold" if (j in bold_cols or i in highlight and j == 0) else "normal")
        y -= row_h
        ax.plot([x0, xs[-1]], [y, y], color=RULE, lw=0.8)
    if note:
        ax.text(x0, 0.3, note, fontsize=10.5, color=MUTED, va="bottom", wrap=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def _ci(p):
    sig = p["ci95"][0] > 0 or p["ci95"][1] < 0
    txt = f"{p['mean']:+.3f}  [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]"
    return (txt, GOOD if sig and p["mean"] > 0 else INK), sig and p["mean"] > 0


def scenarios():
    from floodsim.presets import get_preset
    dirs = (ROOT / "results" / "latest_scenarios.txt").read_text().split()
    merged = {}
    for d in dirs:
        merged.update(json.loads((ROOT / d / "summary.json").read_text())["presets"])
    from floodsim.presets import PRESETS
    order = [v for k in PRESETS for v in [k] + sorted(x for x in merged if x.startswith(k + "+"))]
    rows, hl = [], []
    for key in [k for k in order if k in merged]:
        s = merged[key]
        p = s["paired"]
        if not p.get("n"):
            continue
        cell, sig = _ci(p)
        if sig:
            hl.append(len(rows))
        title = get_preset(key).title.replace("(crowded: 50-60 people)", "· 50-60 people")
        rows.append([title, f"{s['n_at_risk']:.1f}", f"{s['means']['bayes']:.2f}", (f"{s['means']['survival']:.2f}", OURS),
                     cell, f"{p['wins']} / {p['losses']} / {p['ties']}", str(p["n"])])
    table_png(OUT / "1_scenarios.png", "Where survival-aware search beats Bayesian search",
              "Survival of people at risk; both planners fly the same floods with the same gliders",
              ["Scenario", "At risk\nper flood", "Survival\nBayesian", "Survival\nours", "Ours − Bayesian\n[95% CI]", "Wins / losses\n/ ties", "Floods"],
              rows, [4.7, 1.4, 1.3, 1.1, 3.1, 1.8, 1.0], highlight=hl, row_h=0.5,
              note="Green: 95% confidence interval excludes zero. Today's glider (5 kg). 20 floods per scenario "
                   "(80 seeds for the far-town 50-60 people row; seeds failing the scenario's checks are skipped).")


def glider_specs():
    rows = [
        ["Battery energy", "149 Wh", "450 Wh", "Li-ion ~3.4× since 1991 (~4%/yr); assume rate halves"],
        ["Endurance (level, 13 m/s)", "2.5 h", "11.7 h", "Battery × better airframe"],
        ["Power in level flight", "60 W", "39 W", "Lower drag, prop 65% → 80%, avionics 25 → 15 W"],
        ["Thermal camera range", "140 m", "280 m", "Larger sensor arrays (pixel pitch 50 → 12 µm so far)"],
        ["Search ceiling", "400 m", "560 m", "Follows camera range"],
        ["Radio range", "3 km", "15 km", "Satellite direct-to-device + mesh: whole area"],
        ["False alarms per cell", "0.002", "0.0005", "Onboard recognition"],
        ["Confirmation reliability", "95%", "99%", "Onboard recognition"],
        ["Battery swap / charge", "10 min", "5 min", "Faster swaps and charging"],
        ["Mass, size", "1.5 kg, 3.7 m span", "same", "Same airframe"],
    ]
    rows = [[r[0], r[1], (r[2], OURS), (r[3], MUTED)] for r in rows]
    table_png(OUT / "2_glider_specs.png", "Today's glider vs a 2076 glider",
              "Each change extrapolates a historical trend to 2076, slowed where physics sets limits",
              ["", "Today", "2076", "Basis"], rows, [3.3, 2.0, 1.6, 6.0],
              align=["left", "right", "right", "left"], row_h=0.5, bold_cols=(0,),
              note="Camera, recognition and airframe values are judgement calls rather than fitted trends.")


def glider_results():
    from floodsim.planner.experiments import paired

    def rows_of(root):
        d = Path((ROOT / root / "latest_scenarios.txt").read_text().split()[-1])
        return [json.loads(x) for x in open(ROOT / d / "episodes.jsonl") if '"error"' not in x]
    R = {"Today (1.5 kg)": rows_of("results/glider_today15"), "2076": rows_of("results/glider2076")}
    g = lambda k, c: [r for r in R[k] if r["controller"] == c]
    m = lambda k, c, f: float(np.mean([r[f] for r in g(k, c)]))
    rows = []
    for k in R:
        p = paired(g(k, "survival"), g(k, "bayes"))
        rows.append([k, f"{m(k, 'bayes', 'survival_at_risk'):.2f}", (f"{m(k, 'survival', 'survival_at_risk'):.2f}", OURS),
                     f"{m(k, 'bayes', 'at_risk_found'):.2f}", (f"{m(k, 'survival', 'at_risk_found'):.2f}", OURS),
                     f"{m(k, 'survival', 'area_observed_km2'):.0f} km²", f"{p['mean']:+.2f}"])
    pb, ps = paired(g("2076", "bayes"), g("Today (1.5 kg)", "bayes")), paired(g("2076", "survival"), g("Today (1.5 kg)", "survival"))
    rows.append([("2076 gain", GOOD), f"{pb['mean']:+.2f}", (f"{ps['mean']:+.2f}", GOOD),
                 f"{m('2076', 'bayes', 'at_risk_found') - m('Today (1.5 kg)', 'bayes', 'at_risk_found'):+.2f}",
                 (f"{m('2076', 'survival', 'at_risk_found') - m('Today (1.5 kg)', 'survival', 'at_risk_found'):+.2f}", GOOD),
                 "", ""])
    n = len(g("2076", "survival"))
    table_png(OUT / "3_glider_results.png", "A better glider helps only if it searches the right place",
              f"Big town far from the flood, 50-60 people · {n} floods, same floods for both gliders",
              ["Glider", "Survival\nBayesian", "Survival\nours", "At-risk found\nBayesian", "At-risk found\nours",
               "Area searched\nours", "Survival\nours − Bayesian"],
              rows, [1.8, 1.7, 1.6, 2.0, 1.9, 2.1, 2.5], highlight=(len(rows) - 1,), row_h=0.75,
              note=f"Survival of people at risk (0-1); at-risk people found per flood. 2076 gain for ours: "
                   f"{ps['mean']:+.3f} [95% CI {ps['ci95'][0]:+.3f}, {ps['ci95'][1]:+.3f}], {ps['wins']} wins / "
                   f"{ps['losses']} losses; for Bayesian: {pb['mean']:+.3f} [{pb['ci95'][0]:+.3f}, {pb['ci95'][1]:+.3f}]. "
                   f"Planner settings tuned on today's glider.")


if __name__ == "__main__":
    scenarios()
    glider_specs()
    glider_results()
