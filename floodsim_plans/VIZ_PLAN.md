# FloodSim visualisation build plan: replay, compare, results, 3D

Nov 8, 2026 · companion to the environment plan and `PLANNER_PLAN.md`

## 0. How Claude Code should use this plan

Build after planner Phase P5 works (the planner must produce `debug_snapshot`s and a decision log). Earlier phases can be built against the scripted controller.
- **This replaces the environment plan's section 16 as the demo deliverable.** Keep the environment's `plot_world`, `plot_coverage` and `plot_wind_slice` PNGs for debugging. Keep `animate_episode` (GIF) as the backup clip.
- Same working rules: config for every number, tests per phase, `PROGRESS.md`, commit `viz phase N: <name>`.

**What the pitch needs, in priority order**
1. **Compare replay** (the hero demo): our planner and standard Bayesian search on the same flood, side by side, on one clock, with a "survival secured over time" race line underneath.
2. **Results charts**: the evidence that it wasn't one lucky seed (100 test floods).
3. **3D view**: terrain, water and glider paths in 3D, for the opening or Q&A.
4. **Backup clip**: a 40 s screen recording of item 1, plus the environment's GIF, in case the laptop or browser misbehaves on stage.

**Definition of done**
1. `python -m floodsim export-replay --seed <demo> --controllers bayes,survival --out site/` writes `site/compare_<seed>.html`. It opens offline with a double-click, with no server.
2. `python -m floodsim charts --results results/compare_<ts>/ --out site/charts/` writes the PNGs in section 6 and `site/results.html`.
3. `python -m floodsim demo3d --seed <demo> --controller survival --out site/` writes `site/flight3d_<seed>.html`.
4. Viz tests pass (section 8).

---

## 1. Design rules

- **One visual language everywhere** (page, PNGs, 3D):

| Thing | Colour |
|---|---|
| Flood depth (truth) | Blues, alpha by depth (0 → 3 m) |
| Forecast waterline (+20 min, fleet's belief) | Dashed blue contour |
| Lives at stake (planner value) | Amber heat |
| Where people may still be (belief λ̄) | Purple heat (toggle, never both heats at once) |
| Area searched | Faint magenta veil |
| Gliders | One hue per drone (8 fixed hues); mode by line style: solid search, dashed transit, dotted thermal, thick verify, grey RTB |
| Confirmed survivor | Green dot with ring |
| Hidden truth (people not yet found) | Hollow markers, red if at risk, grey if safe; shown only with "Show hidden truth" |
| Controllers in charts | scripted grey, bayes slate blue, heuristic violet, **survival (ours) amber** |

- **Truth vs belief is always labelled.**
  - Anything from ground truth carries a small "truth, hidden from the fleet" tag.
  - Anything from the planner is labelled "fleet's belief".
  - Judges must never confuse what the drones knew with what was really there.
- **Orientation:** the environment uses x east, y north, origin south-west, grids `[iy, ix]` with row 0 in the **south**. Canvas row 0 is the top, so flip once in one helper (`gridToCanvas`) and test it (section 8).
- **Complete at rest:** every page opens on an informative moment (the first confirmed find in a flooding street, or minute 40), not on an empty map at t = 0.
- Respect `prefers-reduced-motion` (no autoplay). Every control is keyboard-reachable; Space plays or pauses, arrow keys step 30 s.

---

## 2. Recording hooks (phase V0)

Recording runs only when `SimConfig(record_grids=True)`, never in fast mode or tuning.

| Source | Every | Fields |
|---|---|---|
| Environment truth | 30 s | true depth (100 m grid), true hazard rating class (0–3), people truth (id, x, y, state, found, current P_surv, at_risk flag), drones (x, y, z, E, mode, goal type and target, RTB energy threshold E_return / E_cap) |
| Environment events | as they happen | detection (true/false), verification (confirm/reject), find, RTB trigger, landing, swap, relaunch, crash, separation breach |
| Planner `debug_snapshot()` (PLANNER_PLAN §10) | 30 s | λ̄, lives-at-stake grid, P_flood at +20 min (bias-corrected), component weights ŵ, count-scale posterior mean, forecast bias b̂, fleet wind estimate, active claims |
| Planner decision log | each decision | time, drone, goal, block, height, expected lives at stake, travel min, lead to danger, reason string |

Write raw recordings to `out/rec_<seed>_<controller>.npz` plus `events.jsonl`.

---

## 3. Replay bundle format (phase V1, `viz/export.py`)

One JSON object per run, embedded in the HTML by the build step. Embedding (rather than `fetch`) means the page works from `file://` with no server.

```jsonc
{
  "meta": { "seed": 20031, "controller": "survival", "label": "Survival-aware (ours)",
            "flood_type": "fluvial", "n_drones": 5, "wind_ms": 6.2, "wind_dir_deg": 240,
            "episode_s": 9000, "frame_dt_s": 30, "grid": {"nx": 100, "ny": 100, "cell_m": 100},
            "camera_range_m": 140, "metrics": { /* EpisodeResult metrics */ } },
  "static": { "elev_u8": "<b64 deflate>", "elev_min": 12.0, "elev_max": 210.0,
              "landcover_u8": "<b64>", "river_u8": "<b64>",
              "roads": [[[x,y],...], ...], "towns": [["Lowmead", x, y, r], ...],
              "base": [x, y], "thermals": [[x, y, r, w], ...], "levees": [[[x,y],...], ...] },
  "grids": {            // [T, ny, nx] uint8, deflate + base64, row 0 = south
    "depth":   {"data": "<b64>", "scale": "linear", "max": 3.0},
    "lam":     {"data": "<b64>", "scale": "sqrt",   "max": 0.21},
    "stake":   {"data": "<b64>", "scale": "sqrt",   "max": 0.09},
    "pflood20":{"data": "<b64>", "scale": "linear", "max": 1.0},
    "seen":    {"data": "<b64>", "scale": "count",  "max": 3} },
  "frames": [ { "t": 0, "drones": [[x, y, z, Efrac, "search", {"type":"Search","block":[12,7],"h":84}, resfrac], ...],
                "kpi": {"secured": 0.0, "found": 0, "at_risk_found": 0, "airborne": 5, "lost": 0},
                "belief": {"w": [..6..], "count_scale": 1.0, "bias_m": 0.0, "wind": [u, v, sd]} }, ... ],
  "people": [ { "id": 3, "cat": "home", "at_risk": true, "t_found": 2310, "surv_final": 0.91,
                "track": [[t, x, y, "roof"], ...] } ],          // truth, keyframes only when state/position changes
  "events": [ {"t": 1830, "drone": 2, "kind": "decision", "text": "G3 → block (12,7) at 70 m: 0.42 expected lives at stake, water dangerous in ~18 min, 4 min away"}, ... ]
}
```

- **Quantisation:** uint8 with `sqrt` scaling for λ̄ and stake (per-run max), linear for depth (0–3 m) and P_flood.
- **Size budget:** ≤ 4 MB per run (300 frames × 5 grids × 10,000 cells ≈ 15 MB raw, about 2–3 MB deflated). If over budget, drop `pflood20` to every 60 s.
- **Decompression in the browser:** `DecompressionStream('deflate')` (zlib format; Python `zlib.compress`).
- `export-replay` builds one HTML per call: the template with every bundle inlined in `<script type="application/json">` blocks.

---

## 4. Replay page (phase V2: single run)

**Start from `page_template.html` in `floodline_code.zip`.** It already has the canvas map, play/scrub, layer toggles, fleet battery bars with a reserve tick, KPI tiles, belief bars, wind compass, decision log, light/dark themes and SVG result charts. About 60% of this phase is renaming and re-wiring.

**Layout** (desktop, wraps to one column under 900 px)

```
┌──────────────────────────────────────────────┬───────────────────────────────┐
│ Title · seed · flood type · drones · wind     │ KPI tiles: survival secured · │
├──────────────────────────────────────────────┤ people confirmed · at-risk    │
│                                              │ found · airborne / lost       │
│        MAP (square canvas, 100×100 cells)     ├───────────────────────────────┤
│  terrain hillshade · roads · towns · base     │ Fleet: per drone  mode pill,  │
│  flood (truth) · forecast waterline · heat    │ battery bar + RTB tick, goal  │
│  searched veil · gliders + trails + targets   ├───────────────────────────────┤
│  confirmed finds · (truth people toggle)      │ Fleet's belief: where people  │
│                                              │ went (6 bars) · head-count    │
├──────────────────────────────────────────────┤ estimate · forecast bias ·    │
│ ▶ 0:40:30 ───────●──────────── speed 1× 2× 4× │ wind (compass, est vs true)   │
│ Layers: [Lives at stake] [Where people may be]├───────────────────────────────┤
│ [Off]  ☐ Show hidden truth  ☐ Forecast line   │ Decisions (last 8, with why)  │
└──────────────────────────────────────────────┴───────────────────────────────┘
```

**Changes from the template:**
- grid 100 × 100 (was 60 × 60);
- up to 8 drones;
- terrain hillshade from `elev_u8` plus land-cover tint (forest darker, urban warmer);
- roads, levees and town labels from vectors;
- thermals as dotted circles;
- people truth uses state glyphs (■ home, ▲ roof, ◆ vehicle, ● outdoor, ○ in water);
- the rail's "How fast is the river rising?" becomes **"Is the forecast right?"**: the b̂ bar ("forecast running 0.2 m low");
- "Where did people go?" shows the 6 component weights plus "head count ≈ 1.2× expected";
- the fleet tick is the autopilot's **E_return** (the RTB trigger), labelled "energy needed to get home".

**Interactions:**
- hover a cell: tooltip with depth (truth), P_flood +20 (belief), λ̄ and value;
- click a drone: highlight its trail and filter the decision log to it;
- click a log entry: jump the clock there.

---

## 5. Compare page (phase V3: the hero demo)

**Two runs of the same seed** (bayes vs survival by default; any two controllers via `--controllers`).

```
┌───────────────────── Standard Bayesian search ─────┬──────── Survival-aware (ours) ─────────┐
│              map A (same layers, same clock)        │              map B                       │
├─────────────────────────────────────────────────────┴──────────────────────────────────────────┤
│ Survival secured over time   ── bayes   ── ours    │ clock cursor │ markers: first arrival per town │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Callouts (auto):  "Ours reached Lowmead 14 min earlier" · "At-risk people found: 7 vs 4"         │
│                   "Ours skipped Hillcrest until minute 62 (dry ground, nobody at risk)"           │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **One clock, one play button,** both maps redrawn each frame. On narrow screens, an A/B toggle instead of side by side.
- **Race line:** cumulative frozen survival of confirmed at-risk people (the metric the judges should remember), with the current time as a vertical cursor.
- **Auto callouts,** computed in `export.py` (not the browser):
  - first time a drone of each run searched inside each town's radius below 0.8R;
  - at-risk people found by each run;
  - final `survival_at_risk` for each;
  - minutes each run spent over dry ground (P_flood < 0.05) in the first hour.

  Only show a callout when it favours either side by a meaningful margin. If Bayesian wins something, show that too: honesty reads well.
- **Heat layer default:** "Lives at stake" on both maps, computed from **our planner's** value function for both runs, so judges see Bayesian search flying past amber areas. Label it "lives at stake (as estimated by our planner)".
- **Demo seed:** the test seed whose ours-minus-bayes difference is closest to the **median** (PLANNER_PLAN §13). Put "median seed, not best" in the subtitle.

**Script for the 40 s recording** (use the playback speed control):
1. 0–5 s: pause at minute 5. "Same flood, same drones, same sensors. Left: standard search. Right: ours."
2. 5–20 s: play at 4×. Watch ours fly to the streets the water reaches next while Bayesian searches the big town on the hill.
3. 20–30 s: pause on the first callout. "Ours reached Lowmead 14 minutes earlier, just as people climbed onto roofs and became visible."
4. 30–40 s: the race line at the end, and the final numbers.

---

## 6. Results charts (phase V4, `viz/charts.py`)

**Inputs:** `results/<run>/summary.json` and `episodes.csv` from PLANNER_PLAN §13. Every chart is written as a 1920 × 1080-friendly PNG (dpi 200) **and** embedded in `site/results.html` as SVG, built from the same JSON, reusing the template's `barChart`.

**Chart titles state the finding with its number;** subtitles state n and what the bars mean.

| # | Chart | Form | Purpose |
|---|---|---|---|
| 1 | **Headline:** paired difference in survival_at_risk, ours − each other controller, 100 test seeds | Dot + 95% CI whisker per comparison, zero line marked, win/loss counts in the labels | The one number |
| 2 | Survival secured over time, 4 controllers | Mean line + SE band, minutes on x | Shows *when* ours pulls ahead |
| 3 | λ_f sweep of the heuristic baseline | Line with SE whiskers, ours as a horizontal band | "Tuning the bonus never catches the survival value" (or honestly, where it does) |
| 4 | By flood type | Grouped bars, 4 controllers × {fluvial, fluvial + pluvial, pluvial, breach} | Where it helps most |
| 5 | Swarm size | Line, bayes vs ours, n = 2/4/6/8 | Where adding drones stops paying off |
| 6 | Ablation | Horizontal bars: ours minus each piece | What each idea contributes (state-aware detection, scenarios, bias learning) |
| 7 | Forecast Brier score by lead | Line | "The fleet worked with an imperfect forecast" |
| 8 | Safety | Small table: crashes, forced landings, separation breaches per controller | Nothing unsafe was traded for speed |

**Style:**
- one sans font;
- light grid;
- no chart junk;
- colour by controller (section 1);
- direct labels instead of legends where possible;
- units on every axis ("survival per at-risk person", "minutes after launch");
- y-axes for differences include 0.

**Check:** a test asserts every plotted number equals `summary.json` (section 8).

---

## 7. 3D view (phase V5, `viz/demo3d.py`)

Extend the environment's `plot3d_html` (plotly, offline HTML):
- Terrain surface from the 100 m elevation grid, coloured by land cover with hillshade. Vertical exaggeration × 3, adjustable.
- Water surface (terrain + true depth, NaN where dry), translucent blue, at **6 times** switched by plotly buttons. 6, not 300 frames, to keep the file under 10 MB.
- Drone paths in 3D, one hue each, dash style by mode. Confirmed finds as green markers. Base as a cone.
- Thermals as translucent cylinders up to z_i.
- **Optional drape:** colour the terrain by "lives at stake" at the chosen time instead of land cover.
- Camera presets: "overview", "follow G1", "valley floor".
- **Fallback:** matplotlib 3D PNG (environment plan).
- **Stretch only:** a three.js version with animated water and moving gliders. Skip unless everything else is done.

---

## 8. Tests (`tests/viz/`)

| Test | Pass criterion |
|---|---|
| Bundle shape | Frame count = episode_s / 30 + 1; every grid decodes to [T, 100, 100]; no NaN; quantisation max ≥ data max |
| Size | Each run bundle ≤ 4 MB; compare HTML ≤ 10 MB; 3D HTML ≤ 10 MB |
| Orientation | A synthetic grid with one hot cell at (iy = 5, ix = 80) renders as a hot pixel in the **lower right** of the canvas (pixel check via a headless render if available, else check `gridToCanvas`) |
| Truth labelling | Every truth layer toggle and tooltip carries the "truth" tag (DOM check) |
| Callouts | Computed callouts match a hand calculation on one seed |
| Charts | Every plotted value equals `summary.json` to 3 dp |
| Smoke | If Playwright/Chrome is available: page loads with no console errors, one screenshot saved to `site/smoke.png`. Otherwise open by hand once. |
| Determinism | Same recording gives a byte-identical bundle |

---

## 9. Build phases (about 3 h; core V0–V4 about 2.5 h)

| Phase | Budget | Build | Done when | Fallback |
|---|---|---|---|---|
| V0. Recording hooks | 20 min | Section 2 | `rec_*.npz` + `events.jsonl` for one seed with both controllers | Truth-only recording (no planner layers) |
| V1. Export bundle | 30 min | Section 3 | Bundle tests pass | Every 60 s instead of 30 s |
| V2. Replay page | 45 min | Section 4 from the template | Opens offline; layers, play, log work | Fewer rail panels (KPIs + fleet + log) |
| V3. Compare page | 40 min | Section 5 | Compare HTML for the demo seed; callouts appear | A/B toggle instead of side by side |
| V4. Charts | 40 min | Section 6, charts 1, 2, 4, 6, 8 first | PNGs + results page; chart test passes | PNGs only (no results page) |
| V5. 3D | 30 min | Section 7 | `flight3d_<seed>.html` opens | Environment's matplotlib 3D PNG |
| V6. Pitch assets | 15 min | 40 s recording (QuickTime/OBS), backup GIF, key-frame PNGs for slides | Files in `site/pitch/` | GIF only |

**Order inside the schedule** (PLANNER_PLAN §16): V0–V1 can start as soon as planner P5 runs, while the `compare` experiment computes. V2–V3 come next, V4 once `summary.json` exists, then V5–V6.

**Cut order:** V5 first, then charts 3/5/7, then the results page (keep the PNGs). **Never cut V3 or the headline chart.**

**Sharing:** the compare page and results page are single self-contained HTML files. To give judges a link, send the files to Claude and it can publish them as a private web page you can share. Otherwise present from the laptop.
