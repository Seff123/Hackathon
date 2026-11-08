# Progress

## Phase 0: scaffold
Package skeleton, all config dataclasses (`floodsim/config.py`) with every plan number as a default, named RNG streams, `configs/default.toml` with the team inputs, CLI stub. `pytest` runs.

## Phase 1: terrain and hydrology
Spectral terrain with a carved meandering valley, towns, roads, levees, land cover; priority-flood, D8, accumulation, HAND. Scenario sampler with the six rejection checks (cheap HAND/pond estimate). Terrain builds in ~0.08 s; specs need ~5 attempts on average.

## Phase 2: flood truth
Local-inertial solver (100 m), hydrograph, moving storm, levee breach, float16 cache (~1-3 MB). ~1 s per scenario, mass residual ~1e-12. 47/50 seeds have peak extent within 3-40%.

## Phase 3: forecast
Precomputed noisy forecast arrays (stage at drainage cells with AR(1) along-river error + bias, observed extent, velocity, rain). Brier at 30 min ≈ 0.06.

## Phase 4: weather
Log wind profile, ridge lift from smoothed terrain gradient, fixed thermal hotspots (Allen-style profile), storm rain.

## Phase 5: glider, autopilot, safety
Drag-polar sink table, vectorised glider step, battery from battery life at 13 m/s. Autopilot modes (transit, search lawnmower, thermal, verify orbit, RTB with energy rule and landing orbit) plus safety layer (terrain look-ahead, geofence, stall, ceiling) and separation. Glider/autopilot tests pass (crosswind transit, thermal climb, terrain safety, RTB ≥ reserve in ≥ 95%, landing, base zone).

## Phase 6: people, sensor, radio
Behaviour state machine with danger time and survival; nadir LWIR camera with Johnson-style detection, false alarms and verification; range-only radio. Sensor tests reproduce the p_d table and camera-range calibration.

## Phase 7: sim loop, scripted controller, metrics
Sim with information boundary (PlannerView whitelist), battery swap and relaunch, ScriptedSweep (equal-area strips, verifies every detection), metrics. 8-drone episode ≈ 4 s fast mode. 50 random 8-drone runs: no separation breaches.

## Phase 8: visualisation and demo
plot_world, plot_coverage, plot_wind_slice, GIF (+MP4) animation, plotly 3D HTML; CLI scenario/build-cache/demo/evaluate.

## Planner P0: hooks
`observe()` hook, n_hat/horizon in the view, extra static maps and forecast accessors; information-boundary test still passes for planner runs.

## Planner P1: detection
State-aware p̂_d per component (home/home_low inside vs roof, road vehicle, outdoor vs in water). Dry town at 84 m: 0.010; flooded: 0.50.

## Planner P2: belief
Six component intensity grids, revisit windows, candidates with P_real, fleet brain ingest with batched p̂_d.

## Planner P3: scenarios
630-scenario posterior from per-component accumulators (Appendix B); synthetic recovery test passes.

## Planner P4: urgency
Marginal-survival value tables V[class, lead, cell] per 5-min bucket; value-shape test reproduces the plan's numbers. Forecast gained an extent-derived depth.

## Planner P5: scoring and controllers
Block tables B[h, lead, block], travel/search time, energy feasibility, claims, spread, hysteresis, verify policy, decision log; `bayes`, `heuristic`, `survival` controllers. Equivalence, claims and feasibility tests pass.

## Planner P6: compare (and P9 tooling)
`compare`, `ablation`, `lambda-sweep`, `swarm-size` and `tune` commands; `summary.json` with mean ± SE, paired differences (CI, wins/losses), splits by flood type and swarm size, secured-survival curves, and the median demo seed. `scripts/run_experiments.py` runs the whole pipeline (tune → compare once on test seeds → ablation → sweeps).

## Planner P7: bias learning / P8: prediction and base_relay comms
Implemented with P2-P4: bias posterior from observed extents (test: a +0.3 m biased forecast is learned as b̂ ≈ −0.3 m), mass-conservative prediction of the open component, base_relay buffering of frames from drones without a radio path to base.

## Viz V0-V6
Recording hooks (rec_*.npz + events jsonl), deflated uint8 replay bundles (~1 MB per run), one offline HTML page for single replays and side-by-side compares (race line, auto callouts, truth/belief labels, keyboard controls), results charts (PNG + results.html), plotly 3D view with 6 water times and camera presets, pitch assets (compare clip MP4/GIF + key frames). Viz tests pass, including a headless-Chrome smoke screenshot.

## Planner P9: results
Pipeline (`scripts/run_experiments.py --trials 10`, 104 min): tuned bayes / heuristic / survival on train seeds 0-19 (val 10000-10019), then compare once on test seeds 20000-20099 (97 accepted). Survival-aware vs Bayesian search +0.001 (CI −0.005, +0.008; 8 wins, 6 losses, 83 ties); vs scripted +0.014 (CI −0.013, +0.041). No significant ablation effect. Full numbers in README. Fixed a bug found while exporting the demo: the lives-at-stake debug snapshot rebuilt the planner's own urgency tables, so recorded runs flew differently from unrecorded ones (seed 20013: 0.638 vs 0.550). It now uses separate tables; regression test in tests/viz. Demo seed is now the median among non-tied seeds (20013).

## Scenario presets (added on request)
`floodsim/presets.py`: 7 named presets (town in / far from the flood, easy escape / trapped valley, sudden breach, slow rise, two gliders). They work by narrowing the scenario bounds and setting generator knobs stored on `spec.preset`; ordinary spec hashes are unchanged. `python -m floodsim scenarios` runs Bayesian search and the survival-aware planner on 20 floods per preset (11 min), measures truth features to confirm each preset's situation, and exports one compare page per preset plus the `site/scenarios.html` selector. Results are in the README. Tests: tests/test_presets.py.

## Extreme 2076 preset
Added `extreme_2076` (climate factor 1.4 on rain and river flow, combined flooding, storm weather). Over 20 floods: 14% of land flooded and 15 people at risk per flood. Survival of people at risk is 0.63 (ours) vs 0.61 (Bayesian), paired +0.014 [−0.025, +0.053], 9 wins / 10 losses / 1 tie. `scenarios --presets <subset>` now merges with the previous scenario run, so the selector keeps every preset.

## Extreme-flood diagnosis and "+crowded" modifier
Forecast miss rate on deep water: 4% in extreme floods vs 5% in ordinary ones, so the forecast is not the cause. The percentile rank of unfound at-risk people's blocks is 0.82 (stake map) vs 0.75 (people map) in ordinary floods and 0.92 vs 0.93 in extreme floods: the maps coincide when most people are at risk. Prediction: more people help only where most are safe. Results: extreme+crowded −0.021 [−0.050, +0.008]; town_far+crowded on seeds 1-80 +0.079 [+0.041, +0.116], 21 wins / 0 losses / 35 ties. Scenario runs now accumulate in results/latest_scenarios.txt, fixing the bug where the selector kept only the last two runs.
