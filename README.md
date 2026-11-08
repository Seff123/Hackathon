# FloodSim: glider search-and-rescue in synthetic floods

FloodSim turns a seed into a random but physically plausible flood scenario (terrain, river, towns, roads, flood, weather, people), flies a swarm of 2-8 battery-limited motor-gliders over it with imperfect thermal cameras and a limited radio, and scores each episode on the survival of people the flood reached. On top of the environment sits a **survival-aware Bayesian search planner** that sends each glider to where searching *now* saves the most expected survival, compared with standard Bayesian search, a flood-bonus heuristic and a scripted lawnmower.

The build follows three plans: the environment plan (`FloodSim Glider SAR Environment Build Plan.md`), `floodsim_plans/PLANNER_PLAN.md` and `floodsim_plans/VIZ_PLAN.md`. Choices made where the plans were silent are in [ASSUMPTIONS.md](ASSUMPTIONS.md); the phase log is in [PROGRESS.md](PROGRESS.md).

## Set-up

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[viz,test]"
```

Team inputs (glider mass, battery life, charge time, signal radius, camera range) are in `configs/default.toml` under `[team]` or can be passed on the command line.

## Commands

```bash
# environment
python -m floodsim scenario --seed 3 --plot out/            # spec + world / wind PNGs
python -m floodsim build-cache --seeds 0-199                 # worlds cached in cache/ (+ seeds_<split>.json)
python -m floodsim demo --seed 0 --out out/                  # GIF (+MP4), 3D HTML, plots, report
python -m floodsim demo --seed 3 --drones 5 --mass-kg 5 --battery-life-min 150 --charge-time-min 10 \
    --signal-radius-km 3 --camera-range-m 140 --out out/
python -m floodsim evaluate --controller scripted --seeds 0-49

# planners and experiments
python -m floodsim evaluate --controller survival --seeds 20000-20099
python -m floodsim compare --controllers scripted,bayes,heuristic,survival --seeds 20000-20099
python -m floodsim ablation --seeds 20000-20099
python -m floodsim lambda-sweep --seeds 20000-20049
python -m floodsim swarm-size --seeds 20000-20049
python -m floodsim tune --planner survival --trials 30 --train-seeds 0-29 --val-seeds 10000-10049
python scripts/run_experiments.py --trials 10                # the whole pipeline (tune -> compare -> ablation -> sweeps)

# visualisation (offline HTML, double-click to open)
python -m floodsim export-replay --seed <demo> --controllers bayes,survival --out site/   # site/compare_<seed>.html
python -m floodsim charts --results results/compare_<ts>/ --out site/charts/              # PNGs + site/results.html
python -m floodsim demo3d --seed <demo> --controller survival --out site/                  # site/flight3d_<seed>.html
python -m floodsim pitch --seed <demo> --out site/pitch/                                   # backup clip + key frames
python -m floodsim scenarios --seeds 1-20 --out site/                                       # presets: site/scenarios.html
python -m floodsim scenarios --list                                                          # list the presets
python -m floodsim export-replay --preset town_far --seed 12 --out site/                     # one preset flood
```

Tests: `pytest` (set `FLOODSIM_SLOW=0` to skip the 50-seed checks and `FLOODSIM_SKIP_PERF=1` to skip timing tests).

## Layout

```text
floodsim/            environment: config, rng, scenario, terrain, hydrology, flood, forecast, weather, glider,
                     autopilot, people, sensor, radio, controllers, sim, metrics, viz, runner, cli
floodsim/planner/    params, detection, belief, scenarios, floodbelief, urgency, scoring, fleet, controllers,
                     experiments, tune
floodsim/vizpage/    record, export (bundles + callouts), page.html, charts, demo3d, pitch
tests/               environment tests; tests/planner, tests/viz
configs/default.toml team inputs and overrides
```

## Results

Numbers from `scripts/run_experiments.py --trials 10` (results in `results/`, charts in `site/results.html`). Three of the 100 test seeds (20000-20099) were rejected by the scenario checks, leaving 97. Every controller flew the same 97 floods; differences are paired.

**Environment check** (scripted lawnmower, seeds 0-49, 50 episodes, 0 errors): survival of at-risk people 0.808 ± 0.040; flood covers 8.6% of land on average (p10 3.5%, p90 13.3%); mean 10 m wind 3.6 m/s; about 14 thermals per scenario; about 25 false alarms per true detection. No crashes and no separation breaches in 50 eight-drone runs.

**Main comparison** (tuned parameters, 10 Sobol trials per planner on train seeds 0-19, picked on validation seeds 10000-10019, test seeds run once):

| Controller | survival_at_risk | found fraction | first find (min) |
| --- | --- | --- | --- |
| Scripted lawnmower | 0.861 ± 0.025 | 0.116 | 70.4 |
| Bayesian search | 0.874 ± 0.024 | 0.243 | 39.2 |
| Flood-bonus heuristic | 0.868 ± 0.025 | 0.188 | 49.4 |
| Survival-aware (ours) | 0.875 ± 0.024 | 0.167 | 70.8 |

| Paired difference (ours − other) | mean | 95% CI | wins / losses / ties |
| --- | --- | --- | --- |
| vs scripted | +0.014 | [−0.013, +0.041] | 14 / 6 / 77 |
| vs Bayesian search | +0.001 | [−0.005, +0.008] | 8 / 6 / 83 |
| vs flood-bonus heuristic | +0.008 | [−0.005, +0.021] | 5 / 7 / 85 |

**What this shows.** The survival-aware planner is not distinguishable from standard Bayesian search on this test set: every confidence interval includes zero, and 83 of 97 floods end in an exact tie. In most floods, either both planners find the same at-risk people or neither does. Bayesian search finds more people overall (0.243 vs 0.167) and finds them sooner. The survival-aware planner spends its time on fewer, higher-stakes people and ends level on survival.

**Ablation, lambda sweep, swarm size.** These ran with the *default*, untuned planner parameters (the pipeline passes tuned parameters to the main comparison only), so their absolute levels differ from the table above. No ablation changes survival_at_risk significantly; the largest effects are removing prediction (−0.011, CI [−0.023, +0.001]) and state-aware detection (−0.008, CI [−0.020, +0.005]). The flood-bonus heuristic at every weight from 0 to 8 is within noise of the survival planner. Paired differences against Bayesian search at 2, 4, 6 and 8 drones all have CIs that include zero.

**Demo seed.** 20013, the median ours − Bayesian difference among the 14 floods where the two differ (+0.008), not the best. Pages: `site/compare_20013.html`, `site/flight3d_20013.html`, `site/results.html`, `site/pitch/`.

**Scenario presets** (`site/scenarios.html`, 20 floods per preset, Bayesian search vs survival-aware, tuned parameters). Each preset narrows the random scenario ranges to force one situation; see `floodsim/presets.py`.

| Preset | Typical flood | ours − Bayesian | 95% CI | wins / losses / ties |
| --- | --- | --- | --- | --- |
| Big town in the flood | 32% of the big town flooded | +0.026 | [−0.014, +0.067] | 11 / 6 / 3 |
| Big town far from the flood | town 1.5 km from the flood edge, 3 people at risk | +0.060 | [−0.011, +0.130] | 3 / 0 / 16 |
| Easy escape to high ground | 78 m walk, 45 m climb to dry ground | −0.002 | [−0.039, +0.036] | 7 / 8 / 5 |
| Trapped in a wide valley | 320 m walk through water to dry ground | −0.007 | [−0.022, +0.008] | 1 / 1 / 18 |
| Sudden levee breach | water reaches people after 70 min | +0.012 | [−0.043, +0.066] | 5 / 2 / 12 |
| Slow-rising river | water reaches people after 80 min | +0.003 | [−0.017, +0.024] | 5 / 4 / 11 |
| Two gliders, many at risk | 9 people at risk, 2 gliders | +0.018 | [−0.011, +0.047] | 6 / 4 / 10 |
| Extreme 2076 flood | 14% of land and 51% of homes flooded, 15 people at risk | +0.014 | [−0.025, +0.053] | 9 / 10 / 1 |

No preset shows a statistically clear difference with 20 floods. The two that lean most towards ours are the ones expected to: the big town far from the flood (3 wins, 0 losses) and a big town in the flood. The hills preset is level, as expected. The trapped-valley preset is mostly ties, against expectation: survival there is high (0.95) because water reaches people late in the sortie (median 93 min). In the extreme 2076 preset, at the top of the plan's 2076 climate range (×1.4 on storm rain and on river flow), survival of people at risk falls to about 0.62, against 0.87 in ordinary floods. Both planners lose about the same amount, and they almost never tie (19 of 20 floods differ), but the wins and losses cancel out. The far-town demo (seed 12, +0.49) is the median of only 3 non-tied floods, so it is a large gap, not a typical one.

**Known limitations.** The value-alignment test in `tests/planner` is marked xfail: the planner's forecast underestimates depth in backwater and breach areas, so its lives-at-stake map misses some of the places where people are in danger (see ASSUMPTIONS.md). This is the most likely reason the survival weighting gains nothing over plain Bayesian search.

