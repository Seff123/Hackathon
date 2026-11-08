# FloodSim

FloodSim turns a seed into a random but physically plausible flood scenario and flies a swarm of 2-8 autonomous motor-gliders over it. It is the environment for tuning a flood-aware Bayesian search planner later. This build contains:

- the world: terrain, river, towns, roads, flood truth, forecast, weather and people
- glider physics and the autopilot
- sensing and radio
- a scripted controller
- scoring and visuals

The Bayesian planner itself is not built yet. It is designed in the appendix of `FloodSim Glider SAR Environment Build Plan.md` and plugs in through the controller interface in `floodsim/controllers.py`.

## Setup

```
python -m venv .venv
.venv\Scripts\python -m pip install -e .[viz,dev]   # numpy, scipy, matplotlib, plotly, pillow, pytest
```

Python 3.11+ is needed (developed on 3.14). `plotly` and `pillow` are optional:

- Without `plotly`, the 3D view falls back to a matplotlib PNG.
- Without `pillow`, no GIF is written.

## Commands

```
python -m floodsim scenario --seed 3 --plot out/          # sample + build one world, plot terrain and flood
python -m floodsim build-cache --seeds 0-199 --split train # cache worlds (writes cache/seeds_train.json)
python -m floodsim demo --seed 3 --drones 5 --out out/    # one episode: GIF, 3D HTML, plots, report
python -m floodsim evaluate --controller scripted --seeds 0-49
```

Team inputs can be set on the command line or in `[team]` in `configs/default.toml`:

```
--mass-kg 5 --battery-life-min 150 --charge-time-min 10 --signal-radius-km 3 --camera-range-m 140
```

These values are still the plan's placeholders. Every other number is a dataclass default in `floodsim/config.py` and can be overridden from TOML.

Worlds are cached in `cache/<spec_hash>.npz` (0.4-4 MB each, about 0.15 s to load). Building one uncached takes 1.5-3 s, mostly the flood solver; seeds that fail rejection checks take longer.

**Low-memory note:** `build-cache` and `evaluate` use at most 4 worker processes by default because the development laptop has 8 GB of RAM. Raise this with `--workers`.

## What `demo` writes (in `--out`)

| File | Content |
| --- | --- |
| `episode_seed<N>.gif` | 2 × 2 panels every 30 s of sim time: truth flood and people; coverage, tracks and detections; altitude and battery; vertical wind and drones |
| `episode3d_seed<N>.html` | Interactive 3D view: terrain, peak water surface, drone paths (hue per drone, dash by mode), radio links, thermal columns |
| `world_*.png`, `coverage_*.png`, `wind_*.png` | Static plots |
| `report_*.txt`, `metrics_*.json` | Metrics table and safety events |

## Tests

```
.venv\Scripts\python -m pytest            # about 5 min (multi-seed tests use the world cache)
set FLOODSIM_SKIP_PERF=1                  # skip the soft timing test
set FLOODSIM_STRICT_PERF=1                # make the timing test fail (not xfail) when over budget
```

Current status: 70 passed, 1 xfailed. The xfail is the soft timing budget (see Known gaps).

## Headline numbers

Scripted sweep controller, seeds 0-49, sampled swarm sizes 2-8, 2.5 h episodes, placeholder team inputs.

| | |
| --- | --- |
| Flood types | 23 fluvial + pluvial, 18 fluvial, 7 pluvial, 2 levee breach |
| Peak flood extent | 3.2-16.5 % of land (mean 6.7 %) |
| Wind at 10 m / 300 m | mean 4.4 / 7.7 m/s (max 13.3 at 300 m) |
| Thermals | 1-30 per scenario, w* mean 0.52 m/s; soaring share 0.5 % of airborne time |
| Detections | 16 per drone-hour, 2.7 % true (false alarms dominate, λ_fa = 0.002 per cell per frame) |
| survival_at_risk | 0.80 ± 0.04 (mean ± SE) |
| Found fraction | 0.11 ± 0.02; time to first find 90 min mean (censored at 150 min) |
| Area observed | 18.4 km² per episode on average |
| Safety | 0 crashes, 0 mid-air collisions, 0 separation breaches, 0 forced landings, 0 geofence violations |

The scripted controller is deliberately naive: it verifies every detection, so most of its time goes on false alarms. It exists to exercise the environment, not as a baseline to beat.

Forecast Brier score at a 30 min lead is 0.05-0.19 on seeds 0-19, scored over the floodplain zone.

## Known gaps

- **Speed:** a fast-mode 2.5 h episode takes about 7.5 s with 8 drones and 4.5 s with 1 drone on the development laptop, against the plan's 6 s and 2 s budgets. Numpy call overhead on tiny (8-element) arrays in the vectorised autopilot dominates. The timing test is soft (xfail with the measured numbers).
- **Seed acceptance:** seeds are accepted only when the flood solver's own peak extent is 3-40 %. The plan's cheap HAND estimate predicts the solver too poorly (about 50 % precision), so it is kept only as a pre-filter.

`ASSUMPTIONS.md` lists every choice made where the plan was silent or had to bend. `PROGRESS.md` has the phase log.

## Layout

```
floodsim/
  config.py         # every number (team inputs, bounds, world, glider, autopilot, sensor, ...)
  rng.py            # named seed streams
  scenario.py       # sample_spec (rejection checks), hydrograph
  terrain.py        # terrain, river, valley, towns, roads, levees, land cover
  hydrology.py      # priority-flood, D8, accumulation, HAND
  flood.py          # local-inertial solver, sources, FloodTimeline, hazard rating
  world.py          # World + cache
  forecast.py       # FloodForecast (agent-visible arrays), StaticMaps, Brier score
  weather.py        # log-profile wind, ridge lift, thermals
  glider.py         # point-mass glider, sink table, battery
  autopilot.py      # modes, safety layer, separation
  flight.py         # physics + autopilot + terminations for the swarm
  people.py         # placement, behaviour, danger time, survival
  sensor.py         # thermal camera, false alarms, verification
  radio.py          # range-only messages
  goals.py          # Transit / Search / Verify / RTB / Loiter
  controllers.py    # Controller interface + ScriptedSweep
  sim.py            # Sim loop, PlannerView (information boundary), EpisodeResult
  metrics.py        # survival and safety scoring
  evaluate.py       # batch runs and scenario statistics
  viz.py            # plots, GIF, 3D HTML
```
