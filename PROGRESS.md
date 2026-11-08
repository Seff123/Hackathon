# Progress

## Phase 0: scaffold
Package skeleton, all config dataclasses (team inputs, scenario bounds and spec, world, forecast, weather, glider, autopilot, sensor, people, controller, swarm, sim), TOML loading, named RNG streams, CLI parser. Project `.venv` (Python 3.14, numpy, scipy, matplotlib, plotly, pillow, pytest). 7 tests pass.

## Phase 1: terrain and hydrology
Spectral terrain with edge rims, meandering river with a carved valley and linear-falling bed, towns, roads, levees (on the 100 m flood grid), land cover and obstacles; priority-flood, D8, accumulation, HAND. Rejection sampling with a cheap HAND flood estimate and people placement. `scenario --plot` works. 14 tests pass.

## Phase 2: flood truth
Vectorised local-inertial solver on the 100 m grid (channel-width faces, Froude cap, negativity limiter, free outflow), river hydrograph, moving storm, levee breach; FloodTimeline with volume-consistent downscaling to 50 m, EA hazard rating; world cache (`cache/<spec_hash>.npz`, 0.4-4 MB, 0.15 s reload) and `build-cache` with `seeds_<split>.json`. Solver ~1.5-2 s per scenario, mass residual ~1e-14. Seeds are accepted only if the solver's peak extent is 3-40% (cheap HAND estimate is a pre-filter). 24 tests pass; `out/flood_seed0.gif` looks plausible.

## Phase 3: forecast
FloodForecast precomputed as arrays (noisy stage forecasts at drainage cells for 5-min issues x 0-60 min leads, noisy observed extent every 10 min, noisy velocity every 10 min, noisy storm model), with prob_flooded / expected_depth / expected_hazard / observed_velocity. Brier at 30 min lead 0.05-0.19 on seeds 0-19. No truth reachable from the forecast. 28 tests pass.

## Phase 4: weather
Constant mean wind with log profile (z0 0.1 m, constant above 600 m), ridge lift from 150 m-smoothed terrain with H_r from 2 km relief, fixed thermal hotspots (Allen-style profile) weighted by land cover; cached with the world. Wind-slice plot shows lift on windward valley walls and lee sink. 31 tests pass.

## Phase 5: glider, autopilot, safety
Quasi-steady point-mass glider with sink-rate table, vectorised over drones; battery from battery_life_min at 13 m/s. Autopilot modes (launch, transit with altitude bands, crosswind lawnmower search, thermal circling, verify orbit, RTB with descending landing orbit, loiter), wind-corrected heading, crosswind-aware transit speed, RTB energy rule, safety layer (terrain look-ahead, geofence, stall, ceiling), predictive swarm separation, base zone. FlightCore wraps physics + autopilot + terminations. 52 tests pass (RTB 50/50 and 49/50 in strong headwind; 0 breaches in 50 busy 8-drone runs; 0 crashes in 50 ridge runs). ~0.7 ms per 8-drone flight step.

## Phase 6: people, sensor, radio
People behaviour state machine (home -> roof, outdoor walk to high ground, in-water drift, lost), danger time and survival, find freezing. Nadir LWIR camera with footprint cells, range-ratio detection model with cover/state modifiers, Poisson false alarms, verification. Range-only radio. 62 tests pass.

## Phase 7: sim loop, scripted controller, metrics
Sim with per-process clocks (1 s flight/radio, 2 s sensing, 10 s people), PlannerView whitelist (no truth reachable, tested), battery swap and relaunch, people stepped to episode end + 2 h, fast mode with identical metrics. ScriptedSweep: equal-area crosswind strips over HAND < 10 m, verifies every detection, soars below 60% battery, repeats its pattern. Metrics per section 15. Fixed along the way: geofence heading chatter, terrain-escape latch, valley walls capped at 0.3 slope, catapult + spiral launch. 70 passed, 1 xfail (soft timing: 8-drone fast episode ~7.5 s vs 6 s budget).
