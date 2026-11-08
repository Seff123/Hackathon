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
