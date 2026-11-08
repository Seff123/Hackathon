# Progress

## Phase 0: scaffold
Package skeleton, all config dataclasses (team inputs, scenario bounds and spec, world, forecast, weather, glider, autopilot, sensor, people, controller, swarm, sim), TOML loading, named RNG streams, CLI parser. Project `.venv` (Python 3.14, numpy, scipy, matplotlib, plotly, pillow, pytest). 7 tests pass.

## Phase 1: terrain and hydrology
Spectral terrain with edge rims, meandering river with a carved valley and linear-falling bed, towns, roads, levees (on the 100 m flood grid), land cover and obstacles; priority-flood, D8, accumulation, HAND. Rejection sampling with a cheap HAND flood estimate and people placement. `scenario --plot` works. 14 tests pass.

## Phase 2: flood truth
Vectorised local-inertial solver on the 100 m grid (channel-width faces, Froude cap, negativity limiter, free outflow), river hydrograph, moving storm, levee breach; FloodTimeline with volume-consistent downscaling to 50 m, EA hazard rating; world cache (`cache/<spec_hash>.npz`, 0.4-4 MB, 0.15 s reload) and `build-cache` with `seeds_<split>.json`. Solver ~1.5-2 s per scenario, mass residual ~1e-14. Seeds are accepted only if the solver's peak extent is 3-40% (cheap HAND estimate is a pre-filter). 24 tests pass; `out/flood_seed0.gif` looks plausible.
