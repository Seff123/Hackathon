# Progress

## Phase 0: scaffold
Package skeleton, all config dataclasses (team inputs, scenario bounds and spec, world, forecast, weather, glider, autopilot, sensor, people, controller, swarm, sim), TOML loading, named RNG streams, CLI parser. Project `.venv` (Python 3.14, numpy, scipy, matplotlib, plotly, pillow, pytest). 7 tests pass.

## Phase 1: terrain and hydrology
Spectral terrain with edge rims, meandering river with a carved valley and linear-falling bed, towns, roads, levees (on the 100 m flood grid), land cover and obstacles; priority-flood, D8, accumulation, HAND. Rejection sampling with a cheap HAND flood estimate and people placement. `scenario --plot` works. 14 tests pass.
