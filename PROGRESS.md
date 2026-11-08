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
