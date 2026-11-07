# FloodSim: Glider SAR Environment Build Plan

Nov 7, 2026 · @Sef

## 0. How Claude Code should use this plan

Build in the phase order of section 18, run the tests after every phase, and keep every number in config so the team can change it later.

**Working rules**

- Python 3.11+, numpy first. Vectorise all grid operations; no per-cell Python loops in hot paths (one-off per-scenario loops such as priority-flood are fine).
- Every number in this plan is a default. Put it in a config dataclass (section 3), never inline.
- Determinism: the same seed must produce an identical world and, with the same planner, an identical episode.
- The planner never touches ground truth (section 14). This is the most important correctness rule for honest tuning.
- After each phase: run `pytest`, add a 2-3 line entry to `PROGRESS.md`, commit with message `phase N: <name>`.
- If a phase overruns its time budget by more than 50%, ship its listed fallback (section 18), record the shortcut in `ASSUMPTIONS.md`, and move on.
- When this plan is ambiguous, pick the simpler option and log it in `ASSUMPTIONS.md` rather than stopping to ask.
- Stretch goals (section 19) only after Phase 9 is complete.

**Definition of done**

1. `python -m floodsim demo --seed 0 --out out/` writes an animated GIF of one episode and an interactive 3D HTML view.
2. `python -m floodsim tune --trials 40 --scenarios 30` writes tuned planner parameters and a chart comparing lawnmower, Bayesian greedy and flood-aware planners on held-out test seeds.
3. All tests pass.
4. One episode with 8 drones runs in under about 6 s in fast mode once its world is cached, and the multi-agent smoke test (marl-smoke) passes.

## 1. Goals and scope

FloodSim turns a seed into a random but physically plausible flood scenario, so the team can tune a flood-aware Bayesian search planner, and train multi-agent RL policies, for a swarm of 2-8 autonomous motor-glider drones (2076 setting) across hundreds of scenarios.

**What the environment must do**

1. Sample a scenario from a seed within bounds: terrain, river, towns, roads, flood event, weather, people.
2. Simulate the ground-truth flood over time (water depth and velocity per cell).
3. Simulate wind with ridge lift, thermals, sink and gusts that the glider can exploit or must avoid.
4. Fly a swarm of 2-8 3D point-mass motor-gliders with battery limits over real elevation, climbing, descending and keeping safe separation.
5. Detect people with imperfect thermal cameras, share observations over a limited-range radio, keep Bayesian belief maps, and plan where each drone flies.
6. Score outcomes by survival-weighted finds, and tune the planner's parameters across scenarios.

**The "model" being trained** is the planner's parameter vector θ (section 13): flood urgency weight, forecast lead time, prior weights, search altitude, soaring thresholds and so on. It is tuned by black-box optimisation over many scenarios. The same environment also exposes a multi-agent RL interface (section 14.1), so the team can train neural swarm policies and compare them with the tuned planner on identical seeds.

**Out of scope for the 3-hour build:** running the RL training itself (the build delivers the interface, demo recording and a smoke test), CFD-accurate wind, a game-engine renderer, real map data. Each has a stretch entry in section 19 or a recipe in section 14.1.

## 2. Assumptions and defaults

These are the choices made where the brief was silent. Each lives in config; change the default rather than the code.

| Item | Default | Reason | Change it if |
| --- | --- | --- | --- |
| Stack | Python 3.11, numpy, scipy, matplotlib, pytest; optional optuna, plotly, pillow | Fast to build, no GIS dependencies | You need GPU or a game engine |
| Domain | 10 km × 10 km | One glider can cover the key areas in a 2-3 h sortie | You want regional searches |
| Grid | 50 m cells (200 × 200) for terrain, sensor and belief; 100 m (100 × 100) for the flood solver | Keeps the flood solver to a few seconds per scenario | You need street-level detail |
| Time steps | Glider 0.5 s, sensor 2 s, people 10 s, belief prediction 60 s, flood snapshots 60 s | Each process at its natural speed | Physics looks jerky (lower glider step) |
| Episode | 2.5 h simulated (bounds 2-3 h) | Matches the battery endurance below | Longer missions |
| Frame | East-North-Up metres, origin at the south-west corner; heading in radians anticlockwise from east | Standard maths convention | Never mid-build |
| Drone | Motor-glider, about 3.7 m span, 5 kg, 120-250 Wh battery | Battery is tight enough that soaring matters | Different airframe |
| Camera | Nadir thermal (LWIR) camera, 1280 × 1024 pixels, 60° field of view; works day and night; cannot see through glass, roofs or dense canopy | Today's SAR drones mostly carry 640 × 512; double the resolution is a modest 2076 step | Different sensor |
| Altitude limits | Search at 60-150 m above ground; 60 m minimum; 400 m above ground ceiling for transit and soaring | Thermal detection above about 150 m is poor; 400 m still allows thermal climbs and about 8 km still-air glides (assumes a 2076 SAR exemption) | You want today's 120 m limit (soaring then barely works) |
| Flood types | Fluvial and pluvial (MVP), levee breach (phase 2), coastal surge (stretch) | Covers inland UK-style floods first | Coastal city scenario |
| People | 3-25 targets, hidden from the planner | Enough to score, few enough to search | Mass-casualty scenario |
| Hazard metric | UK Environment Agency hazard rating, HR = d(v + 0.5) + DF | Recognised, simple, uses depth and velocity | Different country standard |
| Climate factor | Rainfall intensity × 1.0-1.4 for 2076 | Hackathon framing; tunable | You have projection data |
| Base | One launch site on high ground near the largest town, never flooded | Realistic and simplifies return-to-base | Mobile launch vehicle |
| Wind limit | Wind at 10 m of 1-8 m/s, and at most 15 m/s at 300 m above ground | The glider must be able to make headway | Testing extreme weather |
| Detections | A detection triggers a 60 s verification orbit; confirms a true person with p = 0.95 | False alarms cost time, as in reality | Instant confirmation by other means |
| Drones | 2-8 per scenario (sampled) | The team trains swarm behaviour; varying the size gives a scaling curve | You need larger swarms (raise the cap; observation padding grows) |
| Comms | Radio range 1-5 km with terrain line of sight, 0-30% message loss; drones share observations by gossip | Realistic for small drones over flooded valleys; coordination must cope with gaps | You want perfect comms (a flag exists for ablations) |

## 3. Architecture and conventions

The code splits into a world builder (seed to ground truth, cached), a simulation loop (physics, sensing, belief, planning) and a tuning harness on top.

&#91;embedded content: FloodSim architecture · truth, noisy channels, planner, tuner\]

Arrows pointing down are the only routes by which information reaches the planner; the single arrow up is the autopilot's control commands to the glider.

**Repo layout**

```
floodsim/
  __main__.py      # CLI: scenario, build-cache, demo, evaluate, tune, marl-smoke, record-demos, export-flood-dataset
  config.py        # all dataclasses: Bounds, ScenarioBounds, ScenarioSpec, GliderParams, SensorParams, PlannerParams, SwarmParams, SimConfig
  rng.py           # SeedSequence -> named child streams
  scenario.py      # sample_spec(seed, bounds), rejection checks, build_world(spec)
  terrain.py       # spectral noise, valley + river carving, land cover, towns, roads, levees
  hydrology.py     # priority-flood, D8 flow, accumulation, drainage network, HAND
  flood.py         # local-inertial solver, sources, FloodTimeline, hazard rating, cache
  forecast.py      # FloodForecast (planner-visible, noisy)
  weather.py       # mean wind, log profile, turbulence, ridge lift, thermals, rain cells
  glider.py        # GliderState arrays over drones, vectorised point-mass step, energy
  autopilot.py     # modes + safety layer (terrain, geofence, stall, separation)
  people.py        # placement, behaviour state machine, hazard dose, survival
  sensor.py        # footprint, thermal detection model, false alarms, verification
  belief.py        # Poisson intensity grid: prior, predict, update
  swarm.py         # radio model, observation logs and gossip, task claims, thermal sharing
  planners.py      # Lawnmower, BayesGreedy, FloodAwareBayes (all swarm-aware)
  sim.py           # Sim loop, one PlannerView per drone, EpisodeResult
  marl_env.py      # PettingZoo ParallelEnv wrapper: observations, action masks, rewards
  metrics.py       # survival score and secondary metrics
  tune.py          # Optuna / random-search tuner, seed splits, parallel runs
  viz.py           # static plots, GIF animation, plotly 3D
tests/             # one test file per module + test_integration.py
configs/default.toml
PROGRESS.md  ASSUMPTIONS.md  README.md
```

**Conventions**

- SI units everywhere; time `t` in seconds since episode start.
- Grids are numpy arrays indexed `[iy, ix]` (row = north index). Use one helper pair, `world_to_grid(x, y)` and `grid_to_world(iy, ix)`, everywhere.
- Grids are float32; glider state is float64.
- Wind direction is stored as the direction the wind blows towards, radians anticlockwise from east. Convert meteorological "from" degrees with θ = radians(270 − from\_deg).
- Randomness: `np.random.SeedSequence(seed).spawn()` gives separate streams named terrain, hydro, flood, weather, people, sensor. Changing one subsystem must not change the others' draws.
- Anything that varies with time but not with the planner (turbulence sequence, thermal list, storm track) is generated up front at world build, so every planner faces identical conditions (common random numbers).

**Dependencies:** numpy, scipy (ndimage, interpolate, sparse.csgraph, stats.qmc), matplotlib, pytest. Optional: optuna (tuner), plotly (3D HTML), pillow (GIF), pettingzoo (multi-agent interface), torch + stable-baselines3 + supersuit (training smoke test). Every optional import must degrade gracefully.

**Performance budget**

| Step | Target |
| --- | --- |
| `build_world` uncached (flood solver dominates) | < 8 s |
| `build_world` from cache (`cache/<spec_hash>.npz`) | < 0.5 s |
| One 2.5 h episode, fast mode | < 2 s with 1 drone, < 6 s with 8 (physics, sensing and belief vectorised across drones) |

## 4. Scenario randomisation

A seed deterministically produces a `ScenarioSpec` by sampling every parameter below from its bounds; rejection checks then guarantee the scenario is usable before any expensive simulation runs.

**Mechanics**

- `Bounds(lo, hi, dist)` with `dist` in `uniform`, `loguniform`, `int`, `normal_clipped`; categorical choices use a weights dict.
- `sample_spec(seed, bounds) -> ScenarioSpec`. The spec is a flat, JSON-serialisable dataclass with `spec_hash` (SHA-1 of its sorted JSON). The hash keys the world cache.
- Correlations: cloud cover is drawn first; rain intensity scales up and thermal strength scales down with it. The storm moves with the mean wind direction ±30°.
- `sample_seeds(n, strata)` returns seeds balanced across flood types and swarm sizes so tuning sees every combination.

**Terrain bounds**

| Parameter | Range | Distribution |
| --- | --- | --- |
| Relief amplitude | 20-300 m | log-uniform |
| Spectral slope β | 1.8-2.4 | uniform |
| Regional slope towards outlet | 0.001-0.008 | uniform |
| Outlet edge | N, E, S or W | categorical |
| Valley width | 300-2,000 m | uniform |
| Valley depth | 10-60 m | uniform |
| River sinuosity | 1.1-1.8 | uniform |
| Channel width / depth | 20-80 m / 2-5 m | uniform |
| Tributaries | 0-3 | integer |
| Towns / town radius | 1-3 / 300-1,200 m | integer / uniform |
| Levee height near towns | 0-3 m (0 = none) | uniform |
| Forest fraction | 0.05-0.4 | uniform |

**Flood bounds**

| Parameter | Range | Distribution |
| --- | --- | --- |
| Flood type | fluvial 0.45, fluvial + pluvial 0.25, pluvial 0.2, with breach 0.1 | categorical |
| Peak river inflow | 100-1,500 m³/s | log-uniform |
| Base flow | 5-15% of peak | uniform |
| Time to peak | 1-6 h | uniform |
| Hydrograph shape exponent k | 2-5 | uniform |
| Episode start relative to flood onset | −0.5 to +4 h | uniform |
| Peak rainfall | 10-60 mm/h × climate factor 1.0-1.4 | uniform |
| Storm duration / radius | 0.5-4 h / 2-8 km | uniform |
| Manning n: channel, floodplain, urban, forest | 0.03-0.04, 0.04-0.07, 0.06-0.12, 0.08-0.15 | uniform |
| Breach time / width / head drop | 0-2 h after start / 20-100 m / 1-4 m | uniform |
| Forecast error: initial σ, growth, bias | 0.1-0.3 m, 0.1-0.4 m/h, −0.3 to +0.3 m | uniform |

The start offset is what puts episodes on the rising limb, at the peak or on the falling limb, which changes how much the urgency term matters.

**Weather bounds**

| Parameter | Range | Distribution |
| --- | --- | --- |
| Mean wind at 10 m | 1-8 m/s | uniform |
| Wind direction | 0-360° | uniform |
| Turbulence intensity | 0.08-0.25 | uniform |
| Wind drift time constant / direction wander | 10-40 min / 5-20° | uniform |
| Cloud cover | 0.3-1.0 | uniform |
| Convective velocity w\* | 0-2.5 m/s × (1 − cloud cover) × 1.4 | derived |
| Boundary-layer depth zi | 400-1,500 m | uniform |
| Thermal density | 0-0.6 per km² per hour | uniform |
| Thermal lifetime | 5-20 min | uniform |
| Extra convective showers | 0-3 | integer |
| Start time of day | 0-24 h | uniform |

**People and aircraft bounds**

| Parameter | Range | Distribution |
| --- | --- | --- |
| Number of targets | 3-25 | integer |
| Category weights: home, vehicle, outdoor, group | about 0.5, 0.2, 0.2, 0.1 | Dirichlet (concentration 20) |
| Immobile fraction | 0.2-0.5 | uniform |
| Walking speed | 0.5-1.3 m/s | uniform |
| Drift fraction of flow speed when in water | 0.3-0.8 | uniform |
| Glider mass | 4.5-5.5 kg | uniform |
| Battery capacity | 120-250 Wh | uniform |
| Payload power | 20-35 W | uniform |
| Swarm size | 2-8 drones | integer, stratified |
| Radio range | 1-5 km | uniform |
| Message loss | 0-30% | uniform |
| Launch interval | 20-60 s between drones | uniform |
| Per-drone variation | mass, battery and payload power drawn independently for each drone | within the bounds above |

**Rejection checks.** After terrain and a cheap HAND-based flood estimate (not the full solver), reject and resample (up to 20 attempts, then widen the failing bound by 10%) if any check fails:

1. Peak flood extent covers 3-40% of land cells.
2. At least 60% of targets start within 1 km of the peak flood extent.
3. The base is never inside the flood extent and sits on a road.
4. Highest terrain is at most 300 m above the base, so returning to base never needs a long climb.
5. Mean wind at 300 m above ground (log profile with z0 = 0.1 m) is at most 15 m/s, so the glider can make headway at its 18-30 m/s fast-cruise speeds.
6. The river network drains to the outlet edge (at least 80% of land cells drain there).

Re-check 1 and 3 against the full solver output; on failure, log it and move to the next seed rather than looping.

## 5. Terrain, land cover and hydrology

Terrain is fractal noise with a carved, meandering river valley, so water drains realistically; hydrology layers derived from it (flow, drainage, HAND) feed both the flood model and the planner's prior.

**Generation steps (`terrain.py`, `hydrology.py`)**

1. **Base surface.** Spectral synthesis: white noise → FFT → multiply amplitudes by f^(−β/2) → inverse FFT → rescale to the relief amplitude. Add a plane sloping down towards the outlet edge at the regional slope.
2. **River centreline.** Start within ±30% of the midpoint of the edge opposite the outlet; end on the outlet edge. Path = straight line + 2-4 sinusoids with random phases; scale their amplitude until arc length / straight length matches the sampled sinuosity (bisection, 10 iterations). Smooth and rasterise.
3. **Valley.** `distance_transform_edt` from the centreline. Lower terrain by depth × smoothstep(1 − d / (width / 2)). Inside the floodplain, blend towards the centreline elevation plus 0.5-2 m of low-amplitude noise so it undulates gently. Force centreline elevation to fall linearly from inflow to outlet (plus ±0.3 m noise), then lower centreline cells by the channel depth.
4. **Tributaries.** Same method from random edge points to the main river, half the channel size.
5. **Levees.** Where the river passes within 1 km of a town, raise a 1-cell embankment on both banks by the sampled levee height. Breach scenarios cut one of these.
6. **Depression filling.** Priority-flood (Barnes 2014) with a tiny epsilon gives `z_fill`. Keep the raw DEM for the flood solver so ponds can form; store `depression_depth = z_fill − z`.
7. **Flow.** D8 directions on `z_fill`; flow accumulation by processing cells from highest to lowest (`argsort` once, then a loop over 40k cells, about 50 ms). Drainage cells are those with upstream area ≥ 1 km².
8. **HAND (height above nearest drainage).** Process cells in reverse flow order so each inherits its downstream drainage cell: `drain_id[c]` and `HAND[c] = z[c] − z[drain_id[c]]`. Also store `dist_to_river`.
9. **Land cover.** Classes: water (channel), urban (town disks with noisy edges), forest (noise threshold, favouring steeper slopes, away from towns), farmland (flat floodplain and gentle slopes), open (the rest). Obstacle height: forest 15-25 m, urban 6-20 m, else 0. `surface = z + obstacle_height` is used for collisions and sensor occlusion.
10. **Settlements.** Town centres within 1.5 km of the river on low-slope floodplain edges. Building density = Gaussian falloff from each centre. Farms = Poisson points on farmland at 0.5-2 per km².
11. **Roads.** Least-cost paths between towns, farm clusters and the base with `scipy.sparse.csgraph.dijkstra` on a 100 m, 8-neighbour graph. Cost = 1 + 50 × slope + 5 × forest + 1,000 × channel, except 1-2 allowed bridge cells per river crossing.
12. **Population map (planner-visible).** Weighted sum of building density, road mask and farm points, normalised.
13. **Base.** Highest road cell within 2 km of the largest town's edge; re-picked if the full flood solver ever wets it.

**`Terrain` dataclass fields:** `z, z_fill, surface, slope, grad_x, grad_y, landcover, obstacle_h, z0 (roughness), flow_dir, flow_acc, drainage_mask, drain_id, hand, dist_to_river, river_cells, levee_mask, building_density, road_mask, farms, towns, population, base_xy, cell_size`.

**Fallback if over budget:** replace dijkstra roads with straight noisy lines between towns, and skip tributaries.

## 6. Flood model: ground truth and forecast

Ground truth comes from a vectorised local-inertial shallow-water solver on a 100 m grid, run once per scenario and cached; the planner never sees it and instead gets a cheaper, deliberately imperfect HAND-based forecast.

### 6.1 Ground truth (`flood.py`)

**Grid.** Downsample the DEM to 100 m: block minimum on channel cells (keeps the channel continuous), block mean elsewhere. State: depth `h` at cell centres, unit discharges `qx`, `qy` on cell faces (staggered).

**Momentum update per face** (Bates et al. 2010), with water surface η = z + h and face flow depth h\_f = max(η\_i, η\_j) − max(z\_i, z\_j):

```latex
q^{t+\Delta t} = \frac{q^{t} - g\, h_f\, \Delta t\, \dfrac{\eta_j - \eta_i}{\Delta x}}{1 + g\, \Delta t\, n^{2}\, |q^{t}| \,/\, h_f^{7/3}}
```

**Continuity per cell:**

```latex
h^{t+\Delta t} = h^{t} + \Delta t \left( \frac{\sum q_{in} - \sum q_{out}}{\Delta x} + C_r\, R(x, y, t) + \frac{Q_{src}}{\Delta x^{2}} \right)
```

- Time step Δt = 0.7 Δx / √(g h\_max), clamped to 0.5-10 s.
- Wet/dry: set q = 0 on faces where h\_f < 1 mm. Scale a cell's outgoing fluxes so its depth never goes negative.
- Boundaries: upstream river entry cells receive the hydrograph Q(t); the outlet edge's channel cells are free outflow (zero-gradient surface); all other edges are walls.
- Initial state: fill the channel to Manning normal depth for base flow, h0 = (Q n / (W √S))^(3/5), then spin up for 30 simulated minutes (not part of the episode).

**Sources**

- River hydrograph (gamma-shaped): Q(t) = Q\_base + (Q\_peak − Q\_base) × \[(t / t\_p) exp(1 − t / t\_p)\]^k. Tributaries get scaled copies (30-60% of peak) with a 0-30 min lag.
- Rain: R(x, y, t) = I\_peak × exp(−r² / 2R\_c²) × temporal envelope, with the storm centre moving along its track. Runoff coefficient C\_r: urban 0.9, farmland 0.5, open 0.5, forest 0.3.
- Breach (phase 2): at the breach time, lower one levee section by the head drop over the breach width. Overtopping of intact levees happens naturally.

**Outputs.** Every 60 s store depth and the cell-centred velocity (u, v) = q / h (zero where h < 5 cm) as float32 arrays `[T, ny, nx]` in `cache/<spec_hash>.npz`. Run from the end of spin-up to episode end + 1 h so forecasts can be checked against truth.

**`FloodTimeline` API:** `depth(x, y, t)` and `velocity(x, y, t)` (bilinear in space, linear in time), `depth_grid(t)`, `velocity_grid(t)`, `hazard_grid(t)`; `upsample` to 50 m with `scipy.ndimage.zoom(order=1)`.

**Hazard rating** (Environment Agency, FD2320): HR = d(v + 0.5) + DF, with debris factor DF = 0 for d ≤ 0.25 m, 0.5 for 0.25-0.75 m (1.0 on urban or forest cells), 1.0 above 0.75 m. Classes: below 0.75 low; 0.75-1.25 danger for some; 1.25-2.0 danger for most; above 2.0 danger for all.

**Mass check:** track volume in (river, rain), out (outlet), and stored; the residual must stay under 1%.

**Fallback if over budget:** a HAND-based truth model where each cell floods when the river stage at its drainage cell exceeds its HAND, with the flood wave travelling downstream at 1 m/s and velocity approximated from the local slope. Everything downstream still works.

### 6.2 Forecast (`forecast.py`, planner-visible)

The planner knows three things at time t, all noisy:

1. **Observed extent**, refreshed every 10 min (2076 satellite or drone mapping): each cell's wet/dry state, correct with p = 0.95.
2. **River stage forecast** at drainage cells: the true stage plus a scenario-wide bias plus Gaussian error with σ(L) = σ0 + σ\_growth × L for lead time L in hours. The bias makes forecasts consistently wrong in one direction, as real ones often are.
3. **Flood probability** from HAND:

```latex
P_{flood}(c,\, t+L) = \Phi\!\left( \frac{\hat{S}(\mathrm{drain}(c),\, t+L) - \mathrm{HAND}(c)}{\sigma(L)} \right)
```

Cells observed wet now get P = max(P, 0.95). For pluvial ponding, use the same form with forecast rain depth (× runoff coefficient × upstream area) against `depression_depth`.

**API:** `prob_flooded(t_now, lead_s)`, `expected_depth(t_now, lead_s)`, `expected_hazard(t_now, lead_s)`, `observed_velocity(t_now)` (true velocity × (1 ± 0.3) noise), each returning a 50 m grid and cached per 5-minute bucket.

**Report the forecast's Brier score** against truth at 15, 30 and 60 min leads. It gives the team a number for "not high confidence" in the pitch.

## 7. Weather: wind, updrafts and rain

`Weather.wind(x, y, z, t)` returns (u, v, w) as the sum of a drifting mean wind with a log height profile, turbulence, ridge lift, thermals and rain downdrafts, and floods switch thermals off where the ground is wet.

**1. Mean wind.** Speed U10(t) and direction θ(t) follow Ornstein-Uhlenbeck processes around the sampled means (time constant 10-40 min, speed σ = 15% of mean, direction σ = 5-20°). Generate the series for the whole episode at world build.

**2. Height profile (log law).** With z0 sampled from a roughness map smoothed at σ = 500 m (water 0.0002, open 0.03, farmland 0.1, forest 0.8, urban 1.0 m):

```latex
U(z_{agl}) = U_{10} \, \frac{\ln(z_{agl}/z_0)}{\ln(10/z_0)}, \quad z_{agl} \ge 2\ \mathrm{m},\ \text{constant above } 600\ \mathrm{m}
```

**3. Turbulence (Dryden-like).** Per-axis OU noise along the glider's path: σ\_u = σ\_v = TI × U(z), σ\_w = 0.7 σ\_u; length scales L\_uv = 200 m and L\_w = min(50 + 0.5 z\_agl, 300) m; time constant τ = L / V\_air. Discrete update x ← x e^(−Δt/τ) + σ √(1 − e^(−2Δt/τ)) ξ. Pre-generate the unit-noise sequence ξ per step index so every planner gets the same gusts.

**4. Ridge lift.** With z\_s = terrain smoothed at σ = 150 m and V\_h the horizontal mean wind at that height:

```latex
w_{ridge} = k_r \, (\mathbf{V}_h \cdot \nabla z_s)\, e^{-z_{agl}/H_r}, \qquad H_r = \max(0.8 \times \text{local relief in 2 km},\ 50\ \mathrm{m})
```

k\_r = 0.8-1.2. Positive on windward slopes, negative (sink) in the lee, clipped at −2 m/s. Precompute ∇z\_s and H\_r at world build.

**5. Thermals** (simplified from Allen 2006). Each thermal has a spawn time, centre, core strength w\_core = 0.8-1.5 × w\*, radius r\_t = 0.1-0.2 zi (clamped 60-200 m), and lifetime (2 min ramp up, steady, 2 min decay). Centres drift with the mean wind at 0.5 zi.

```latex
w_{th}(r, z) = w_{core}\, g\!\left(\tfrac{z_{agl}}{z_i}\right) \left[ e^{-(r/r_t)^2} - 0.25\, e^{-((r - 2 r_t)/r_t)^2} \right], \quad g(s) \propto s^{1/3}(1 - 1.1 s),\ g_{max} = 1
```

- Spawning: a Poisson process at the sampled density × land-cover heating factor (urban 1.5, farmland 1.0, open 1.0, forest 0.5, water 0). **No thermals spawn on cells with flood depth > 10 cm**, and none at night (20:00-06:00).
- Background sink inside the boundary layer balances the thermals' mass flux: w\_env ≈ −(area fraction × mean w\_core), typically −0.1 to −0.4 m/s.
- Generate the full thermal list at world build. At run time keep a spatial hash of active thermals, refreshed every 10 s, so each wind query checks only nearby ones.

**6. Rain.** The pluvial storm from the flood spec plus 0-3 smaller showers, all moving with the wind. Downdraft under rain intensity I (mm/h): w\_rain = −min(3, 0.05 I) m/s. Rain intensity is also exposed to the sensor model.

**7. Air density** (ISA): ρ(z) = 1.225 (1 − 2.2558 × 10⁻⁵ z)^4.2559 kg/m³ with z above sea level.

**API:** `wind(x, y, z, t)` scalar and allocation-free for the glider step; `wind_grid(z_agl, t)` vectorised for plots; `rain(x, y, t)`; `thermals_active(t)` (debug only).

**What the glider can sense** (passed to the planner): variometer netto reading = true w + N(0, 0.1 m/s) with 1 s lag, and a mean-wind estimate with N(0, 1 m/s) error per component. The planner never calls `Weather` directly.

**Fallback if over budget:** drop turbulence and showers; keep mean wind, profile, ridge lift and thermals.

## 8. Glider dynamics and battery

The glider is a quasi-steady 3D point mass: lift balances weight in a banked turn, drag comes from a parabolic polar, and an energy equation gives climb or sink, which stays stable and fast at Δt = 0.5 s.

**Default parameters (`GliderParams`)**

| Parameter | Default | Notes |
| --- | --- | --- |
| Mass m | 5.0 kg | randomised 4.5-5.5 |
| Wing area S / span b | 0.90 m² / 3.67 m | aspect ratio 15 |
| Zero-lift drag C\_D0 | 0.018 |  |
| Oswald efficiency e | 0.85 | k = 1 / (π e AR) ≈ 0.025 |
| Max lift coefficient C\_Lmax | 1.3 |  |
| Airspeed limits | 9-30 m/s |  |
| Bank limit / roll rate | 45° (60° for safety manoeuvres) / 45°/s |  |
| Static thrust / pitch speed | 20 N / 40 m/s | T\_max(V) = 20 (1 − V/40) |
| Propulsive efficiency η | 0.65 |  |
| Payload + avionics power | 25 W | randomised 20-35 |
| Battery | 180 Wh | randomised 120-250 |
| Reserve on landing | 20% | tunable |

**Sanity numbers these produce** (use as test targets): best glide ratio ≈ 23.6 at ≈ 10.2 m/s, sinking ≈ 0.43 m/s; stall ≈ 8.3 m/s level; sink ≈ 0.55 m/s circling at 35° bank and 10.5 m/s (turn radius ≈ 16 m); full-throttle climb ≈ 2.9 m/s at 12 m/s; level cruise at 14 m/s draws ≈ 79 W, about 2.3 h on 180 Wh.

**State:** x, y, z (m), true airspeed V (m/s), air-relative heading ψ (rad), bank φ (rad), battery energy E (Wh), flags `stalled`, `crashed`, `landed`.

**Swarm:** every state field is an array over drones (shape `[n_drones]`), and `glider_step` advances all drones in one vectorised numpy call. Do not loop over drones in Python inside the physics step.

**Controls from the autopilot each step:** commanded airspeed V\_cmd, commanded bank φ\_cmd, throttle τ ∈ \[0, 1\].

**Equations of motion**

```latex
\begin{aligned}
\dot{\phi} &= \mathrm{clip}\!\left(\frac{\phi_{cmd} - \phi}{0.5\,\mathrm{s}},\ \pm 45^{\circ}/\mathrm{s}\right), \qquad \dot{V} = \mathrm{clip}\!\left(\frac{V_{cmd} - V}{3\,\mathrm{s}},\ \pm 2\ \mathrm{m/s^2}\right) \\
C_L &= \frac{2\, m g}{\rho V^2 S \cos\phi}, \qquad D = \tfrac{1}{2}\rho V^2 S\,(C_{D0} + k\, C_L^2), \qquad T = \tau\, T_{max}(V) \\
\dot{h}_{air} &= \frac{(T - D)\,V}{m g} - \frac{V \dot{V}}{g}, \qquad V_h = \sqrt{\max(V^2 - \dot{h}_{air}^2,\ 0.25 V^2)}, \qquad \dot{\psi} = \frac{g \tan\phi}{V} \\
\dot{x} &= V_h \cos\psi + u_w, \qquad \dot{y} = V_h \sin\psi + v_w, \qquad \dot{z} = \dot{h}_{air} + w_w \\
\dot{E} &= -\frac{T V / \eta + P_{payload}}{3600}\ \ \mathrm{Wh/s}
\end{aligned}
```

- If C\_L > C\_Lmax: set `stalled`, clamp C\_L, add 3 m/s extra sink, count a violation.
- If E ≤ 0: force τ = 0.
- Integrate with semi-implicit Euler at Δt = 0.5 s: update φ and V first, then rates, then position. Offer RK2 behind a flag for validation runs.
- Wind is treated as quasi-steady; wind-gradient (dynamic soaring) terms are ignored. Note this in `ASSUMPTIONS.md`.

**Termination conditions**

- Crash: z ≤ surface(x, y) (terrain + trees and buildings).
- Geofence violation: more than 500 m outside the domain.
- Forced landing: battery empty and less than 30 m above ground.
- Landed: within 100 m of base, below 30 m above it, in RTB mode.

Record the trajectory every 2 s (x, y, z, V, φ, E, mode, netto) for plots and metrics.

## 9. Autopilot and flight modes

The planner sets goals; the autopilot turns them into V\_cmd, φ\_cmd and throttle every 0.5 s, and a safety layer that runs last always wins.

**Modes**

| Mode | Entered when | Behaviour | Exits when |
| --- | --- | --- | --- |
| TRANSIT | Planner sends `Transit(xy, alt_band)` | Wind-corrected heading to target, MacCready speed, altitude band logic | Within 150 m of target |
| SEARCH | Planner sends `Search(box, agl, spacing)` | Wind-aligned lawnmower legs at fixed height above ground, throttle holds altitude, bank ≤ 25° on legs | Pattern complete or planner replans |
| THERMAL | Netto > w\_enter for 4 s, and soaring is allowed (see below) | Circle at 35° bank, 1.1 × min-sink speed at that bank, centring routine | Climb over last circle < 0.3 m/s, reached min(ceiling, zi − 50 m), or 15 min elapsed |
| VERIFY | Planner sends `Verify(cell)` | Descend to 70 m above ground, orbit at 80 m radius for 60 s | Orbit done (sensor returns confirm or reject) |
| RTB | Energy rule below, or planner request | Transit to base, then descending orbit and landing | Landed |

**Wind-corrected heading.** For desired ground track χ\_d and crosswind component w\_c (wind projected on the track's left normal), the air heading is ψ\_d = χ\_d − asin(clip(w\_c / V\_h, −1, 1)). Ground speed = V\_h cos(ψ\_d − χ\_d) + along-track wind. If ground speed < 1 m/s, flag `cannot_progress` to the planner. Bank command: φ\_cmd = clip(atan(V × 0.5 × wrap(ψ\_d − ψ) / g), ±35°).

**Speed to fly (MacCready).** Precompute a table over MacCready setting M (0-3 m/s) and headwind U\_h (−15 to +15 m/s): V\_cmd = argmin over V of (sink(V) + M) / (V − U\_h). In sink (netto < −1 m/s) add 2 m/s.

**Altitude band logic** (TRANSIT and default): below the band's lower edge, full throttle at best-climb speed until 50 m above it (hysteresis); above the upper edge, throttle off and speed up to 18 m/s; inside the band, glide with throttle off. Gliding is the energy-saving default.

**Search altitude hold.** Throttle τ = clip(τ\_ff + 0.02 × (z\_target − z), 0, 1), with feed-forward τ\_ff = D(V) / T\_max(V) for level flight. Legs run parallel to the wind (no crab, so the footprint stays straight); leg spacing = 2 × agl × tan(30°) × (1 − overlap). End turns use 35° bank and the camera is ineffective during them.

**Soaring allowed when** battery fraction < E\_soar, or the next transit needs more height than the current altitude, and z < ceiling − 100 m. Both thresholds are planner parameters.

**Thermal centring** (simplified Reichmann rule). Keep the netto samples and positions from the last full circle. Shift the circle centre 0.3 × radius towards the position-weighted average of samples above the circle's mean netto. Circle radius r = V² / (g tan φ).

**Return-to-base rule.** Each second estimate E\_return = (distance to base / ground speed against the forecast wind at best-glide speed) × level-flight power, × 1.3 safety factor, + reserve. Trigger RTB when E ≤ E\_return. Skip the rule while a glide home is possible: height above base ≥ distance / (0.7 × L/D\_max × (1 − headwind / V)).

**Safety layer** (runs after every mode, overrides it):

1. Terrain: sample `surface` along the predicted path 10 s ahead. If clearance < 60 m, full throttle, best-climb speed, and turn towards lower terrain (up to 60° bank).
2. Geofence: within 300 m of the boundary and heading outward, turn back towards the centre.
3. Stall protection: V\_cmd ≥ 1.15 × stall speed at the current load factor.
4. Ceiling: above the altitude ceiling, throttle off and leave any thermal.

**Separation (swarm).** Every 0.5 s, check all drone pairs (at most 28) in one vectorised step:

- Inside 300 m horizontally and 50 m vertically: the higher-id drone climbs or descends 30 m away from the other and turns right 30°.
- Inside 100 m horizontally and 30 m vertically: log a separation breach.
- Inside 10 m: mid-air collision; both drones crash.
- Transit legs use altitude layers 30 m apart by drone id.
- Drones may share a thermal only when circling in the same direction (set by its first user) with at least 30 m vertical spacing, up to `max_per_thermal` drones (a planner parameter).

**Fallback if over budget:** thermal mode circles around the entry point without centring; RTB uses a fixed 25% energy trigger.

## 10. People: placement, behaviour and survival

People are hidden ground truth: they start where people plausibly are, react to rising water, and build up a hazard dose that decides how much finding them earlier is worth.

**Placement (`people.py`).** Draw N targets, then a category for each from the sampled weights:

- HOME: a building cell, weighted by building density × (1 + 3 × \[HAND < 5 m\]) so that scenarios contain people at risk.
- VEHICLE: a road cell, weighted towards low crossings (HAND < 3 m).
- OUTDOOR: farmland or open land within 2 km of the river.
- GROUP: 2-6 people at one point; one target with a `size`.

A sampled fraction is immobile (elderly or injured). The planner's prior (section 12) uses similar but not identical weights, so prior and truth disagree a little, as they would in reality.

**Behaviour states** (updated every 10 s using the true depth d and speed v at the person's position)

| State | Thermal detection factor | Shelter factor s | Transitions |
| --- | --- | --- | --- |
| HOME\_INSIDE | 0.02 (roofs block LWIR) | 0.2 | d ≥ 0.5 m → ROOF |
| ROOF | 1.0 | 0.1 | none |
| VEHICLE | 0.2 (glass blocks LWIR) | 0.6 | d > 0.6 m and v > 1 m/s → IN\_WATER at 0.5 per min |
| OUTDOOR | 1.0 | 1.0 | mobile and (d > 0.1 m or flood within 200 m) → walk to higher ground; arrives → SAFE\_HIGH; d > 0.5 m → IN\_WATER |
| SAFE\_HIGH | 1.0 | 0.0 | none |
| IN\_WATER | 0.8 (warm head on cold water; small size handled in section 11) | 1.5 | in forest or urban cell → CLINGING at 0.1 per min; d < 0.3 m → OUTDOOR |
| CLINGING | 0.6 | 1.0 | none |

**Walking to higher ground:** each 10 s, move up to walk speed × 10 s towards the neighbouring cell with the highest HAND, avoiding cells with d > 0.5 m; arrive when HAND is at least 3 m above the start. **Drifting in water:** velocity = drift fraction × flow velocity + random walk (σ = 0.3 m/s). People swept off the map count as lost.

**Hazard dose and survival**, using the hazard rating HR from section 6:

```latex
D_i(t) = \int_0^t s_i(\tau)\, \max\!\big(0,\ HR_i(\tau) - 0.75\big)\, d\tau, \qquad P^{surv}_i(t) = \exp\!\left(-\frac{D_i(t)}{D_{crit}}\right) \exp\!\left(-\frac{t - t^{affected}_i}{\tau_{base}}\right)
```

- D\_crit = 1,800 s (20 min at HR = 2 leaves an exposed person, s = 1, near 0.43 survival and someone in the water, s = 1.5, near 0.29); τ\_base = 12 h covers cold and exhaustion. Both in config.
- t\_affected = first time the person's depth exceeds 10 cm. People never affected keep survival 1 and are excluded from the at-risk metric.
- When found and confirmed, a rescue delay (default 15 min, bounds 10-30) passes, during which the dose keeps accumulating; then survival freezes.
- Unfound people keep accumulating until episode end + 2 h, so not finding someone is penalised.

Truth lives only in `World.people`; the planner never sees it.

## 11. Sensor and detection

A downward thermal camera reports, every 2 s frame, detect or no-detect for each 50 m cell in its footprint. Detection depends mainly on how many pixels a person covers, so it falls quickly with height and is much harder for people in the water.

**Camera (2076 assumption).** Uncooled long-wave infrared (LWIR), 1280 × 1024 pixels, 60° field of view. Today's SAR drones mostly carry 640 × 512 sensors, so this is a modest step rather than a leap. LWIR cannot see through glass, roofs or dense canopy, and rain attenuates it.

**Footprint.** Circle of radius r\_fp = z\_agl × tan(30°) around the nadir point. Frames taken above 200 m are skipped because detection there is negligible. The bank modifier switches detection off in steep turns.

**Pixels on target.** Ground sample distance GSD = 2 z\_agl tan(30°) / 1280 ≈ 0.0009 × z\_agl metres per pixel, and N = d\_target / GSD. Target size d\_target: 0.5 m for a person on land, a roof or a vehicle roof (shoulder width seen from above); 0.3 m for a person in the water (head and shoulders); groups use 0.5 × √size.

**True detection probability** (Johnson-style target transfer probability function):

```latex
p_d = 0.95\, \frac{(N/N_{50})^{E}}{1 + (N/N_{50})^{E}} \left(1 - 0.3\,\frac{r^2}{r_{fp}^2}\right) m_{bank}\, m_{speed}\, m_{rain}\, m_{contrast}\, m_{cover}\, m_{state}, \qquad E = 2.7 + 0.7\,\frac{N}{N_{50}}
```

N50 = 4 pixels, about double the lab figure for detection, to allow for cluttered flood scenes (debris, livestock, warm roofs). Clip p\_d to \[0, 1\].

| Height above ground | Person on land (0.5 m) | Person in water (0.3 m) |
| --- | --- | --- |
| 60 m | 0.93 | 0.73 |
| 100 m | 0.73 | 0.34 |
| 150 m | 0.41 | 0.13 |
| 200 m | 0.23 | 0.06 |

Values are at the footprint centre, before modifiers. Searching therefore happens at 60-150 m above ground, and people in the water need passes near 60 m.

| Modifier | Value |
| --- | --- |
| m\_bank | 1 up to 15° of bank, falling linearly to 0 at 30° |
| m\_speed (microbolometer motion blur) | 1 for ground speed ≤ 15 m/s, else 15 / ground speed |
| m\_rain (LWIR attenuation) | max(0.2, 1 − I / 40) with I in mm/h |
| m\_contrast | night 1.0; day 0.9 − 0.15 × (1 − cloud cover), since sun-warmed ground lowers contrast |
| m\_cover | open, farmland and water 1.0 (water is a cold, even background); urban 0.6 (warm surfaces); forest 0.2 (canopy blocks LWIR) |
| m\_state | thermal detection factor from the behaviour table (section 10) |

**False alarms.** Per frame, the number of false detections \~ Poisson(λ\_fa × cells in footprint), λ\_fa = 0.002 (bounds 0.0005-0.005), × 3 on farmland (livestock) and × 2 on urban cells (warm vents, engines, debris), placed uniformly in the footprint.

**Correlated looks.** Consecutive frames overlap almost entirely, so treating each as independent makes the belief overconfident. Each cell receives at most one belief update per 20 s (`min_revisit_s`); within that window keep the frame with the highest p̂\_d.

**What the planner's model knows.** The planner uses the same pixel model with an average target size of 0.4 m and an average m\_state of 0.8, plus the bank, speed, rain, contrast and cover modifiers it can know. It cannot know any person's state. The gap between p̂\_d and the true p\_d is deliberate.

**Output per frame:** `Observation(t, cell_idx: int array, p_hat: float array, detections: list[cell_idx])`.

**Swarm:** each drone carries its own camera. Every observation is tagged with the drone id and a sequence number, so it can be shared without double counting (section 12.1).

**Verification.** In VERIFY mode (70 m above ground, 60 s orbit), a true person in the cell is confirmed with p = 0.95 (otherwise reported as rejected); a false alarm is rejected with p = 0.97. A confirmed person is marked found with their find time and removed from the active truth list.

## 12. Bayesian belief grid

The belief is a Poisson intensity grid λ(c), the expected number of unfound people in cell c, which handles an unknown number of people with a one-line multiplicative update and no renormalisation.

**Why intensity rather than a single-target probability map.** There are several targets and their count is unknown. Under a Poisson assumption this is the PHD filter: a missed detection multiplies λ by (1 − p̂\_d), and the chance of at least one person in a cell is 1 − e^(−λ(c)).

**Prior** (built only from planner-visible maps):

```latex
\lambda_0(c) = \hat{N}\ \mathrm{normalise}\big(w_{pop}\,\mathrm{pop}(c) + w_{road}\,\mathrm{road}(c) + w_{farm}\,\mathrm{farm}(c) + w_{out}\,\mathrm{open\_near\_river}(c) + w_{exp}\,P_{flood}(c, t_0 + 1\,\mathrm{h}) + w_0\big)
```

N̂ is the planner's estimate of how many people are affected: the true count × lognormal noise (σ = 0.3). The weights are tunable parameters (section 13).

**Prediction step** (every 60 s):

1. Split λ into mobile (fraction m) and static parts.
2. Escape drift: move mobile mass uphill (along ∇HAND) at walking speed where P\_flood > 0.5, using a semi-Lagrangian backtrace with `map_coordinates(order=1)`.
3. Diffusion: Gaussian blur of the mobile part with σ = 0.5 × walk speed × 60 s / cell size.
4. Water drift: in cells with expected depth > 0.6 m and speed > 1 m/s, a fraction (0.5 per min × Δt) of the mass is advected along the observed flow velocity × drift fraction.
5. Conserve mass; track mass leaving the domain as `lost_mass`.

**Measurement update** for each observed cell (respecting `min_revisit_s`):

- No detection: λ(c) ← (1 − p̂\_d) λ(c).
- Detection (PHD update with clutter density κ = λ\_fa):

```latex
\lambda^{+}(c) = (1 - \hat{p}_d)\,\lambda(c) + \frac{\hat{p}_d\, \lambda(c)}{\kappa + \hat{p}_d\, \lambda(c)}, \qquad P_{real}(c) = \frac{\hat{p}_d\, \lambda(c)}{\kappa + \hat{p}_d\, \lambda(c)}
```

- The planner verifies when P\_real exceeds its threshold ν.
- Verification confirmed: λ(c) ← max(0, λ⁺(c) − 1). Rejected: λ(c) ← 0.05 λ(c).

**Planning aggregates:** macro-cells of 500 m (10 × 10 cells) holding Σλ and the expected-find sum used by the planner.

**API:** `Belief.from_prior(view, params)`, `predict(dt, forecast)`, `update(observation)`, `apply_verification(cell, confirmed)`, `p_any()`, `total()`, `macro()`.

### 12.1 Sharing the belief across the swarm

Drones share **observations, not beliefs**: each drone keeps its own belief grid and applies every observation exactly once, so merging never double counts a look.

- **Messages.** Each observation (drone id, sequence number, time, cell indices, p̂\_d values, detections) goes into the sending drone's log. Claims, detections, verification results and thermal sightings are messages too.
- **Radio model.** Two drones, or a drone and the base, can exchange messages when within the sampled radio range and terrain does not block the straight line between them. Each message is lost independently at the sampled loss rate; delivered messages arrive after 1 s.
- **Gossip.** Every 10 s each drone sends teammates in range the message ids it holds; they request the ones they lack, including messages relayed from drones now out of range. The base relays too.
- **Late observations.** Apply a received observation to the current belief as if it happened now, and skip it if its sequence number was already applied. This is a small approximation (prediction steps have moved some mass since) that keeps the code simple. Late detections still go through the P\_real check.
- **Prediction** runs locally on each drone on the same 60 s clock, so beliefs stay in step when comms are good and drift apart when they are not.
- **Perfect-comms flag** (`SwarmParams.perfect_comms`) gives every drone one shared belief, for ablations and for the centralised critic in section 14.1.

## 13. Planners

Three planners share one interface; the flood-aware planner adds an urgency term from the forecast and is the one whose parameters θ get tuned.

**Interface.** `Planner.reset(view)` and `Planner.decide(view) -> Goal`, called every `replan_s` and on events (target reached, detection, thermal exit, `cannot_progress`, RTB trigger). Goals: `Transit`, `Search`, `Verify`, `RTB`, plus a per-goal `allow_soaring` flag.

**Baselines**

1. **Lawnmower:** a fixed wind-aligned sweep over the floodplain (HAND < 10 m) at fixed height; ignores belief except to verify detections.
2. **Bayesian greedy:** the score below with λ\_f = λ\_h = 0, so pure expected finds per unit time.
3. **Flood-aware Bayesian:** the full score, tuned.

**Score for a macro-cell M**

```latex
S(M) = \frac{\sum_{c \in M} \big(1 - e^{-\hat{p}_d(c)\,\lambda(c)}\big)\, U(c,t)}{\big(T_{travel}(M) + T_{search}(M)\big)^{\beta}} - \mu\, \frac{E_{cost}(M)}{E_{remaining}}
```

```latex
U(c, t) = 1 + \lambda_f\, P_{flood}(c,\, t+L) + \lambda_h\, \frac{\mathbb{E}[HR](c,\, t+L)}{2}
```

- T\_travel = distance / expected ground speed along the track (cruise speed + along-track wind, minimum 2 m/s). T\_search = macro-cell area / (sweep width × search speed).
- E\_cost = level-flight power × (T\_travel + T\_search) at the planned speeds.
- Pick the best macro-cell, but only switch away from the current one if S\_new > (1 + hysteresis) × S\_current, to stop dithering.
- Inside the chosen macro-cell, fly a wind-aligned lawnmower; if one cell holds over half its λ, spiral out from that cell instead.
- Verify a detection when P\_real > ν.
- Search height: default to the h that maximises sweep width × p̂\_d(h), that is 2h tan(30°) × p̂\_d(h), which is about 80 m with the planner's 0.4 m average target; then add the tuned offset and clamp to the 60-150 m search band.

**Tunable parameters θ (`PlannerParams`)**

| Parameter | Range | Default | Controls |
| --- | --- | --- | --- |
| λ\_f flood urgency | 0-10 | 2 | Pull towards cells likely to flood soon |
| λ\_h hazard urgency | 0-5 | 1 | Pull towards dangerous water |
| L forecast lead | 0-60 min | 20 min | How far ahead urgency looks |
| β travel discount | 0.5-2 | 1 | Preference for nearby cells |
| μ energy penalty | 0-2 | 0.5 | Caution with battery |
| Prior weights w\_pop, w\_road, w\_farm, w\_out, w\_exp, w\_0 | 0-1 each | 0.3, 0.2, 0.1, 0.1, 0.3, 0.02 | Where people are believed to start |
| Mobile fraction m | 0-0.6 | 0.3 | Belief spread over time |
| Search height offset | −20 to +70 m | 0 | Footprint vs detection |
| Overlap | 0-0.4 | 0.15 | Sweep spacing |
| replan\_s | 30-180 s | 60 s | Replanning interval |
| Hysteresis | 0-0.5 | 0.15 | Target stickiness |
| w\_enter thermal entry | 0.3-2 m/s | 0.8 m/s | Soaring eagerness |
| E\_soar battery threshold | 0.3-0.9 | 0.6 | When soaring becomes allowed |
| MacCready M | 0-2 m/s | 0.5 m/s | Speed vs height trade |
| ν verify threshold | 0.1-0.9 | 0.4 | Time spent on false alarms |
| RTB reserve | 0.1-0.3 | 0.2 | Safety margin |

### 13.1 Swarm coordination

Each drone runs its own copy of the planner on its own belief, and drones coordinate through broadcast claims rather than a central controller, so the swarm keeps working when the radio drops out.

- **Claims.** When a drone picks a macro-cell it broadcasts a claim (cell, drone id, score, expiry = now + `claim_expiry_s`). Other drones skip cells they know are claimed. If two drones claim the same cell, the higher score keeps it (ties: lower drone id) and the other replans.
- **Spread term.** Subtract `spread_penalty` × (number of teammates' last-known positions and claims within `spread_radius` of M) from S(M), so drones fan out instead of chasing the same high-probability area.
- **Lawnmower baseline.** Split the floodplain into n strips of equal area, one per drone, and sweep each strip.
- **Launch.** Drones leave the base one at a time at the sampled launch interval; each first targets the best unclaimed cell.
- **Verification.** Only the drone that made a detection verifies it; teammates treat that cell as claimed until the result arrives.

**Extra tunable parameters (`SwarmParams`)**

| Parameter | Range | Default | Controls |
| --- | --- | --- | --- |
| claim\_expiry\_s | 60-900 s | 300 s | How long a claim blocks teammates |
| spread\_radius | 0.5-4 km | 1.5 km | Distance at which teammates repel |
| spread\_penalty | 0-2 | 0.5 | How strongly drones spread out |
| max\_per\_thermal | 1-3 | 2 | Thermal crowding |

## 14. Simulation loop and information boundary

`Sim` advances the glider every 0.5 s and runs slower processes on their own clocks; the planner only ever receives a `PlannerView`, never the world.

**Clocks**

| Process | Period |
| --- | --- |
| Glider physics, autopilot, safety layer | 0.5 s |
| Sensor frame and belief measurement update | 2 s |
| Trajectory logging | 2 s |
| People behaviour and hazard dose | 10 s |
| Belief prediction | 60 s |
| Stage forecast refresh / observed extent refresh | 300 s / 600 s |
| Planner decision | `replan_s` or on events |
| Grid snapshots for animation (only when `record_grids=True`) | 30 s |

**Loop sketch**

```python
def run(self) -> EpisodeResult:
    goal = self.planner.decide(self.view())
    while not self.done():
        w = self.world.weather.wind(*self.glider.pos, self.t) + self.turb.next()
        cmd = self.autopilot.step(self.glider, goal, self.sensed(w), self.view_static)
        self.glider = glider_step(self.glider, cmd, w, self.params, DT)
        self.check_termination()
        if self.tick(SENSOR_DT):  self.observe_and_update_belief()
        if self.tick(PEOPLE_DT):  self.world.people.step(self.t, self.world.flood)
        if self.tick(PREDICT_DT): self.belief.predict(PREDICT_DT, self.forecast)
        if self.planner_event() or self.tick(self.planner.params.replan_s):
            goal = self.planner.decide(self.view())
        self.t += DT
    return self.collect_result()
```

With a swarm, `self.glider` holds arrays over drones. Physics and separation are vectorised; sensing, belief updates, gossip and planner calls loop over the drones (at most 8).

**Public API**

```python
spec = sample_spec(seed, ScenarioBounds())
world = build_world(spec, cache_dir="cache/")        # deterministic, cached
sim = Sim(world, FloodAwareBayes(params), SimConfig(fast=True))
result = sim.run()                                  # EpisodeResult
```

**`PlannerView` (whitelist)**: time; its own glider state with GPS noise (σ = 2 m); battery fraction; netto variometer reading; mean-wind estimate; the belief; the `FloodForecast` object; static maps (DEM, surface, land cover, roads, population, HAND, depression depth); base location; teammates' last-heard positions, batteries and claims (only through the radio model); detections and verification results; rain-radar estimate (true rain × (1 ± 0.2)).

**Forbidden to the planner:** `world.people`, `FloodTimeline`, `Weather` internals, the thermal list, true detection probabilities. Build `PlannerView` by copying whitelisted fields only, and add a test that walks `vars()` of the view recursively and fails if any truth type is reachable.

**Episode ends** when time reaches the episode length, every drone has landed at base, crashed or made a forced landing, or (truth-side check) every target is found.

**`EpisodeResult`**: `spec_hash`, planner name and params, metrics (section 15), trajectory array, finds (person id, time, state at find), verification log, mode time breakdown, energy log, safety events.

### 14.1 Multi-agent RL interface (`marl_env.py`)

A PettingZoo `ParallelEnv` wraps `Sim`, so the team can train neural swarm policies on exactly the same scenarios, physics and information limits as the tuned planner.

- **Agents:** `drone_0` to `drone_7`. The swarm size is drawn per episode (2-8); unused slots are inactive agents (always masked, zero reward), so the agent list stays fixed for vectorised training.
- **Step:** the env advances the simulation until at least one drone needs a decision (target reached, detection, thermal exit, `replan_s` elapsed). Drones that do not need a decision have only `continue` unmasked. Expect about 150 decisions per drone per episode.
- **Action:** `Discrete(403)`: one of 400 macro-cells (20 × 20 grid of 500 m), plus `soar`, `verify_last_detection` and `continue`. Return-to-base stays automatic (the safety rule). Mask cells unreachable with the battery reserve and, optionally, cells claimed by teammates the drone has heard from.
- **Observation per drone:** a dict, every part normalised to roughly \[0, 1\].

| Part | Shape | Contents |
| --- | --- | --- |
| Local map | 7 × 24 × 24 | 3 km window at 125 m around the drone: belief 1 − e^(−λ), P\_flood at lead L, expected HR, terrain height, time since last observed, teammates' claims, teammates' positions |
| Global map | 7 × 20 × 20 | The same channels over the whole domain at macro-cell resolution |
| Self | 16 | Position, height above ground, airspeed, heading (sin, cos), battery fraction, netto, wind estimate, mode one-hot, time |
| Teammates | 7 × 8, plus mask | Last-heard relative position, height, battery, claimed cell, age of information, active flag |

- **Centralised critic state:** the perfect-comms belief plus every drone's true state. Only the critic sees this, never the actors (centralised training, decentralised execution).
- **Team reward each step:** the sum over newly observed cells of p̂\_d × λ × U (expected urgency-weighted finds, scaled to about 1 per episode); + 5 × survival-at-find for each confirmed find; − 5 per crash or collision; − 0.5 per separation breach; − 0.2 per false-alarm verification. Evaluate policies with survival\_at\_risk (section 15), the same metric as the planner.
- **Training recipe (outside the build):** start with parameter-shared PPO (one network for every drone, so it handles any swarm size) through SuperSuit and Stable-Baselines3; move to MAPPO with the centralised critic (for example BenchMARL or RLlib) for stronger results. Warm-start with behaviour cloning from the tuned planner's demonstrations.
- **Throughput:** about 1,200 decisions per 8-drone episode at about 6 s each, so roughly 1,600 decisions per second on 8 cores, or about 5 million per hour.
- **Datasets:** `record-demos --planner flood_aware --params best_params.json --seeds 0-199` writes (observation, action mask, action) tuples for behaviour cloning. `export-flood-dataset` writes (terrain, HAND, current extent, rain) → flooded-in-30-min pairs from the cache for a supervised flood forecaster.
- **Smoke test (`marl-smoke`):** PettingZoo's `parallel_api_test` for swarm sizes 2, 5 and 8, then 2,000 steps of random masked actions. If torch and stable-baselines3 are installed, also 10,000 steps of parameter-shared PPO.

## 15. Metrics, objective and tuning

The planner is tuned to maximise mean survival of at-risk people across many scenarios, with penalties for crashes, and judged once on held-out test seeds.

**Metrics per episode (`metrics.py`)**

- **survival\_at\_risk** (primary): mean P\_surv over people who were ever affected (section 10). Falls back to all people if none were affected.
- found fraction; time to first find; mean time to find (found people only).
- High-risk coverage: share of cells with P\_flood > 0.5 that were observed at least once in the first 30 min.
- Time spent verifying false alarms; energy used; share of time soaring; climb gained from lift versus motor.
- Safety: crashes, forced landings, geofence violations, stall count, terrain-warning count.

**Swarm metrics:** mid-air collisions, separation breaches, repeat coverage (share of observed cells seen by more than one drone within 10 min), share of messages lost, and time to first find by swarm size. In J(θ), \[crash\] becomes the fraction of drones lost.

**Objective**

```latex
J(\theta) = \frac{1}{|S|}\sum_{s \in S} \Big[\, \mathrm{survival\_at\_risk}_s(\theta) - 1.0\,[\mathrm{crash}_s] - 0.2\,[\mathrm{forced\ landing}_s] \Big]
```

Report mean ± standard error. Use common random numbers: every θ is evaluated on the same seed list.

**Seed splits**

| Split | Seeds | Use |
| --- | --- | --- |
| Train | 0-199 (30-60 per trial, stratified by flood type) | Tuning |
| Validation | 10,000-10,099 | Picking the best trial |
| Test | 20,000-20,199 | Final report, run once |

**Tuner (`tune.py`)**

1. `python -m floodsim build-cache --seeds 0-199 10000-10099 20000-20199` builds all worlds in parallel first.
2. Optuna TPE sampler if installed; otherwise Sobol random search (`scipy.stats.qmc.Sobol`) for the first half of the budget, then Gaussian perturbations around the best 5.
3. Each trial runs its scenario batch with `ProcessPoolExecutor` (one episode per task, workers load worlds from the cache).
4. Pick the best 3 trials by train score, re-score on validation, keep the best. Evaluate it and all baselines on the test seeds once.

**Rough budget on an 8-core laptop:** cache of 500 worlds at about 4 s each ≈ 4-5 min; 60 trials × 40 scenarios × about 4 s (swarms of 2-8) ≈ 20 min.

**Outputs** in `results/tune_<timestamp>/`:

- `best_params.json`, `trials.csv`, `test_metrics.csv`.
- `comparison.png`: survival\_at\_risk on test seeds for lawnmower, Bayesian greedy, flood-aware with defaults, flood-aware tuned (bars with 95% intervals).
- `per_flood_type.png`: the same split by flood type.
- `lambda_sweep.png`: survival versus λ\_f from 0 to 10 with other parameters at their tuned values. This is the headline pitch chart: it shows whether flood urgency helps.
- `forecast_brier.png`: forecast Brier score by lead time.

Also `survival_vs_swarm_size.png`: survival\_at\_risk on test seeds for 2 to 8 drones, tuned planner versus lawnmower. It shows where adding drones stops paying off.

If the flood-aware planner does not beat Bayesian greedy, report that honestly with the per-type breakdown; that is a finding, not a failure.

## 16. Visualisation and demo outputs

The demo needs one animated episode and one interactive 3D view; everything else is for debugging and the tuning report.

| Function | Output | Purpose |
| --- | --- | --- |
| `plot_world(world, t)` | PNG: hillshade, land-cover tint, flood depth in blues (alpha by depth), rivers, roads, towns, base, true people (debug flag) | Check scenarios look plausible |
| `plot_belief(view, t)` | PNG: belief heatmap (1 − e^(−λ)), glider track, footprint, detections | Debug the planner |
| `plot_wind_slice(world, z_agl, t)` | PNG: horizontal wind quiver over a vertical-wind colour map (ridge lift, thermals, rain sink) | Check updrafts |
| `animate_episode(result, world, path)` | GIF, 2 × 2 panels per 30 s of sim: truth flood + people; belief + track; altitude and battery over time; vertical wind at glider height | The main demo clip |
| `plot3d_html(result, world, path)` | Plotly HTML: terrain surface, translucent water surface where wet, 3D path per drone (hue per drone, line style by mode), radio links as faint lines, thermal columns as translucent cylinders | Interactive pitch visual |
| `tuning_report(results_dir)` | PNGs listed in section 15 | Results slide |

- GIF via pillow; also write MP4 if ffmpeg is on the path.
- If plotly is missing, fall back to a matplotlib 3D PNG.
- Keep the colour scheme consistent: flood depth blues, belief magenta, one hue per drone for tracks and footprints, with mode shown by line style (solid search, dashed transit, dotted thermal, thick verify).

**CLI**

```
python -m floodsim scenario --seed 3 --plot out/
python -m floodsim build-cache --seeds 0-199
python -m floodsim demo --seed 3 --drones 5 --planner flood_aware --params best_params.json --out out/
python -m floodsim evaluate --params best_params.json --split test
python -m floodsim tune --trials 60 --scenarios 40 --workers 8
python -m floodsim marl-smoke
python -m floodsim record-demos --planner flood_aware --params best_params.json --seeds 0-199
python -m floodsim export-flood-dataset --seeds 0-199 --lead-min 30
```

## 17. Tests and acceptance

Each module gets tests that check physics against known answers, so a plausible-looking but wrong simulation fails loudly.

| Module | Test | Pass criterion |
| --- | --- | --- |
| scenario | Sampler respects bounds over 200 seeds | Every field inside its range |
| scenario | Determinism | Same seed → identical arrays (hash) and identical `EpisodeResult` |
| terrain | Drainage | ≥ 80% of land cells drain to the outlet edge |
| hydrology | Filled DEM | No interior pits; HAND ≥ 0 everywhere and 0 on drainage cells |
| flood | Still lake | Flat water with no inflow: max \|q\| < 1e-6 after 1 h |
| flood | Mass conservation | Residual < 1% of volume in |
| flood | Dam-break sanity (1D strip) | Front advances monotonically, depth never negative |
| flood | Extent bounds over 50 seeds | Peak extent 3-40% of land in ≥ 90% of seeds |
| flood | Hazard rating formula | Hand-computed cases match |
| forecast | Informative but imperfect | Brier score at 30 min lead between 0.02 and 0.2 across 20 seeds; all P in \[0, 1\] |
| weather | Log profile | Speed increases with height up to 600 m |
| weather | Ridge lift sign | Positive windward, negative lee on a synthetic Gaussian hill |
| weather | Thermal shape | Peak at centre, negative ring; none at night or over flooded cells |
| weather | Turbulence | Sample std within 10% of target over 1 h |
| glider | Still-air glide at best-glide speed | Sink = V / (L/D) ± 3% |
| glider | Steady turn | Radius = V² / (g tan φ) ± 3% |
| glider | Climb and power | Full-throttle climb 2-4 m/s; 1 h level at 14 m/s uses 79 Wh ± 10% |
| glider | Stall and crash | Stall flag below stall speed; crash flagged flying into a hill |
| autopilot | Crosswind transit | Reaches a waypoint 5 km away in a 10 m/s crosswind |
| autopilot | Thermal centring | Entering 100 m off-centre of a 2 m/s thermal, averages > 1 m/s climb within 3 circles |
| autopilot | Terrain safety | No crash in 50 random runs towards a ridge |
| autopilot | RTB | Lands with ≥ reserve energy in ≥ 95% of 50 runs |
| sensor | Empirical detection rate | Matches p\_d within 2% over 10,000 trials; zero detections at \|φ\| > 30° |
| belief | Update maths | Miss update = (1 − p̂\_d) λ exactly; detection update matches the formula |
| belief | Prediction | Mass conserved to 1e-6 apart from tracked edge loss |
| sim | Information boundary | No truth object reachable from `PlannerView` |
| integration | 3 planners × 5 seeds | No exceptions; all metrics finite |
| performance (soft, skip with `FLOODSIM_SKIP_PERF=1`) | Timing | Uncached build < 8 s; cached episode < 2 s |
| swarm | Separation | No mid-air collisions and no pair inside 100 m horizontal and 30 m vertical in 50 random 8-drone runs |
| swarm | Observation sharing | Each observation is applied exactly once per drone; with unlimited range and no loss, every drone's belief equals the perfect-comms belief |
| swarm | Comms gaps | Beliefs diverge out of range; after reconnection and gossip, every message has been applied once by every drone |
| planners | Lawnmower partition | Strips cover the floodplain with no gaps, each within 5% of equal area |
| marl\_env | API and masks | parallel\_api\_test passes for 2, 5 and 8 drones; masked actions are never executed; inactive agents get zero reward |

**Project-level acceptance:** the tuning run completes and produces every output in section 15. Whether the flood-aware planner wins is a result to report, not a test.

## 18. Build phases (about 3 hours)

Ten phases totalling about 215 minutes (the swarm and RL interface add about 35), each ending with passing tests, a `PROGRESS.md` entry and a commit; every phase has a fallback so the pipeline stays runnable end to end.

| Phase | Budget | Build | Done when | Fallback if over budget |
| --- | --- | --- | --- | --- |
| 0. Scaffold | 10 min | `pyproject.toml`, package skeleton, all config dataclasses and bounds, RNG streams, CLI stub, `PROGRESS.md`, `ASSUMPTIONS.md` | `pytest` runs; `python -m floodsim --help` works | none needed |
| 1. Terrain and hydrology | 25 min | Section 5 | Terrain and hydrology tests pass; `scenario --plot` shows terrain, river, towns, roads | Straight noisy roads, no tributaries |
| 2. Flood truth | 30 min | Section 6.1: solver, sources, cache, hazard rating | Flood tests pass; flood-only GIF for seed 0 looks plausible | HAND-based truth model (section 6.1 fallback) |
| 3. Forecast | 10 min | Section 6.2 | Brier test passes | Drop pluvial forecasting; fluvial only |
| 4. Weather | 20 min | Section 7 | Weather tests pass; wind slice plot shows ridge lift and thermals | Drop turbulence and showers |
| 5. Glider, autopilot, safety | 35 min | Sections 8 and 9, vectorised over drones, with separation | Glider and autopilot tests pass | Thermal circling without centring; fixed 25% RTB trigger |
| 6. People, sensor, belief | 25 min | Sections 10, 11, 12 including 12.1 (shared observations) | Their tests pass | People static except roof transitions; belief without water drift |
| 7. Planners, sim loop, metrics | 25 min | Sections 13 including 13.1 (claims), 14, 15 (metrics) | Integration tests pass; `demo` GIF for seed 3 | Skip spiral-out; fixed search height |
| 8. Tuning and demo outputs | 15 min | Section 15 (tuner), section 16 | Small tune run (10 trials × 10 scenarios) writes all outputs; 3D HTML renders | Random search only; matplotlib 3D PNG |
| 9. Multi-agent RL interface | 20 min | Section 14.1: ParallelEnv, observations, masks, rewards, record-demos, export-flood-dataset | marl-smoke passes for 2, 5 and 8 drones | Fixed 8-drone swarm; random-action smoke test only |

**Order of work inside each phase:** write the dataclass and function signatures, write the tests from section 17, implement until the tests pass, then make the phase's plot and look at it. A plot that looks wrong (water flowing uphill, a glider sinking 5 m/s in still air) is a bug even if tests pass.

**After Phase 9:** run the full tune (section 15 budget), write `README.md` with the demo commands and the headline numbers, then start stretch goals.

## 19. Stretch goals

Only after Phase 9, roughly in order of pitch value per hour of work:

1. **Coastal and estuary surge** with 2076 sea-level rise (0.3-1.2 m) and a tide: one domain edge becomes coast with a time-varying water level boundary.
2. **Heterogeneous swarm**: long-endurance high scouts plus low, slow verifiers with sharper cameras, and a planner or policy that assigns roles.
3. **Ridge-soaring mode**: fly figure-eights along windward slopes where the planner's expected ridge lift exceeds 1 m/s.
4. **Real terrain**: load a GeoTIFF DEM (for example Environment Agency LIDAR or Copernicus) with `rasterio`, and a real river line, for a demo over a real UK town.
5. **Communication relays**: a drone holds height to relay messages for teammates beyond radio range, chosen by the planner or the learned policy.
6. **Full dynamic point-mass model** (flight-path angle as a state, RK4 at 0.05 s) to validate the quasi-steady model.
7. **Rescue logistics**: a boat with travel time over water, so rescue delay depends on where people are found.
8. **Learned flood forecaster** trained on the `export-flood-dataset` output, replacing the HAND forecast and compared by Brier score.

## 20. References

Methods this plan draws on; cited from memory, so check titles before quoting them in the pitch.

- Bates, Horritt and Fewtrell (2010), a simple inertial formulation of the shallow water equations for 2D flood inundation modelling, *Journal of Hydrology*. Basis of the flood solver.
- Barnes, Lehman and Mulla (2014), Priority-Flood depression filling, *Computers & Geosciences*. Depression filling.
- Rennó et al. (2008), HAND terrain descriptor, *Remote Sensing of Environment*. Height above nearest drainage.
- Defra and Environment Agency (2006), Flood Risks to People (FD2320/FD2321). Hazard rating formula and classes.
- Allen (2006), updraft model for autonomous soaring UAVs, NASA. Thermal shape and vertical profile.
- MIL-F-8785C / MIL-HDBK-1797. Dryden turbulence model.
- Mahler (2003), multitarget Bayes filtering via first-order moments (the PHD filter), *IEEE Transactions on Aerospace and Electronic Systems*. Intensity-grid belief.
- Stone (1975), *Theory of Optimal Search*. Bayesian search theory behind the planner.
- Reichmann, *Cross-Country Soaring*. Thermal centring and MacCready speed-to-fly.
