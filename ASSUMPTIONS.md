# Assumptions and simplifications

## Known simplifications (from the build plan, section 2)

- The camera is assumed stabilised (gimballed), so banking does not affect detection, even at 35-45° in turns.
- Thermals can sit over floodwater; only the river channel gets zero weight.
- Soaring rarely pays off: flood weather is cloudy, the median w* is about 0.6 m/s, and a typical thermal core (about 0.7 m/s) barely beats the circling sink (about 0.6 m/s).
- Wind is constant per scenario, with one roughness everywhere and no turbulence or gusts.
- Episodes run in daylight only.
- The radio is range-only: no terrain blocking, loss or latency.
- The base zone has no separation or collision checks.
- Bank and airspeed change instantly, and the glider uses a sink-rate table.
- People walk straight to higher ground, and drift in water is deterministic.
- Survival depends on danger time rather than a full hazard-dose model.
- Wind is treated as quasi-steady; wind-gradient (dynamic soaring) terms are ignored.

## Choices made where the plan was ambiguous (simpler option taken)

### Terrain (phase 1)
- Terrain is built relative to a plane sloping to the outlet: valley floor + valley walls (smoothstep) + a gentle cross-valley ramp + spectral noise damped near the river. This guarantees the river valley drains to the outlet edge, so check 6 rarely rejects.
- The floodplain occupies the inner 75% of the valley width and slopes gently towards the river ((0.25 × regional slope + 0.0005) per metre), with ±(floodplain noise / 2) undulation.
- The river centreline is rasterised 4-connected at both 50 m and 100 m so the flood solver's face-based fluxes can follow it. The coarse (100 m) channel bed is forced monotone.
- Town centres are sampled within 1.5 km of the river weighted towards low slope near the valley edge, at least 2 km apart (relaxed to 1 km if needed). Roads run from each town to a base site (highest cell within 2 km of the largest town's edge); the base is then the highest road cell there. Obstacles are cleared within 150 m of the base.
- Breach scenarios force a levee height of at least 1 m so there is something to breach.
- Rain intensity is scaled by cloud cover (0.7-1.3×, clipped to the peak-rain bounds) to implement "rain intensity scales up with cloud cover".
- Flood type and swarm size are drawn from a stream that depends on the seed only (not the rejection attempt), so stratified seed lists stay stable.
- HAND for the forecast uses the full drainage network (as planned); the cheap rejection estimate uses height above the *main river* along the flow path (`hand_river`).
- `lambda_fa` is fixed at the 0.002 default per scenario (bounds kept in config for sensitivity runs).

### Flood (phase 2)
- On channel faces the solver uses an effective Manning n = n_channel × Δx / channel width, so a 100 m cell conveys like the real 20-80 m channel.
- All outlet-edge cells (not only the channel) are free-outflow at normal depth; otherwise floodplain water ponds against the outlet wall.
- 50 m depth is downscaled from the 100 m solver by comparing the interpolated water surface with the 50 m terrain (capped by the interpolated depth off-channel). Plain linear upsampling smeared channel depth onto the banks and made every scenario look ~2% flooded.
- The breach lowers the levee cells nearest the largest town by the head drop (not below bank − 1 m).
- The storm moves along the wind direction ± the sampled deviation at 0.5 × U10, clipped to 1-5 m/s (the plan gives no speed; faster storms dropped only ~20 mm anywhere and never produced surface flooding). Its envelope is a half-sine over its duration starting at flood onset; it passes the domain centre (± 20%) at mid-duration. Fluvial-only scenarios have no in-domain rain; pluvial-only scenarios keep the river at base flow.
- The cheap rejection estimate (no solver) = river stage from a compound-channel Manning curve against height above the river, plus pluvial fill-and-spill ponding on the 100 m DEM. Conveyance factors (channel 0.7, floodplain 0.3) were calibrated against the solver, and the estimate must clear the 3-40% band by a 1.45× margin. On seeds 0-49, 47/50 full-solver peak extents fall within 3-40%.

### Forecast (phase 3)
- A cell counts as flooded when depth > 0.1 m, so P_flood = Φ((Ŝ − HAND − 0.1)/σ).
- Forecast error scales with drainage size: f = 1 on the main river, √(upstream area / 25 km²) (≤ 1) on tributaries, with σ floored at 0.05 m. Without this, every dry ditch would sit at P ≈ 0.5. `stage_offset` is added as offset × f.
- Pluvial forecast: pond depth = min(depression depth, forecast rain × runoff coefficient × min(upstream cells, 20)) in depressions deeper than 5 cm; P = Φ((pond − 0.1)/(0.1 + 0.3·pond)); combined with fluvial as 1 − (1 − P_f)(1 − P_p). Forecast rain = true accumulation × one scenario-wide factor (1 + 0.2 N(0,1)).
- Observed velocity noise is per cell, factor clip(1 + 0.3 N(0,1), 0, 2), refreshed every 10 min.
- The Brier score is computed over land cells at issue times every 30 min. It is dominated by the 5% of cells the observed extent misreports.
- Forecast products are rebuilt from the cached flood on load (deterministic), not cached separately.

### Weather (phase 4)
- Thermal background sink (≈ −0.01 m/s) is omitted. Hotspots are checked by a vectorised distance over all of them (at most 30) instead of a spatial grid.

### Glider, autopilot and safety (phase 5)
- The autopilot runs per drone in scalar Python (at most 8 drones); physics and separation are vectorised across drones.
- The autopilot uses the drone's own wind estimate (10 m reference vector + fixed N(0, 1 m/s) error per component per drone, scaled with the log law to its height), never the true wind.
- Orbits (thermal, verify, base) use a vector-field orbit tracker with a bank feed-forward instead of a fixed bank, so circles hold their position in wind. Thermal circling uses a 40 m orbit around the point where lift was first felt, at 1.2 × stall speed at 35° bank (no centring).
- The soaring height limit (below ceiling − 100 m) is checked when lift is first felt; "next transit needs more height" is not modelled.
- Search altitude hold targets the highest surface at 0, 5 and 10 s ahead + the search height; more than 20 m high → glide down at 20 m/s; more than 30 m low → full-throttle climb.
- RTB flight: level at ≥ 80 m above the surface/base until high enough to glide home with a 25% margin, then glide; within 300 m of base a 70 m descending orbit, landing on touch-down within 100 m. The return-energy rule is skipped while a glide home is possible *and* the battery is above reserve + 2%.
- The geofence turns back only when the drone is within 300 m of the boundary, heading outward and predicted to leave the domain within 10 s (latched 15 s). Without the prediction, waypoints near the edge made the geofence and the mode fight and the drone drift out.
- Launch: drones launched together hold over the base (150 m orbit, 80 m above base) for 20 s × launch slot before climbing straight out, so departures are spaced.
- Separation: conflicts start inside 300 m horizontally and 40 m vertically, or when the closest point of approach within 20 s is inside 300 m; they last until the pair is 400 m apart horizontally. Who is above is latched when the conflict starts; the upper drone gets an altitude floor 50 m above the lower one's effective altitude (floors propagate up through clusters). Drones never get pushed down, so avoidance can't drive them into terrain. The upper drone turns right while closing within 40 m vertically; both turn right while within 25 m vertically. Avoidance also runs inside the base zone (except for drones holding before departure or on final approach); breaches and collisions are only counted outside it. A separation floor overrides the ceiling throttle cut. With these rules, 50 random 8-drone runs show no breaches.

### People, sensor, radio (phase 6)
- The footprint radius uses the camera height above the terrain at the nadir point. True detections use the camera range to each person (roof height for people at home or on roofs).
- Verification matches the nearest unfound person within 75 m of the verified cell centre (people may have moved since the detection).
- A person swept off the map has survival 0.
- "At risk" (ever affected) is also tracked for people after they are found: whether the water later reaches the place they were found. Without this, finding someone before the water arrives would remove them from the at-risk set and could *lower* survival_at_risk.
- The radio delivers to nodes in direct range; `connected_to_base` (multi-hop) is exposed for the planner's base_relay comms.

### Sim and metrics (phase 7)
- Time to first find / mean time to find are reported as the episode length when nobody is found, so every metric is finite.
- The rain-radar estimate in the view is the true storm rain (100 m, mm/h) × one scenario-wide factor (1 ± 0.2 N), refreshed every 5 min.
- PlannerView exposes `n_hat`, `episode_end_s`, `horizon_s`, `connected_to_base` and the drone's own autopilot status (mode, goal kind, waypoint index, E_return).

## Planner (PLANNER_PLAN.md)
- P0 hooks: `Controller.observe(views_all)` every sensor tick; PlannerView carries `n_hat`, `episode_end_s`, `horizon_s`, `connected_to_base`; static maps include `building_density`, `road_mask` and `dist_to_river`; FloodForecast gained `observed_extent`, `sigma`, `stage_offset=`, `fluvial_terms` (bias learning) and `products_all_leads` (vectorised urgency tables).
- Forecast depth: observed-wet cells also get an extent-derived depth (FwDET-style: highest flood-edge elevation within ~250 m minus the DEM, after a 3×3 majority filter), used as max(HAND depth, extent depth) with a 0.3 m sd in the hazard terms. Pure HAND underestimated depth badly where water comes from the main river or a breach but the local drainage is a dry tributary. It is still imprecise (MAE ≈ 0.6 m), so the "value aligns with scoring" test is marked as an expected failure: at the few finds where truth survival dropped (people on roofs in breach/backwater water), the planner's danger estimate lags.
- Scoring units: S = B (1 − claimed) / T_minutes^β − μ (E_cost / E_now) × median(B/T). The energy penalty is scaled by the median value rate so μ is dimensionless. With T in seconds and no scaling, the energy term was ~100× the value term and every planner just picked the nearest block.
- Camera range in the planner's look update: drone altitude − DEM (ground components) or − surface (home components, on roofs). Revisit windows open at the first look and apply the best p̂_d after 20 s.
- Claims cover only the chosen block (not the blocks the transit path crosses at search height); transit happens at the separate transit layers.
- Urgency tables: danger rates are evaluated at 10-min leads (0-60 min) and the 5-min steps are linearly interpolated; P(HR > θ) is computed at 50 m and block-averaged to 100 m. This halves the table cost.
- Prediction: the mobile part (fraction m) of the open component is kept as its own grid; only mobile mass in cells with P_flood > 0.5 moves (forward bilinear push uphill along ∇HAND at 0.9 m/s, diffusion σ = 0.5 × 0.9 × dt / 50 m, and a 0.5/min share riding fast deep water). The push is mass-conservative; mass leaving the domain goes to `lost_mass`.
- Bias learning: the learned b is the correction added to the forecast stage, so a forecast biased +0.3 m is learned as b̂ ≈ −0.3 m.
- base_relay comms: a drone's frames enter the fleet belief only while it has a multi-hop radio path to base; otherwise they are buffered with the drone's altitude at frame time. Verification results and decisions are not delayed.
- Soaring is allowed when battery < E_soar and the best block's score is below the median of the last 10 decisions (fleet-wide history).
- If no block is feasible (energy), the planner sends RTB.
- Planner overhead is ≈ 1.8 s (bayes) and ≈ 2.8 s (survival) per cached 8-drone fast episode; the performance test allows 3 s.
