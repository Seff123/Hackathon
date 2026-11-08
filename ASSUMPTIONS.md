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
