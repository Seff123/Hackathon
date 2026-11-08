# Assumptions and simplifications

## Known simplifications (from plan section 2)

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
- Wind is quasi-steady for the glider; wind-gradient (dynamic soaring) terms are ignored.

## Choices made during the build (where the plan was ambiguous)

- Team inputs are the plan's placeholders (5 kg, 150 min, 10 min swap, 3 km radio, 140 m camera).
- Breach scenarios force the sampled levee height to at least 1 m, so there is always a levee to cut.
- Rain intensity correlation with cloud cover: peak = lo + (hi - lo) x (0.5 u + 0.5 c), with c the cloud cover normalised to its bounds and u uniform, then x climate factor.
- The episode length is a fixed 2.5 h bound (the sampler supports widening it to 2-3 h).
- RNG streams are keyed by (seed, attempt, fixed stream id), so adding a stream never shifts the others.
- The world cache file is `cache/<spec_hash>.npz`; it also stores a hash of the world, forecast, weather and people config and is rebuilt when that changes.
- Terrain: non-outlet domain edges get a raised rim (0.35 x relief over 2.5 km) so the domain behaves as one catchment draining to the outlet edge.
- Terrain: closed basins are filled with a gentle gradient (0.025 m per cell, so they drain towards their spill point) and only ponds up to 0.5 m deep are kept (z = max(z, z_fill_grad - 0.5 m)); spectral noise otherwise leaves deep basins that real fluvial landscapes do not have. Shallow ponds remain for pluvial flooding.
- Terrain: the river's centreline elevation is a straight line from inflow to outlet placed below the land everywhere along the path (slope at least 0.0005), so the valley can become a gorge on high-relief seeds.
- Depression filling uses priority-flood + epsilon; flow directions are D8 steepest descent on the filled DEM (float64), so every cell reaches an edge.
- Levees are built on the 100 m flood grid (cells next to the coarse channel within 1 km of a town) and mirrored onto the 50 m grid, so they are 100 m wide.
- Roads: links between towns plus 2 radial roads per town (the first heads away from the river). The base is the highest road cell within 2 km of the largest town, at least 600 m from the domain edge; obstacles within 150 m of the base are cleared as a landing area.
- Spectral slope beta (1.8-2.4) is read as the 1D profile slope, the usual convention for terrain, so the 2D amplitude filter is f^-(beta+1)/2. With f^-beta/2 in 2D the 50 m terrain had a median slope of 12% and pits in 20% of cells.
