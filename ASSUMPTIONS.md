# Assumptions and simplifications

## Known simplifications (from plan section 2)

- The camera is assumed stabilised (gimballed), so banking does not affect detection, even at 35-45° in turns.
- Thermals can sit over floodwater; only the river channel gets zero weight.
- Soaring rarely pays off: flood weather is cloudy, the median w* is about 0.6 m/s, and a typical thermal core (about 0.7 m/s) is weaker than the team airframe's circling sink (about 1.4 m/s at 35 deg bank; it was about 0.6 m/s for the plan's 5 kg placeholder).
- Wind is constant per scenario, with one roughness everywhere and no turbulence or gusts.
- Episodes run in daylight only.
- The radio is range-only: no terrain blocking, loss or latency.
- The base zone has no separation or collision checks.
- Bank and airspeed change instantly, and the glider uses a sink-rate table.
- People walk straight to higher ground, and drift in water is deterministic.
- Survival depends on danger time rather than a full hazard-dose model.
- Wind is quasi-steady for the glider; wind-gradient (dynamic soaring) terms are ignored.

## Choices made during the build (where the plan was ambiguous)

- Team inputs (2026-11-08): mass 1.5 kg and radio range 8 km from the team; 15 m/s is the cruise and the battery-life reference airspeed (V_ref, was 13 m/s). Battery life (150 min), charge time (10 min) and camera range (140 m) are still the plan's placeholders.
- Team airframe: 1 m span with best glide at 15 m/s (team values). Derived, with the plan's cd0 0.018, Oswald 0.85 and CLmax 1.3: wing area 0.237 m^2 (aspect ratio 4.2, best L/D 12.5, stall 8.8 m/s); thrust 4.2 N (the plan's thrust/weight ~0.29, 3.1 m/s climb at 15 m/s); normal climbs at 15 m/s; payload + avionics 10 W (the plan's 25 W suits a 5 kg airframe). 150 min at 15 m/s gives a ~93 Wh battery. Circling sink at 35 deg bank is ~1.4 m/s, so soaring only pays in strong thermals.
- Tests that assumed the 5 kg placeholder now compute their targets from the configured airframe; the thermal-climb test uses a core 2 m/s stronger than the configured circling sink.
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
- Flood extent check: the cheap HAND estimate cannot predict the solver well (about 50% precision on 40 calibration seeds), so it is used as a loose pre-filter (2-60% of land) and the solver's own peak extent (episode start to end + 2 h) must lie in 3-40% before a seed is accepted. The solver result is reused by build_world, so accepted seeds cost nothing extra; rejected attempts cost one solver run (~1.5 s).
- Flood depth on the 50 m grid: the 100 m solver depth is downscaled per 100 m block using the lower of the solver water surface and the level that holds the block's water volume over the 50 m terrain. This keeps land beside the channel dry at base flow and stops thin sheet flow on slopes showing as deep pools.
- The river channel is narrower than a 100 m flood cell: faces between two channel cells pass channel_width_m / 100 of the per-unit-width discharge (continuity only); levee breaches use breach_width_m / 100 the same way.
- Free outflow along the whole outlet edge (normal depth with the local surface slope, floored at the river slope); only wet cells contribute. Unit discharges are capped at Froude number 1 for stability.
- River inflow is spread over the first 3 channel cells of the flood grid. Storms start at flood onset, are centred on a random town (plus 1 km noise) at mid-storm, move at the 10 m wind speed (direction = wind +/- 30 deg) and have a sin^2 time envelope.
- Breach: the levee cell (among the 5 nearest a town centre) is lowered by the head drop at breach time (episode start + breach time).
- Forecast stage error: a 2D Gaussian random field with unit variance and 2 km correlation length (corr(2 km) = 1/e), sampled at drainage cells, stands in for the AR(1) profile along the river; it is equally smooth along the river and also covers tributaries.
- Forecast stage at a dry drainage cell is -0.5 m (before bias and noise), and P_flood uses the same 0.1 m wet threshold as the truth: P = Phi((S - HAND - 0.1) / sigma). Without these, dry tributary cells read P = 0.5.
- Pluvial forecast: ponding estimate = cumulative forecast rain x runoff coefficient x min(upstream cells, 25), compared with max(depression depth, 0.1 m) + 0.1 m with sigma/2, only where depression depth > 5 cm. The rain forecast is the true storm with centre noise (500 m), speed noise (10%) and intensity x (1 +/- 0.2).
- Brier score is computed over land cells with HAND < 10 m (the floodplain zone) and averaged over 5-min issue times; scoring all land would be dominated by trivially dry hillsides. The 5% observed-extent errors with the 0.95 floor alone contribute about 0.04.
- Thermal lookup is a vectorised loop over all hotspots (at most ~30), not a spatial grid; it is cheaper in numpy at this count.
- Autopilot transit airspeed: 15 m/s + half the headwind, but at least |crosswind| + 5 m/s (capped at 25 m/s); without the crosswind floor a 12 m/s crosswind forced a 75 deg crab and halved ground speed. The RTB energy rule uses the same airspeed and the crab-corrected ground speed, and adds the payload energy of the landing orbit ((height above base) / 1 m/s x payload power).
- RTB also triggers whenever battery <= 1.1 x reserve, even if a glide home looks possible (otherwise a loitering drone could drain its battery while "able to glide").
- Altitude-band climbs use max(climb_speed, headwind + 8 m/s), capped at the fastest-climb speed, so a climb into a strong headwind still makes progress (energy per metre climbed is nearly the same).
- Separation: every drone in a conflict steers each second towards (its heading turned right 30 deg) + 1.5 x (unit vector away from each conflicting drone's closest-approach position); a fixed 30 deg right turn alone left 20 breaches in 50 busy 8-drone runs, mostly where three drones met; vertically, each connected conflict cluster keeps its lowest-id drone at its height and stacks the others 50 m apart above the cluster's lowest drone in id order (the plan's higher-id rule generalised to 3+ drones), climbing at the fastest-climb speed, tracked every second. Manoeuvres hold 15 s after the last conflict.
- Launch: a catapult releases the drone 30 m above the pad (the launcher itself is not modelled); catapults are spaced evenly around an 80 m orbit over the base; each drone spirals up that orbit at full throttle until it reaches its own transit layer above the base (layers are 50 m apart by drone id, then heads out; a single catapult crew launches the drones 20 s apart in id order at the start, so drones in the spiral are ~58 m apart vertically), then climbs straight out towards its first goal until it reaches its transit layer (bases can sit on 0.3 slopes, where a straight-out climb from the pad hit the hillside). The terrain floor is not enforced during the pad orbit, and warnings are not counted while launching inside the base zone. Thermal circling is allowed only from TRANSIT, SEARCH or LOITER; the "next transit needs more height" soaring condition is not implemented.
- When a goal is finished (transit arrival, pattern done, verification done, thermal exit without a goal) the drone loiters in a 150 m orbit at its current height above the surface until the controller gives a new goal.
- The autopilot's thermal-top check uses the scenario's boundary-layer depth zi (treated as known from a sounding).
- Separation conflicts are detected by the plan's rule (inside 300 m and 40 m) and also predictively: closest point of approach within 45 s closer than 250 m horizontally and 40 m vertically (pairs within 1.5 km). The swarm test flies 8 drones on random crossing routes.
- Base zone: within 500 m of base and below 150 m above it, plus any drone within 500 m that is in its landing orbit or in the launch spiral (otherwise drones arriving high share the landing orbit with separation active and can livelock).
- People swept off the map by drift are "lost": they cannot be found and count with survival 0.
- Camera range for a person on a roof is measured to the roof top (terrain + obstacle height); the footprint uses the camera's height above the ground at the nadir point. Cover modifier comes from the person's cell land cover.
- Verification: when the 60 s orbit ends, the first unfound person currently in the cell is confirmed with p = 0.95; an empty cell is rejected with p = 0.97 (otherwise a false confirmation is logged).
- Radio: a message sent at t is delivered at the next 1 s radio tick to every node (drones in the air or on the pad, plus the base) within signal_radius of the sender at that moment; out-of-range targets never receive it (no store-and-forward in the environment).
- Valley walls are widened where needed so their slope stays below 0.3 (at 50 m resolution), since the glider's best climb gradient is about 0.2 and gorges on high-relief seeds otherwise reached 0.5.
- The terrain-escape manoeuvre latches: once triggered it keeps full-throttle climb at the fastest-climb speed and its initial turn direction until the look-ahead clears the floor by 20 m for 10 s.
- Scripted search legs are clipped to 350 m inside the domain, and the geofence turn latches until the drone heads clearly inward (cos > 0.7) or leaves the 300 m edge zone.
- Terrain look-ahead predicts along the current turn (constant bank) plus wind, not a straight line, so orbits do not raise false warnings.
- The RTB energy rule is evaluated every 5 s (the hard 1.1 x reserve floor every second) to keep the 8-drone episode near the 6 s budget; battery state changes by < 0.15 Wh in 5 s.
- Metrics are always finite: time to first find and mean time to find are censored at the episode length when nobody is found, and high-risk coverage is 1.0 when no cell floods in the first 30 min.
- Performance: on the development laptop (8 GB RAM, Python 3.14) a fast-mode 2.5 h episode takes about 7 s with 8 drones and 4.5 s with 1 drone, against the plan's 6 s / 2 s budget. Per-step cost is dominated by numpy call overhead on tiny (n <= 8) arrays in the vectorised autopilot; the timing test is soft (xfail with the measured numbers unless FLOODSIM_STRICT_PERF=1).
- Separation manoeuvres ignore only drones in the base zone that are launching or in the landing orbit; other drones near the base still avoid each other (breaches are still not counted inside the base zone, as in the plan).
- Launch climbs fly at max(climb speed, wind estimate + 5 m/s) (capped at the fastest-climb speed) so the spiral holds position in strong wind aloft, 
