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
