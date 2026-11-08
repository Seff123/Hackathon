# Assumptions

## Known simplifications (from the plan, section 2)

- The camera is assumed stabilised (gimballed), so banking does not affect detection, even at 35-45° in turns.
- Thermals can sit over floodwater; only the river channel gets zero weight.
- Soaring rarely pays off: flood weather is cloudy, the median w* is about 0.6 m/s, and a typical thermal core (about 0.7 m/s) barely beats the circling sink (about 0.6 m/s).
- Wind is constant per scenario, with one roughness everywhere and no turbulence or gusts.
- Wind is treated as quasi-steady; wind-gradient (dynamic soaring) terms are ignored.
- Episodes run in daylight only.
- The radio is range-only: no terrain blocking, loss or latency.
- The base zone has no separation or collision checks.
- Bank and airspeed change instantly, and the glider uses a sink-rate table.
- People walk straight to higher ground, and drift in water is deterministic.
- Survival depends on danger time rather than a full hazard-dose model.

## Decisions taken where the plan was ambiguous

- Python 3.14 is used (the plan says 3.11+).
- Levee breach scenarios are built now (team decision), not deferred.
- Breach scenarios sample the levee height from 1-3 m (not 0-3 m) so there is always a levee to breach.
- Pluvial-only scenarios keep the river at base flow (no flood hydrograph). Fluvial-only scenarios have no rain.
- Rain correlation with cloud cover: the uniform draw inside the rain bounds is blended 50/50 with the normalised cloud cover, so rain stays inside its bounds and rises with cloud.
- Random streams: `SeedSequence([seed, attempt]).spawn()` with a fixed, named stream order.
