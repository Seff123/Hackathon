# FloodSim planner build plan: survival-aware Bayesian search

Nov 8, 2026 · companion to `FloodSim_Glider_SAR_Environment_Build_Plan.md` (the "environment plan")

## 0. How Claude Code should use this plan

Build this **after** the environment's Phase 7 (sim loop, scripted controller, metrics) passes. The environment's Phase 8 (visualisation) can wait; `VIZ_PLAN.md` replaces most of it.

**Relationship to the environment plan's appendix:**
- This plan **replaces appendix A1–A3**. It keeps the Poisson intensity belief, the claims and the tuning set-up.
- It keeps the appendix planner (`U = 1 + λ_f P_flood + λ_h HR/2`) as a **baseline**, so the comparison is fair.
- It changes four things, each for a reason found in testing or in the environment's own rules (section 1).

**Working rules** (same as the environment plan):
- numpy first, every number in a config dataclass, determinism per seed, no truth access.
- After each phase: `pytest`, a `PROGRESS.md` entry, commit `planner phase N: <name>`.
- If a phase fails its tests after a second focused attempt, take its fallback (section 15) and log it in `ASSUMPTIONS.md`.

**Definition of done**
1. `python -m floodsim evaluate --controller survival --seeds 20000-20099` runs and writes metrics.
2. `python -m floodsim compare --controllers scripted,bayes,heuristic,survival --seeds 20000-20099` writes `results/compare_<ts>/` (section 13).
3. All planner tests pass. The information-boundary test still passes.
4. The planner adds at most about 2 s to an 8-drone fast-mode episode.

Appendix B of this file contains reference code for the trickiest maths. It was **verified on synthetic data**: copy it, don't re-derive it.

---

## 1. The idea, and what changed from the appendix

**One sentence:** each glider flies to the 500 m block where **searching now saves the most expected survival per minute**, compared with leaving those people for ground teams after the episode. Its picture of where people are learns, from what the fleet has and hasn't found, which kind of behaviour this flood's population followed.

**What changed and why**

| # | Appendix plan | This plan | Why |
|---|---|---|---|
| 1 | Urgency as a bonus multiplier: `U = 1 + λ_f P_flood(t+L) + λ_h HR/2` | **Marginal survival**: survival if confirmed at arrival time minus survival if never found, from the forecast's hazard over time | The bonus keeps dry towns at full value and has a cliff at the single lead L. The environment scores `survival_at_risk`, so the planner should maximise exactly that. In our earlier simulator this change took the planner from no better than Bayesian search to **+18–22% lives saved**. The bonus version stays as a tuned baseline. |
| 2 | One detection model: average target 0.4 m, average m_state 0.8 | **State-aware detection**: people at home are nearly invisible (0.02) until the water reaches 0.5 m and they climb onto the roof (1.0). People in vehicles are 0.2. | With a constant 0.8, the belief would mark a dry town as "searched" after one pass, when only about 1% of the people indoors could have been seen. That error would persist all episode. |
| 3 | Prior weights `w_pop, w_road, w_out, w_exp, w_0` tuned offline | **Survivor likelihood**: keep one intensity grid per behaviour component, and learn the mix and the head count *online* from finds and misses (scenario posterior) | The truth's category weights change per scenario (Dirichlet), and N̂ is off by about ±30%. Offline tuning can only find the average mix. The posterior finds this flood's mix. |
| 4 | Forecast taken at face value | **Optional:** learn the forecast's scenario-wide stage bias by comparing past forecasts with the observed extent every 10 min | The forecast has a deliberate bias of −0.3 to +0.3 m. Correcting it sharpens *when* water arrives, which drives the value. |

**Reused and credited in the pitch:** Bayesian search theory (Stone), PHD/Poisson intensity belief (Mahler; Dames 2020), Johnson-criteria detection, Coast Guard-style weighted scenarios (SAROPS), the Environment Agency hazard rating (FD2320). **Ours:** coupling a learning flood forecast and a hazard-based survival model to the search objective, plus learning the population's behaviour mix during the search.

---

## 2. Small environment additions (all agent-visible; add in planner Phase 0)

None of these exposes truth. Add the information-boundary test cases for each.

| Addition | Where | Why |
|---|---|---|
| `Controller.observe(views_all)` hook, called every sensor tick (2 s) for **all** drones | `sim.py` | `decide(views_due)` only sees drones that are due. The fleet belief must take every frame from every drone. |
| `PlannerView.n_hat` (true count × lognormal σ = 0.3, drawn once per scenario) | `sim.py` | The appendix defines N̂, but the view doesn't carry it |
| `PlannerView.episode_end_s` | `sim.py` | The value function needs the scoring horizon (episode end + 2 h) |
| `building_density` and `road_mask` in static maps (not just the combined `population`) | `sim.py` | Separate behaviour components |
| `FloodForecast.observed_extent(t_now)` → bool grid (50 m) | `forecast.py` | The planner already "knows" this per section 6.2, but the API has no accessor |
| `stage_offset=0.0` keyword on `prob_flooded`, `expected_depth`, `expected_hazard` (adds the offset to Ŝ before the formulas) | `forecast.py` | Needed only for bias learning (section 7). Skip if that phase is cut. |
| `FloodForecast.sigma(lead_s)` | `forecast.py` | Same |

---

## 3. Package layout

```
floodsim/planner/
  __init__.py
  params.py        # PlannerParams, SwarmParams (all defaults from section 12)
  detection.py     # state-aware p̂_d (section 4)
  belief.py        # ComponentBelief: component grids, looks, finds, prediction (section 5)
  scenarios.py     # scenario grid, prior, posterior helpers (section 6)
  floodbelief.py   # forecast bias posterior (section 7, optional)
  urgency.py       # danger rate, cumulative danger, marginal survival tables (section 8)
  scoring.py       # block tables, travel/search time, energy, height choice (section 9)
  fleet.py         # FleetBrain: ingest frames, claims, assignment, verify policy, debug log (section 10)
  controllers.py   # BayesGreedyPlanner, HeuristicFloodPlanner, SurvivalPlanner (section 11)
  experiments.py   # compare, ablation, sweeps, paired stats (section 13)
  tune.py          # equal-budget tuning for each planner (section 13)
tests/planner/     # section 14
```

All three Bayesian planners share `FleetBrain`, the belief, detection, scoring and claims. **They differ only in the per-cell value function.** That keeps the comparison fair and the code small.

---

## 4. Detection model the planner uses (`detection.py`)

The planner uses the same range curve as the environment (section 11):

```latex
r = \frac{d_{target}}{0.5\ \mathrm{m}} \cdot \frac{R}{z_{cam}}, \qquad p(z) = 0.95\,\frac{r^{E}}{1 + r^{E}}, \qquad E = 2.7 + 0.7\,r
```

It doesn't know each person's state, so it averages over a **state mix per behaviour component**. The mix is driven by the forecast at the time of the look:
- p_deep = P(depth ≥ 0.5 m)
- p_wet = P(depth > 0.1 m)

| Component | States and weights | Size, state factor |
|---|---|---|
| `home`, `home_low` | home_inside (1 − p_deep), roof (p_deep) | 0.5 m, 0.02 / 0.5 m, 1.0 |
| `road` | vehicle (1) | 0.5 m, 0.2 |
| `open`, `high_ground`, `anywhere` | outdoor (1 − p_deep), in_water (p_deep) | 0.5 m, 1.0 / 0.3 m, 0.8 |

```latex
\hat{p}_d^{(k)}(c, z) = m_{cover}(c) \sum_{s} \pi^{(k)}_s(c)\; p\big(z \mid d_s\big)\; f_s
```

- **p_deep from the forecast:** p_deep = P_flood × sigmoid((E[d | wet] − 0.5)/0.15), with E[d | wet] = expected_depth / max(P_flood, 0.05). Observed-wet cells use P_flood = max(P_flood, 0.95).
- m_cover comes from land cover (urban 0.6, forest 0.2, else 1.0). Land cover is a static map, so the planner may know it.
- **Sanity check (verified):** at 84 m over a town, p̂_d is 0.010 dry and 0.50 flooded. This single fact makes timing matter: a pass over a dry town finds almost nobody indoors.

---

## 5. Belief (`belief.py`)

### 5.1 Representation
- **Components** k ∈ {home, home_low, road, open, high_ground, anywhere} (K = 6). Each is a normalised map built only from planner-visible layers:
  - `home` ∝ building_density
  - `home_low` ∝ building_density × [HAND < 5 m]. Flood-exposed homes are a reasonable public-map prior; note it in `ASSUMPTIONS.md`.
  - `road` ∝ road_mask × (1 + 2[HAND < 3 m])
  - `open` ∝ open land within 2 km of the river
  - `high_ground` ∝ cells with HAND ≥ 3 m within 1 km of the expected flood edge at t₀ + 1 h (where walkers end up)
  - `anywhere` ∝ all land
- `lam[k]` (float32 `[K, 200, 200]`) = expected number of **unfound** people in each cell if everyone followed component k, each component starting at total N̂.
- **The fleet keeps one belief** (the "fleet brain", section 10). The default is `comms_mode="base_relay"`: a drone's frames enter the belief only while it is connected to base through the radio graph (signal radius between drones and base). Otherwise they wait in a buffer and are applied when it reconnects. Fallback: `"perfect"`.

### 5.2 Updates
- **Look (every observed cell in a frame):**
  - A cell gets at most one update per `min_revisit_s` = 20 s; within that window keep the frame with the highest p̂_d. Consecutive frames overlap almost entirely.
  - Before updating, add `Σ p̂_d^(k) λ_k(c)` to the per-component accumulator `A[k]` (section 6).
  - Then λ_k(c) ← (1 − p̂_d^(k)(c)) λ_k(c).
  - This is exact Poisson thinning: the undetected people are still a Poisson field, with or without detections or false alarms in that look.
- **Detection in the frame:**
  - Store a candidate {cell, time, drone, `lam[:, c]` before the look, `pd[:, c]`}.
  - Candidate probability: P_real = p̂_d λ̄ / (κ + p̂_d λ̄), where λ̄ is the mixed intensity (section 6) and κ = λ_fa per cell per frame (0.002).
- **Verification confirmed:** call `confirmed_find(lam_before, pd)`, which adds the log-likelihood term for the scenario posterior (section 6). The person is already out of `lam` (removed by the look), so **don't subtract 1**. The appendix's "λ⁺ − 1" double-counts under this representation.
- **Verification rejected:** treat the orbit as a look with p̂_d = 0.95 for every component: λ ← 0.05 λ, matching the appendix.

### 5.3 Prediction (every 60 s; cuttable to diffusion only)
- Only the `open` component moves, since homes, roofs and vehicles stay put in the environment:
  - mobile fraction m (default 0.3) drifts uphill along ∇HAND at walking speed where P_flood > 0.5 (semi-Lagrangian, `map_coordinates(order=1)`);
  - Gaussian diffusion with σ = 0.5 × 0.9 m/s × 60 s / 50 m;
  - in cells with expected depth > 0.6 m and observed speed > 1 m/s, 0.5 per minute of the mass is carried along `observed_velocity` × 0.5.
- Mass leaving the domain goes to `lost_mass`.
- Each component is a separate grid, so prediction is K-independent. Only one component moves, so it is cheap.

---

## 6. Survivor likelihood: scenario posterior (`scenarios.py`)

A **scenario** j is a mixture weight vector w_j over the K components plus a head-count scale s_j:

```latex
\lambda_j(c) = s_j \sum_k w_{jk}\, \lambda_k(c)
```

- **Scenario set:** all weight vectors on a simplex grid with step 0.25 (126 vectors for K = 6) × scales {0.5, 0.7, 1.0, 1.4, 2.0} = 630 scenarios.
- **Prior:** Dirichlet with mean (home, home_low, road, open, high_ground, anywhere) = (0.30, 0.20, 0.20, 0.15, 0.10, 0.05) and concentration 6, × lognormal on s with σ = 0.3 (matching N̂'s noise).

**Why this is cheap.** Every update is linear per cell and identical across components. So the Poisson log-likelihood of scenario j needs only the per-component accumulators, never a grid per scenario:

```latex
\log p(j \mid \text{data}) = \log \pi_j \;-\; s_j \sum_k w_{jk} A_k \;+\; \sum_{\text{confirmed finds } i} \log\!\Big(s_j \sum_k w_{jk}\, \hat{p}^{(k)}_{d,i}\, \lambda_k(c_i)\big|_{\text{before look}}\Big) \;+\; \text{const}
```

- **What the planner plans with:** the mixed intensity λ̄(c) = Σ_k ŵ_k λ_k(c), where ŵ_k = Σ_j P(j | data) s_j w_jk.
- **Verified on synthetic data** (Appendix B):
  - truth: 50% of people on high ground and 1.4× the expected count;
  - prior: 10% high ground;
  - after looking at 60% of cells and 11 finds, the posterior weighted high ground at 0.35 and rated the count at 1.18×. Roads were correctly ruled out.
- **Optional uncertainty bonus** (default off; in the ablation only):
  - add β_scen × H(post) × Σ_j P(j) |λ_j(c) − λ̄(c)| to each cell's value;
  - H is the posterior entropy normalised to [0, 1]. Compute the mean absolute deviation from 8 posterior samples, not all 630.
  - In our earlier simulator this bonus did not add lives, which is why it is off.

---

## 7. Learning the forecast's bias (`floodbelief.py`, optional phase)

The stage forecast has a scenario-wide bias b ∈ [−0.3, 0.3] m. Learn it from the observed extent:
- Keep a posterior over b ∈ {−0.5, −0.4, …, +0.5} m. Prior: normal, mean 0, σ = 0.2.
- Every 10 min, when a new observed extent arrives at time t, take each stored forecast issued at t − L (L = 15, 30, 60 min) for time t:

```latex
P_b(c) = \Phi\!\left(\frac{\hat{S}(\mathrm{drain}(c), t) + b - \mathrm{HAND}(c)}{\sigma(L)}\right), \qquad P^{obs}_b = 0.95\,P_b + 0.05\,(1 - P_b)
```

```latex
\Delta \log p(b) = \tau \sum_{c \in \mathcal{S}} \big[o_c \log P^{obs}_b(c) + (1 - o_c)\log(1 - P^{obs}_b(c))\big]
```

- 𝒮 = up to 2,000 random cells with 0.05 < P₀ < 0.95 (where the forecast is uncertain).
- τ = 0.1 tempers the evidence: cells near each other share errors (the AR(1) profile), so they aren't independent.
- Use the posterior mean b̂ as `stage_offset` in every forecast call the planner makes.
- Log b̂ for the visualisation ("fleet believes the forecast is 0.2 m low").
- **Fallback:** b̂ = 0.

---

## 8. Urgency: marginal survival (`urgency.py`)

The environment's survival model (section 10):
- **Survival:** P_surv = exp(−T_danger / T_crit), with T_crit = 40 min.
- **Danger time** accrues while HR > 0.75 for exposed people (outdoors, in the water), and while HR > 2 for sheltered people (home, roof, vehicle).
- **Finding someone:** survival freezes at confirmation.
- **Not finding someone:** danger accrues until episode end + 2 h.

The planner mirrors this with its own copy T̂_crit (tunable, default 40 min). Two **exposure classes** are enough:
- **sheltered** (home, home_low, road): danger threshold HR > 2;
- **exposed** (open, high_ground, anywhere): HR > 0.75, mixed with the outdoor/in-water states.

**Danger rate per class** at forecast lead ℓ (leads 0–60 min in 5 min steps, from the forecast cache):

```latex
P(HR > \theta) \approx P_{flood}\;\Phi\!\left(\frac{\mathbb{E}[HR \mid wet] - \theta}{\sigma_{HR}}\right), \qquad \mathbb{E}[HR \mid wet] = \frac{\mathbb{E}[HR]}{\max(P_{flood}, 0.05)}, \quad \sigma_{HR} = 0.5
```

**Cumulative danger and value**, with D_past(c) = expected danger already accumulated (integrated by the planner from lead 0 at every 5 min refresh since episode start):

```latex
D(c, \ell) = \int_{t}^{t+\ell} r(c, \tau)\, d\tau, \qquad D_{end}(c) = D(c, 60\,\mathrm{min}) + \rho \cdot r(c, 60\,\mathrm{min}) \cdot (T_{end} + 2\,\mathrm{h} - t - 60\,\mathrm{min})
```

```latex
V(c, \ell) = \exp\!\left(-\frac{D_{past} + D(c,\ell)}{\hat{T}_{crit}}\right) - \exp\!\left(-\frac{D_{past} + D_{end}(c)}{\hat{T}_{crit}}\right) + \varepsilon
```

- ρ (tail factor, default 1.0) holds the 60-min hazard beyond the forecast horizon.
- ε = 0.02 is a small value for finding anyone, which helps the secondary "found fraction" metric.
- **Verified behaviour** (Appendix B):

| Cell | Value at 0 / 15 / 30 / 60 min |
|---|---|
| Dry all episode | 0, 0, 0, 0 |
| Dangerous from 15 min | 1.00, 0.94, 0.64, 0.30 (go now) |
| Dangerous from 50 min | 1.00, 1.00, 1.00, 0.73 (no rush yet) |
| Dangerous for the last 3 h | 0.01 (beyond help) |

- **Recompute** the per-class value tables `V[class, lead, cell]` (2 × 13 × 200 × 200, float32) once per 5-min forecast bucket. That is cheap.

**What this buys over the bonus multiplier:**
- dry towns get almost no pull;
- the cliff at one lead time disappears;
- shallow, slow water doesn't count;
- people already beyond help aren't chased;
- combined with section 4, the planner learns to **arrive as the water arrives**: before that, people are invisible indoors; after it, their danger time grows.

---

## 9. Choosing a goal (`scoring.py`)

**Blocks:** 500 m macro-cells (20 × 20 = 400). **Heights:** h ∈ {60 m, 0.6R, 0.8R, R}, clamped to the search band (60 m to R = camera_range_m).

**Block value table**, refreshed every 60 s (and after any confirmed find), at 100 m resolution (2 × 2 sums of the 50 m grids) to keep it fast:

```latex
B[h, \ell, M] = \sum_{c \in M} \sum_k \hat{w}_k\, \lambda_k(c)\; \hat{p}^{(k)}_d(c, h)\; V_{class(k)}(c, \ell)
```

- Bayesian greedy: replace V by 1.
- Heuristic baseline: replace V by U(c, t + L).

**Per drone i and candidate block M:**
- **Travel time:** T_travel = distance / max(2 m/s, cruise + along-track wind). Cruise = 15 m/s + ½ headwind (capped at 25), matching the autopilot. Wind is the fleet's mean-wind estimate: the average of every drone's estimate, so its error shrinks with √n.
- **Search time:** T_search(h) = 500² / (2h tan 30° (1 − overlap) × V_search) + 20 s per leg end. V_search = max(13 m/s, wind at h + 4 m/s), capped at 25 m/s, the same rule as `ScriptedSweep`, so crosswind legs hold their track.
- **Arrival lead:** ℓ = T_travel + ½ T_search(h), interpolated in the 13 lead steps and clipped at 60 min.
- **Energy:** E_cost = level power × (T_travel + T_search) from the glider performance table (GliderParams are team inputs, so planner-visible).
- **Feasibility:** E_cost + E_home(M) × 1.3 + reserve ≤ E_now, where E_home uses the autopilot's return formula from the block centre. Infeasible blocks are dropped. The autopilot's RTB rule still fires on its own; the planner just doesn't send a drone somewhere it can't come back from.

```latex
S_i(M, h) = \frac{B[h, \ell_i(M,h), M] \cdot (1 - \mathrm{claimed}(M))}{\big(T_{travel} + T_{search}(h)\big)^{\beta}} - \mu\,\frac{E_{cost}}{E_{now}} - \mathrm{spread}_i(M)
```

- **Goal:** `Search(box = M, agl = h, spacing = 2h tan 30° (1 − overlap))` for the best (M, h). Keep the current block unless S_new > (1 + hysteresis) × S_current.
- **Height falls out of the value:** flooded blocks favour low passes (people in the water are 0.3 m targets), dry high ground favours high, wide passes. Log the chosen height in the decision string.
- **Soaring:** `allow_soaring = battery < E_soar and the best block's score is below the median of the last 10 decisions` (no urgent work nearby).
- **Verification policy:** send `Verify(cell)` when P_real ≥ ν (default 0.3). Only the detecting drone verifies; its search block stays claimed. Optionally lower ν where V is high: ν_eff = ν × (1 − 0.5 V / V_max).

---

## 10. Fleet brain and swarm coordination (`fleet.py`)

- **`FleetBrain`** is the single object behind every planner controller:
  - `observe(views_all)` ingests frames, detections, verifications and wind estimates, subject to `comms_mode`;
  - `decide(views_due)` assigns goals.
- **Claims:** assignment is sequential greedy over due drones (lowest battery first). A drone's chosen block, and the blocks its transit path crosses at search height, are claimed until it leaves them or `claim_expiry_s`. Claimed blocks get `claimed = 1` for others.
- **Spread:** spread_i(M) = spread_penalty × (number of teammates' targets within spread_radius of M) × (median block score).
- **Lost or landed drones:** their claims are released at once. After a battery swap, the drone gets a fresh decision on relaunch.
- **Decision log** (feeds the visualisation): every goal is logged as a dict with time, drone, goal type, block, height, expected lives at stake, travel minutes and lead to danger, plus a one-line reason. Examples:
  - "G3 → block (12,7) at 70 m: 0.42 expected lives at stake, water dangerous in ~18 min, 4 min away"
  - "G2 verify (113,56): P_real 0.61"
- **`debug_snapshot()`** every 30 s of sim time, only when `record_grids=True`. It returns float16 grids at 100 m:
  - λ̄ (where people may still be)
  - Σ_k ŵ_k λ_k V (lives at stake now)
  - P_flood at +20 min with the bias correction
  - plus ŵ (component weights), the scenario count-scale posterior, the b posterior, the fleet wind estimate and active claims.

---

## 11. Controllers compared

| Name (`--controller`) | Value per cell | Notes |
|---|---|---|
| `scripted` | n/a | The environment's `ScriptedSweep` (lawnmower over HAND < 10 m) |
| `bayes` | 1 (expected finds) | Same belief, detection, scenarios, scoring, claims as ours |
| `heuristic` | U = 1 + λ_f P_flood(t+L) + λ_h E[HR](t+L)/2 | The environment appendix's planner, tuned with equal budget |
| `survival` (ours) | V (section 8) | Full plan |

**Ablations** (flags on `survival`):
- `--no-state-pd`: constant m_state 0.8, as in the appendix;
- `--no-scenarios`: fixed prior weights, no posterior;
- `--no-bias`;
- `--scen-bonus`: β_scen = 1;
- `--no-prediction`;
- `--comms perfect`.

---

## 12. Parameters (`PlannerParams`, `SwarmParams`)

| Parameter | Default | Tune range | Used by |
|---|---|---|---|
| T̂_crit | 40 min | 20–80 min | survival |
| σ_HR | 0.5 | 0.2–1.0 | survival |
| tail factor ρ | 1.0 | 0.3–1.5 | survival |
| ε welfare | 0.02 | 0–0.1 | survival |
| λ_f, λ_h, L | 2, 1, 20 min | 0–10, 0–5, 0–60 min | heuristic |
| β travel discount | 1.0 | 0.5–2 | all Bayesian |
| μ energy penalty | 0.5 | 0–2 | all Bayesian |
| overlap | 0.15 | 0–0.4 | all |
| hysteresis | 0.15 | 0–0.5 | all Bayesian |
| replan_s | 60 s | 30–180 s | all Bayesian |
| ν verify threshold | 0.3 | 0.1–0.9 | all Bayesian |
| mobile fraction m | 0.3 | 0–0.6 | all Bayesian |
| scenario prior concentration | 6 | 2–20 | all Bayesian |
| E_soar, w_enter | 0.6, 0.8 m/s | as environment | all |
| claim_expiry_s, spread_radius, spread_penalty | 300 s, 1.5 km, 0.5 | as appendix | all Bayesian |
| comms_mode | base_relay | perfect, base_relay | all Bayesian |

---

## 13. Experiments, tuning and outputs (`experiments.py`, `tune.py`)

**Seeds:** train 0–59 for tuning, validation 10000–10049, **test 20000–20099 (run once at the end)**. Use common random numbers: every controller sees the same worlds.

**Equal tuning budget.** Each of `bayes`, `heuristic` and `survival` gets the same number of trials:
- Optuna TPE if installed, else Sobol;
- 30 trials × 30 train seeds;
- best 3 re-scored on validation.

Rough laptop cost: about 7 s per 8-drone episode (environment + planner) × 2,700 per planner / 8 cores ≈ 40 min per planner. **If time is short, skip tuning and use defaults for every planner**, and say so on the slide.

**Runs** (each writes `episodes.csv`: one row per seed with every metric from environment section 15, plus planner diagnostics):
1. `compare`: 4 controllers × 100 test seeds. **This is the headline.**
2. `ablation`: `survival` and each ablation flag × 100 test seeds.
3. `lambda_sweep`: `heuristic` with λ_f ∈ {0, 0.5, 1, 2, 4, 8} × 50 seeds, with ours as a reference line.
4. `swarm_size`: `bayes` vs `survival` with n ∈ {2, 4, 6, 8} × 50 seeds each.
5. **Demo episodes:** `survival` and `bayes` on the chosen demo seed with `record_grids=True`, saving `debug_snapshot`s and the decision log (consumed by `VIZ_PLAN.md`).

**Statistics** (`summary.json`):
- per controller: mean ± SE of `survival_at_risk`, found fraction, time to first find, at-risk people found, crashes, forced landings;
- **paired differences by seed** (ours − bayes, ours − heuristic) with SE and win/loss counts;
- the same split by flood type and by swarm size;
- mean "survival secured over time" curves (cumulative frozen survival of found at-risk people, every 5 min).

**Demo seed rule:** pick the test seed whose ours-minus-bayes difference is closest to the **median** difference, not the best one. Say this in the pitch.

---

## 14. Tests (`tests/planner/`)

| Test | Pass criterion |
|---|---|
| Detection curve | p(z) matches the environment's section 11 table within 0.02; dry-town p̂_d < 0.02, flooded-town p̂_d > 0.4 at 0.6R |
| Look update | Σλ falls by exactly Σ p̂_d λ over observed cells; no cell negative |
| Revisit window | Two frames within 20 s update a cell once (highest p̂_d kept) |
| Scenario recovery | Synthetic test from Appendix B: posterior mean weight on the true dominant component > 0.3; count-scale posterior mean within [1.0, 2.0] for truth 1.4 |
| Scenario prior only | With no data, the posterior equals the prior (to 1e-9) |
| Value shapes | The four synthetic cells in section 8 give the listed behaviour |
| Value aligns with scoring | On 5 seeds, the planner's expected survival secured by confirmed finds correlates (r > 0.5) with the truth-side survival frozen at those finds |
| Bias learning | With a synthetic forecast biased by +0.3 m, the b posterior mean is within 0.1 m after 60 min |
| Prediction | Mass conserved up to `lost_mass`; zero motion where P_flood < 0.5 |
| Equivalence | `survival` with V ≡ 1 picks the same goals as `bayes` on 3 seeds |
| Claims | No two drones search the same block at once in 20 runs with 8 drones |
| Feasibility | A drone is never assigned a block it cannot return from under the autopilot's formula |
| Information boundary | Still passes with the new view fields |
| Performance | Planner adds < 2 s to a cached 8-drone fast episode |
| Determinism | Same seed and controller → identical `EpisodeResult` |

---

## 15. Build phases (about 4 h of Claude Code time; core about 3 h)

| Phase | Budget | Build | Done when | Fallback |
|---|---|---|---|---|
| P0. Hooks | 15 min | Section 2 additions, `planner/` skeleton, params | Information-boundary test passes with new fields | none |
| P1. Detection | 15 min | Section 4 | Detection tests pass | none |
| P2. Belief | 40 min | Section 5.1–5.2, fleet brain ingest (`perfect` comms first) | Look/revisit tests pass; λ̄ map plot looks sane on seed 3 | none |
| P3. Scenarios | 25 min | Section 6 (copy Appendix B) | Scenario tests pass | Fixed prior weights |
| P4. Urgency | 35 min | Section 8 | Value tests pass; lives-at-stake map plot for seed 3 lights up streets about to flood | Use P(HR>θ) at a single lead L = 20 min (worse but works) |
| P5. Scoring and controllers | 45 min | Sections 9–11 (claims, heights, verify, decision log) | `evaluate --controller survival --seeds 0-9` runs; equivalence and claims tests pass | Fixed height 0.6R; no spread term |
| P6. Compare | 20 min + compute | Section 13 runs 1, 2 (no tuning) | `results/compare_*/summary.json` exists | 50 seeds instead of 100 |
| P7. Bias learning | 30 min | Section 7 | Bias test passes | Skip (b̂ = 0) |
| P8. Prediction and base_relay comms | 30 min | Sections 5.3, 5.1 | Prediction test passes | Diffusion only; perfect comms |
| P9. Tuning and sweeps | 30 min + compute | Section 13 tuning, runs 3–4 | `lambda_sweep` and `swarm_size` outputs exist | Skip tuning; defaults for all |

**Cut order if the night runs out:** P9, then P8, then P7. **Never cut P6.** A comparison on 50 seeds with defaults beats a perfect planner with no results.

---

## 16. Overnight schedule (one laptop, one Claude Code session)

| Time | Work | Checkpoint |
|---|---|---|
| now–05:30 | Environment Phases 0–7 | `evaluate --controller scripted --seeds 0-9` runs |
| 05:30–08:30 | Planner P0–P6 (start `compare` before 08:30, let it run) | `summary.json` with 4 controllers |
| 08:30–10:00 | Visualisation (VIZ_PLAN V0–V3) while P7–P9 or more seeds run | Compare page opens on the demo seed |
| 10:00–11:00 | Charts (VIZ_PLAN V4), demo seed, slides | Every slide has a real number |
| 11:00–12:00 | Code freeze, rehearse twice, record a backup video | |

**If the environment slips past 06:30:** start the planner anyway against the environment's HAND-based flood fallback. The planner only touches the forecast API, so it doesn't care which truth model is underneath.

---

## Appendix A: decisions behind the defaults

- **Two exposure classes, not five states:** the scoring only distinguishes thresholds 0.75 and 2.0, so two classes capture everything the value needs.
- **Per-component grids rather than per-scenario grids:** 6 grids instead of 630. The scenario posterior is exact under the Poisson model because every update is linear and identical across components.
- **No "−1" on a confirmed find:** under Poisson thinning, the look that produced the detection already removed the detected person from the undetected intensity.
- **Fleet belief rather than per-drone beliefs with gossip:** halves the code and the runtime. `base_relay` keeps the radio limit honest (frames from out-of-range drones arrive late).

## Appendix B: reference code (verified on synthetic data)

Copy into `planner/detection.py`, `belief.py`, `scenarios.py` and `urgency.py`, then adapt names and shapes. The test that produced the numbers quoted above is included.

```python
# ---- detection.py (core) ----
STATE_SIZE = {"home_inside": 0.5, "roof": 0.5, "vehicle": 0.5, "outdoor": 0.5, "in_water": 0.3}
STATE_FACTOR = {"home_inside": 0.02, "roof": 1.0, "vehicle": 0.2, "outdoor": 1.0, "in_water": 0.8}

def pd_range(z_cam, size, R):
    r = (size / 0.5) * (R / np.maximum(z_cam, 1.0))
    E = 2.7 + 0.7 * r
    return 0.95 * r ** E / (1.0 + r ** E)

def state_mix(component, p_deep):
    if component in ("home", "home_low"):
        return {"home_inside": 1 - p_deep, "roof": p_deep}
    if component == "road":
        return {"vehicle": np.ones_like(p_deep)}
    return {"outdoor": 1 - p_deep, "in_water": p_deep}

def pd_hat(z_cam, R, cover_mod, component, p_deep):
    out = 0.0
    for s, w in state_mix(component, p_deep).items():
        out = out + w * pd_range(z_cam, STATE_SIZE[s], R) * STATE_FACTOR[s]
    return out * cover_mod
```

```python
# ---- belief.py + scenarios.py (core) ----
class ComponentBelief:
    def __init__(self, comps, n_hat, scen_w, scen_s, log_prior):
        self.lam = n_hat * comps / comps.sum(axis=(1, 2), keepdims=True)   # [K, ny, nx]
        self.W, self.s, self.log_prior = scen_w, scen_s, log_prior          # [J,K], [J], [J]
        self.A = np.zeros(self.lam.shape[0])                                # sum of pd*lam per component
        self.det_ll = np.zeros(len(scen_s))

    def post(self):
        lp = self.log_prior - self.s * (self.W @ self.A) + self.det_ll
        p = np.exp(lp - lp.max()); return p / p.sum()

    def w_eff(self):
        return (self.post() * self.s) @ self.W

    def intensity(self):
        return np.tensordot(self.w_eff(), self.lam, axes=1)

    def look(self, iy, ix, pd_k):              # pd_k: [K, n] for the n observed cells
        lam_c = self.lam[:, iy, ix]
        self.A += (pd_k * lam_c).sum(axis=1)
        self.lam[:, iy, ix] = lam_c * (1.0 - pd_k)

    def confirmed_find(self, lam_at_look_k, pd_k_cell):   # values stored at detection time
        lam_j = self.s * (self.W @ (pd_k_cell * lam_at_look_k))
        self.det_ll += np.log(np.maximum(lam_j, 1e-12))

    def rejected(self, iy, ix, pd_verify_k):              # verify orbit = a very good look
        self.look(np.atleast_1d(iy), np.atleast_1d(ix), pd_verify_k[:, None])

def scenario_grid(prior_mean, step=0.25, scales=(0.5, 0.7, 1.0, 1.4, 2.0), conc=6.0, scale_sigma=0.3):
    K = len(prior_mean); n = int(round(1 / step))
    ws = [np.array(c) / n for c in itertools.product(range(n + 1), repeat=K) if sum(c) == n]
    alpha = conc * np.asarray(prior_mean) + 1e-3
    W, S, LP = [], [], []
    for w in ws:
        for s in scales:
            lp = ((alpha - 1) * np.log(np.clip(w, 1e-3, None))).sum() - np.log(s) ** 2 / (2 * scale_sigma ** 2)
            W.append(w); S.append(s); LP.append(lp)
    LP = np.array(LP); return np.array(W), np.array(S), LP - LP.max()
```

```python
# ---- urgency.py (core) ----
def survival_value(rate_leads, lead_s, t_now, t_end_plus2h, d_past, t_crit=2400.0, tail_factor=1.0):
    """rate_leads [L, ...] danger rate (0-1) at leads lead_s (s, from 0). Returns V[L, ...], D[L, ...]."""
    dt = np.diff(lead_s).reshape((-1,) + (1,) * (rate_leads.ndim - 1))
    D = np.concatenate([np.zeros_like(rate_leads[:1]),
                        np.cumsum(0.5 * (rate_leads[1:] + rate_leads[:-1]) * dt, axis=0)])
    tail_s = max(t_end_plus2h - (t_now + lead_s[-1]), 0.0)
    D_end = D[-1] + tail_factor * rate_leads[-1] * tail_s
    return np.exp(-(d_past + D) / t_crit) - np.exp(-(d_past + D_end) / t_crit), D

def danger_rate(p_hr_low, p_hr_high, p_exposed):
    return p_exposed * p_hr_low + (1 - p_exposed) * p_hr_high
```

**Verified outputs** (synthetic test `test_ref.py`, shipped alongside this plan):

```
pd at 84 m over town: dry 0.010, flooded (people on roofs) 0.501
found 11  posterior mean weights [0.33 0.03 0.3 0.35 0.] scale 1.18   (truth [0.25 0 0.25 0.5 0], scale 1.4)
value at leads 0,15,30,60 min:  dry [0 0 0 0]; danger@15 [1.00 .94 .64 .30]; danger@50 [1.00 1.00 1.00 .73]; beyond help [.01 .01 .01 .00]
```
