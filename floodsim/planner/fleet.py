"""FleetBrain: one fleet belief behind every planner controller (PLANNER_PLAN section 10).

observe(views_all) ingests every drone's frames (subject to comms), decide(views_due) assigns goals.
The three Bayesian planners differ only in the per-cell value function."""
from __future__ import annotations

import math

import numpy as np

from ..goals import RTB, Search, Verify
from ..sensor import cover_modifier
from ..config import SensorParams
from . import scoring
from .belief import ComponentBelief, component_maps
from .detection import p_deep_from_forecast, pd_components
from .floodbelief import BiasBelief
from .params import COMPONENTS, PlannerParams, SwarmParams
from .urgency import UrgencyTables, coarsen, heuristic_U


class FleetBrain:
    def __init__(self, value_mode: str = "survival", params: PlannerParams | None = None,
                 swarm: SwarmParams | None = None, name: str = "survival"):
        assert value_mode in ("survival", "bayes", "heuristic", "custom")
        self.mode = value_mode
        self.p = params or PlannerParams()
        self.sw = swarm or SwarmParams()
        self.name = name
        self.ready = False
        self.decision_log = []
        self.custom_value = None

    # ------------------------------------------------------------------ set-up
    def _setup(self, v):
        p = self.p
        self.maps = v.maps
        self.fc = v.forecast
        self.R = float(v.camera_range_m)
        self.gp = v.glider
        self.ap = v.autopilot
        self.n = v.n_drones
        self.horizon = v.horizon_s
        self.base = v.base_xy
        n = self.maps.dem.shape[0]
        self.nn = n
        self.f = 2                                    # 50 m -> 100 m
        self.bf = int(round(p.block_m / (self.maps.cell * self.f)))   # 100 m cells per block side
        self.nb = n // self.f // self.bf
        self.dem = self.maps.dem.astype(np.float64)
        self.dem_flat = self.dem.reshape(-1)
        self.surf_flat = self.maps.surface.astype(np.float64).reshape(-1)
        self.cover = cover_modifier(self.maps.landcover, SensorParams())
        self.cover_flat = self.cover.reshape(-1)
        self.cover100 = coarsen(self.cover)
        self.obst100 = coarsen((self.maps.surface - self.maps.dem).astype(np.float32))
        comps = component_maps(self.maps, self.fc, v.t)
        self.belief = ComponentBelief(comps, float(v.n_hat), p)
        self.bias = BiasBelief(p, seed=0)
        self.urg = UrgencyTables(p, self.horizon)
        h = [60.0 if fr is None else fr * self.R for fr in p.heights_frac]
        self.heights = np.unique(np.clip(h, 60.0, max(self.R, 60.0)))
        bc = (np.arange(self.nb) + 0.5) * p.block_m
        self.bx, self.by = np.meshgrid(bc, bc)        # [nb, nb] row = north index
        self.claims = {}                             # block (iy, ix) -> (drone, expiry)
        self.current = {}                            # drone -> dict(kind, block, h, score)
        self.candidates = {i: [] for i in range(self.n)}
        self.buffers = {i: [] for i in range(self.n)}
        self.wind10 = {}
        self.last_predict = v.t
        self.table_t = -1e9
        self.table_dirty = True
        self.B = None
        self.score_hist = []
        self.pdeep_key = None
        self.ready = True
        self.n_frames = 0
        self.diag = {"verify_sent": 0, "confirmed": 0, "rejected": 0, "cand": 0}

    def _p_deep(self, t):
        key = int(t // 300)
        if key != self.pdeep_key:
            b = self.bias.mean()
            pf = self.fc.prob_flooded(t, 0.0, b)
            ed = self.fc.expected_depth(t, 0.0, b)
            self.pdeep = p_deep_from_forecast(pf, ed)
            self.pdeep_flat = self.pdeep.reshape(-1)
            self.pdeep100 = coarsen(self.pdeep)
            self.pdeep_key = key
        return self.pdeep

    # ------------------------------------------------------------------ observe
    def observe(self, views_all: dict) -> None:
        v0 = next(iter(views_all.values()))
        if not self.ready:
            self._setup(v0)
        t = v0.t
        self._p_deep(t)
        cells_all, zc_bld, zc_gnd, owners, frames = [], [], [], [], []
        for i, v in views_all.items():
            self.wind10[i] = v.wind_est
            new = [(fr, v.own.z) for fr in v.new_frames]
            if self.sw.comms_mode == "perfect" or v.connected_to_base:
                batch = self.buffers[i] + new
                self.buffers[i] = []
            else:
                self.buffers[i] += new
                batch = []
            for fr, z in batch:
                c = fr.cell_idx
                cells_all.append(c)
                zc_gnd.append(z - self.dem_flat[c])
                zc_bld.append(z - self.surf_flat[c])
                owners.append(np.full(len(c), len(frames)))
                frames.append((i, fr))
        if cells_all:
            cells = np.concatenate(cells_all)
            pd = pd_components(np.concatenate(zc_bld), np.concatenate(zc_gnd), self.R, self.cover_flat[cells],
                               self.pdeep_flat[cells], self.p.state_pd)
            own = np.concatenate(owners)
            for k, (i, fr) in enumerate(frames):
                self.n_frames += 1
                if not fr.detections:
                    continue
                sel = np.flatnonzero(own == k)
                fc = cells[sel]
                for det in fr.detections:
                    j = sel[np.flatnonzero(fc == det)[0]]
                    lam_b = self.belief.lam_at(det)
                    pdk = pd[:, j].astype(np.float64)
                    pr = self.belief.p_real(lam_b, pdk, self.p.kappa)
                    self.diag["cand"] += 1
                    self.candidates[i].append({"cell": int(det), "t": fr.t, "lam": lam_b, "pd": pdk, "p_real": pr})
            self.belief.queue_look(cells, pd, t)
        self.belief.flush(t)
        # slower processes
        if self.p.prediction and t - self.last_predict >= self.p.predict_every_s:
            self.belief.predict(t - self.last_predict, self.fc, t, self.maps)
            self.last_predict = t
            self.table_dirty = True
        if t > 0 and int(t) % int(self.fc.extent_dt) == 0:
            if self.bias.update(self.fc, t):
                self.pdeep_key = None

    # ------------------------------------------------------------------ tables
    def fleet_wind10(self):
        if not self.wind10:
            return (0.0, 0.0)
        w = np.array(list(self.wind10.values()))
        return tuple(w.mean(axis=0))

    def _value_tables(self, t):
        if self.mode == "survival":
            if self.urg.t_built is None or int(t // 300) != int(self.urg.t_built // 300):
                self.urg.update(self.fc, t, self.bias.mean())
            return self.urg.V, self.urg.leads
        if self.mode == "heuristic":
            U = heuristic_U(self.fc, t, self.p, self.bias.mean())
            return np.stack([U, U])[:, None], np.array([0.0])
        if self.mode == "custom" and self.custom_value is not None:
            return self.custom_value(self, t)
        ones = np.ones((2, 1) + self.cover100.shape, np.float32)
        return ones, np.array([0.0])

    def _refresh_tables(self, t, force=False):
        if not force and not self.table_dirty and t - self.table_t < self.p.table_every_s and self.B is not None:
            return
        self._p_deep(t)
        w = self.belief.w_eff()
        Wk = scoring.block_sum(w[:, None, None] * self.belief.lam, self.f).astype(np.float32)   # [K, 100, 100]
        if getattr(self, "_pdh_key", None) != self.pdeep_key:
            self._pd_h = np.stack([pd_components(np.full_like(self.obst100, h), h + self.obst100, self.R, self.cover100,
                                                 self.pdeep100, self.p.state_pd) for h in self.heights])
            self._pdh_key = self.pdeep_key
        pd_h = self._pd_h                                                          # [H, K, 100, 100]
        X = scoring.class_sums(Wk, pd_h)                                          # [2, H, 100, 100]
        V, leads = self._value_tables(t)
        self.V, self.leads = V, leads
        self.B = scoring.block_table(X, V, self.bf)                                # [H, L, nb, nb]
        if self.p.beta_scen > 0:
            self.B = self.B + self._scenario_bonus(pd_h)
        self.Wk = Wk
        self.table_t = t
        self.table_dirty = False

    def _scenario_bonus(self, pd_h):
        post = self.belief.post()
        H = np.sum(-post[post > 0] * np.log(post[post > 0])) / np.log(len(post))
        rng = np.random.default_rng(int(self.table_t) % 1000)
        js = rng.choice(len(post), size=8, p=post)
        lam100 = scoring.block_sum(self.belief.lam, self.f)
        lbar = np.tensordot(self.belief.w_eff(), lam100, axes=1)
        dev = np.mean([np.abs(np.tensordot(self.belief.S[j] * self.belief.W[j], lam100, axes=1) - lbar) for j in js], axis=0)
        bonus = self.p.beta_scen * H * scoring.block_sum(dev, self.bf)
        return bonus[None, None]

    # ------------------------------------------------------------------ decide
    def decide(self, views_due: dict) -> dict:
        v0 = next(iter(views_due.values()))
        if not self.ready:
            self._setup(v0)
        t = v0.t
        for i, v in views_due.items():
            self.wind10[i] = v.wind_est
            for (tv, cell, confirmed) in v.new_verifications:
                self._verification(i, cell, confirmed)
            if "relaunch" in v.reasons or not v.own.airborne:
                self.current.pop(i, None)
        self._release(t, views_due)
        self._refresh_tables(t)
        goals = {}
        order = sorted(views_due, key=lambda i: views_due[i].battery_frac)
        for i in order:
            v = views_due[i]
            if not v.own.airborne or v.own.mode in ("rtb", "ground", "down"):
                self._drop_claim(i)
                continue
            g = self._decide_one(i, v, t)
            if g is not None:
                goals[i] = g
        return goals

    def _verification(self, i, cell, confirmed):
        cands = self.candidates[i]
        match = [c for c in cands if c["cell"] == cell]
        c = match[0] if match else None
        if c is not None:
            cands.remove(c)
        if confirmed:
            self.diag["confirmed"] += 1
            if c is not None:
                self.belief.confirmed_find(c["lam"], c["pd"])
        else:
            self.diag["rejected"] += 1
            self.belief.rejected(int(cell), self.p.verify_pd)
        self.table_dirty = True
        self.current.pop(i, None)

    def _release(self, t, views):
        for b in [b for b, (d, exp) in self.claims.items() if exp < t]:
            del self.claims[b]
        for i, v in views.items():
            if not v.own.airborne or v.own.mode in ("rtb", "down", "ground"):
                self._drop_claim(i)

    def _drop_claim(self, i):
        for b in [b for b, (d, _) in self.claims.items() if d == i]:
            del self.claims[b]
        self.current.pop(i, None)

    def _nu_eff(self, cell):
        nu = self.p.nu_verify
        if self.p.nu_value_scaled and self.mode == "survival" and self.urg.V is not None:
            iy, ix = divmod(int(cell), self.nn)
            V = self.urg.V[:, 0, iy // self.f, ix // self.f].max()
            nu = nu * (1 - 0.5 * V / max(float(self.urg.V.max()), 1e-6))
        return nu

    def _decide_one(self, i, v, t):
        cur = self.current.get(i)
        # verification policy: only the detecting drone verifies; its search block stays claimed
        if cur is not None and cur["kind"] == "verify" and "verify_done" not in v.reasons:
            return None
        cands = [c for c in self.candidates[i] if c["p_real"] >= self._nu_eff(c["cell"]) and t - c["t"] < 600.0]
        self.candidates[i] = [c for c in self.candidates[i] if t - c["t"] < 600.0 and c in cands]
        if cands:
            c = max(cands, key=lambda c: c["p_real"])
            self.current[i] = {"kind": "verify", "block": cur["block"] if cur else None, "h": None, "score": None}
            self.diag["verify_sent"] += 1
            iy, ix = divmod(c["cell"], self.nn)
            self._log(t, i, "Verify", None, None, None, None, None,
                      f"G{i + 1} verify ({ix},{iy}): P_real {c['p_real']:.2f}")
            return Verify(cell=c["cell"])
        best = self._score_blocks(i, v, t)
        if best is None:
            self._drop_claim(i)
            self._log(t, i, "RTB", None, None, None, None, None, f"G{i + 1} → base: no block it can return from")
            return RTB()
        (biy, bix), h, S, stake, travel_min, lead_min = best
        self.score_hist = (self.score_hist + [S])[-10:]
        keep = cur is not None and cur["kind"] == "search" and cur["block"] == (biy, bix) and cur["h"] == h and \
            "search_done" not in v.reasons
        if cur is not None and cur["kind"] == "search" and "search_done" not in v.reasons and cur["block"] is not None:
            # hysteresis: keep the current block unless the new one is clearly better
            cs = self._score_of(i, v, t, cur["block"], cur["h"])
            if cs is not None and S <= (1 + self.p.hysteresis) * cs:
                keep = True
                biy, bix = cur["block"]
                h = cur["h"]
        self._claim(i, (biy, bix), t)
        if keep:
            return None
        self.current[i] = {"kind": "search", "block": (biy, bix), "h": h, "score": S}
        bm = self.p.block_m
        box = (bix * bm, biy * bm, (bix + 1) * bm, (biy + 1) * bm)
        wind = self.fleet_wind10()
        Vs = scoring.search_speed(self.p, wind, h)
        _, spacing = scoring.search_time(self.p, h, Vs, bm)
        soar = v.battery_frac < self.p.e_soar and len(self.score_hist) >= 3 and S < float(np.median(self.score_hist))
        lead_txt = f"water dangerous in ~{lead_min:.0f} min" if lead_min is not None else "no danger forecast"
        self._log(t, i, "Search", (int(bix), int(biy)), float(h), float(stake), float(travel_min), lead_min,
                  f"G{i + 1} → block ({bix},{biy}) at {h:.0f} m: {stake:.2f} expected lives at stake, "
                  f"{lead_txt}, {travel_min:.0f} min away")
        return Search(box=box, agl=float(h), spacing=float(spacing), speed=float(Vs), allow_soaring=bool(soar),
                      leg_angle_deg=90.0, tag=f"block{bix},{biy}")

    def _claim(self, i, block, t):
        for b in [b for b, (d, _) in self.claims.items() if d == i and b != block]:
            del self.claims[b]
        self.claims[block] = (i, t + self.sw.claim_expiry_s)

    # ------------------------------------------------------------------ scoring
    def _block_terms(self, i, v, t):
        p, gp, ap = self.p, self.gp, self.ap
        x, y = v.own.x, v.own.y
        wind10 = self.fleet_wind10()
        f150 = float(scoring.log_factor(150.0))
        wind_t = (wind10[0] * f150, wind10[1] * f150)
        dist, T_tr, cruise = scoring.travel(self.bx, self.by, x, y, wind_t, ap)
        return dist, T_tr, cruise, wind10, wind_t

    def _score_all(self, i, v, t):
        p, gp, ap = self.p, self.gp, self.ap
        dist, T_tr, cruise, wind10, wind_t = self._block_terms(i, v, t)
        E_now = max(v.own.E, 1e-6)
        E_home = scoring.e_return(gp, ap, self.base[0], self.base[1], self.bx, self.by, wind_t)
        claimed = np.zeros((self.nb, self.nb))
        for (biy, bix), (d, exp) in self.claims.items():
            if d != i and exp >= t:
                claimed[biy, bix] = 1.0
        spread = np.zeros((self.nb, self.nb))
        others = [c["block"] for d, c in self.current.items() if d != i and c.get("block") is not None]
        nL = self.B.shape[1]
        S_all = np.full((len(self.heights), self.nb, self.nb), -np.inf)
        stake_all = np.zeros_like(S_all)
        lidx_all = np.zeros(S_all.shape, dtype=int)
        Tsum_all = np.zeros_like(S_all)
        Ecost_all = np.zeros_like(S_all)
        med = None
        for hi, h in enumerate(self.heights):
            Vs = scoring.search_speed(p, wind10, h)
            T_se, _ = scoring.search_time(p, h, Vs, p.block_m)
            lead = np.clip(T_tr + 0.5 * T_se, 0, 3600.0)
            lidx = np.clip(np.round(lead / 300.0).astype(int), 0, nL - 1) if nL > 1 else np.zeros_like(lead, dtype=int)
            Bh = np.take_along_axis(self.B[hi], lidx[None], axis=0)[0]
            E_cost = (scoring.level_power(gp, cruise) * T_tr + float(scoring.level_power(gp, Vs)) * T_se) / 3600.0
            feas = E_cost + E_home <= E_now
            T = (T_tr + T_se) / 60.0          # minutes
            gain = Bh * (1 - claimed) / np.maximum(T, 0.1) ** p.beta_travel
            S_all[hi] = np.where(feas, gain, -np.inf)
            Ecost_all[hi] = E_cost / E_now
            stake_all[hi] = Bh
            lidx_all[hi] = lidx
            Tsum_all[hi] = T
        # energy penalty: mu x (E_cost / E_now) in units of the median value rate, so mu is dimensionless
        fin = S_all[np.isfinite(S_all) & (S_all > 0)]
        scale = float(np.median(fin)) if len(fin) else 0.0
        S_all = S_all - p.mu_energy * Ecost_all * scale
        if others and self.sw.spread_penalty > 0:
            fin = S_all[np.isfinite(S_all)]
            med = float(np.median(fin)) if len(fin) else 0.0
            for (oy, ox) in others:
                d = np.hypot(self.bx - self.bx[oy, ox], self.by - self.by[oy, ox])
                spread += (d <= self.sw.spread_radius)
            S_all = S_all - self.sw.spread_penalty * spread[None] * abs(med)
        return S_all, stake_all, lidx_all, T_tr

    def _score_blocks(self, i, v, t):
        S_all, stake_all, lidx_all, T_tr = self._score_all(i, v, t)
        self._last_scores = (i, S_all)
        if not np.isfinite(S_all).any():
            return None
        hi, biy, bix = np.unravel_index(int(np.argmax(S_all)), S_all.shape)
        lead_min = self._lead_to_danger(biy, bix)
        return ((int(biy), int(bix)), float(self.heights[hi]), float(S_all[hi, biy, bix]),
                float(stake_all[hi, biy, bix]), float(T_tr[biy, bix] / 60.0), lead_min)

    def _score_of(self, i, v, t, block, h):
        li, S_all = getattr(self, "_last_scores", (None, None))
        if li != i or S_all is None:
            S_all, *_ = self._score_all(i, v, t)
        hi = int(np.argmin(np.abs(self.heights - h)))
        s = S_all[hi, block[0], block[1]]
        return float(s) if np.isfinite(s) else None

    def _lead_to_danger(self, biy, bix):
        if self.mode != "survival" or self.urg.rate0 is None or not hasattr(self.urg, "last_rates"):
            return None
        sl = (slice(biy * self.bf, (biy + 1) * self.bf), slice(bix * self.bf, (bix + 1) * self.bf))
        r = self.urg.last_rates[1][:, sl[0], sl[1]].mean(axis=(1, 2))
        k = np.flatnonzero(r > 0.3)
        return float(self.urg.leads[k[0]] / 60.0) if len(k) else None

    # ------------------------------------------------------------------ logging / debug
    def _log(self, t, i, kind, block, h, stake, travel_min, lead_min, text):
        self.decision_log.append({"t": float(t), "drone": int(i), "goal": kind, "block": block, "h": h,
                                  "stake": stake, "travel_min": travel_min, "lead_min": lead_min, "text": text})

    def debug_snapshot(self, t):
        if not self.ready:
            return None
        lam = scoring.block_sum(self.belief.intensity(), self.f).astype(np.float16)
        # "lives at stake" always uses the survival value function, whatever planner is flying. It has its own
        # tables: rebuilding the planner's tables here at other times would change what the planner does.
        if not hasattr(self, "_urg_dbg"):
            self._urg_dbg = UrgencyTables(self.p, self.horizon)
        u = self._urg_dbg
        if u.t_built is None or int(t // 300) != int(u.t_built // 300):
            u.update(self.fc, t, self.bias.mean())
        V0 = u.V[:, 0]
        snap = {"t": float(t), "lam": lam,
                "w": self.belief.w_eff().round(4).tolist(), "count_scale": round(self.belief.count_scale(), 4),
                "bias_m": round(self.bias.mean(), 4), "wind": [float(a) for a in self.fleet_wind10()],
                "claims": [[int(b[0]), int(b[1]), int(d)] for b, (d, _) in self.claims.items()],
                "pflood20": coarsen(self.fc.prob_flooded(t, 1200.0, self.bias.mean())).astype(np.float16)}
        if V0 is not None:
            w = self.belief.w_eff()
            Wk = scoring.block_sum(w[:, None, None] * self.belief.lam, self.f)
            stake = Wk[:3].sum(0) * V0[0] + Wk[3:].sum(0) * V0[1]
            snap["stake"] = stake.astype(np.float16)
        return snap
