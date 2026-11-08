import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from floodsim.config import SimConfig
from floodsim.planner.belief import ComponentBelief
from floodsim.planner.detection import pd_hat, pd_components
from floodsim.planner.floodbelief import BiasBelief
from floodsim.planner.params import PlannerParams
from floodsim.planner.scenarios import scenario_grid
from floodsim.planner.urgency import survival_value
from floodsim.sensor import pd_range

from conftest import get_world, skip_perf


def test_detection_curve():
    R = 140.0
    for z, exp in [(70, 0.90), (105, 0.70), (140, 0.48), (210, 0.21)]:
        assert abs(pd_range(z, 0.5, R) - exp) < 0.02
    dry = float(pd_hat(84, R, 0.6, "home", np.array(0.0)))
    wet = float(pd_hat(84, R, 0.6, "home", np.array(1.0)))
    assert dry < 0.02 and wet > 0.4
    v = pd_components(np.array([84.0, 84.0]), np.array([90.0, 90.0]), R, np.array([0.6, 0.6]), np.array([0.0, 1.0]))
    assert v.shape == (6, 2) and v[0, 0] < 0.02 and v[0, 1] > 0.4


def _belief(K=6, n=40, n_hat=20.0, seed=0, params=None):
    rng = np.random.default_rng(seed)
    comps = np.stack([gaussian_filter(rng.uniform(size=(n, n)), 2) + 1e-3 for _ in range(K)])
    comps /= comps.sum(axis=(1, 2), keepdims=True)
    return ComponentBelief(comps, n_hat, params or PlannerParams())


def test_look_update_exact():
    b = _belief()
    cells = np.arange(0, 400, 3)
    pd = np.random.default_rng(1).uniform(0, 0.9, (6, len(cells)))
    lam = b.lam.reshape(6, -1)
    before = lam.sum(axis=1).copy()
    expect = (pd * lam[:, cells]).sum(axis=1)
    b.look_flat(cells, pd)
    assert np.allclose(before - b.lam.reshape(6, -1).sum(axis=1), expect)
    assert np.allclose(b.A, expect)
    assert (b.lam >= 0).all()


def test_revisit_window():
    b = _belief()
    c = np.array([123])
    lam0 = b.lam_at(123)
    b.queue_look(c, np.full((6, 1), 0.3), 100.0)
    b.queue_look(c, np.full((6, 1), 0.6), 110.0)
    b.queue_look(c, np.full((6, 1), 0.2), 118.0)
    assert b.flush(115.0) == 0
    assert b.flush(121.0) == 1
    assert np.allclose(b.lam_at(123), lam0 * (1 - 0.6))
    assert b.flush(200.0) == 0


def _blobs(n, sigma, seed, N=60):
    r = np.random.default_rng(seed)
    g = np.zeros((N, N))
    for _ in range(n):
        g[r.integers(0, N), r.integers(0, N)] += r.uniform(0.5, 1.5)
    g = gaussian_filter(g, sigma) + 1e-6
    return g / g.sum()


def test_scenario_recovery():
    """Appendix B synthetic test: truth puts half the people on 'high ground' and 1.4x the count."""
    rng = np.random.default_rng(0)
    N = 60
    comps = np.stack([_blobs(6, 3, 1), _blobs(4, 1.5, 2), _blobs(10, 4, 3), _blobs(5, 2, 4), np.ones((N, N))])
    comps = comps / comps.sum(axis=(1, 2), keepdims=True)
    K = 5
    p = PlannerParams(prior_mean=(0.45, 0.2, 0.2, 0.1, 0.05))
    b = ComponentBelief(comps, 20.0, p)
    truth_w = np.array([0.25, 0.0, 0.25, 0.5, 0.0])
    lam_true = 20.0 * 1.4 * np.tensordot(truth_w, comps, axes=1)
    people = rng.poisson(lam_true)
    order = rng.permutation(N * N)[: int(0.6 * N * N)]
    for idx in order:
        lam_b = b.lam_at(int(idx))
        pdk = np.full((K, 1), 0.7)
        b.look_flat(np.array([idx]), pdk)
        iy, ix = divmod(int(idx), N)
        n_det = rng.binomial(people[iy, ix], 0.7)
        people[iy, ix] -= n_det
        for _ in range(n_det):
            b.confirmed_find(lam_b, pdk[:, 0])
    post = b.post()
    w_mean = post @ b.W
    s_mean = post @ b.S
    assert w_mean[3] > 0.3 and w_mean[1] < 0.15
    assert 1.0 < s_mean < 2.0


def test_scenario_prior_only():
    b = _belief()
    W, S, LP = scenario_grid(b.p.prior_mean, conc=b.p.scenario_concentration)
    prior = np.exp(LP - LP.max())
    prior /= prior.sum()
    assert np.allclose(b.post(), prior, atol=1e-9)
    assert len(S) == 630


def test_value_shapes():
    lead = np.arange(0, 3601, 300.0)
    L = len(lead)
    t_now, t_end2 = 3600.0, 3600.0 + 2.5 * 3600 + 7200
    rate = np.zeros((L, 4))
    rate[lead >= 900, 1] = 1.0
    rate[lead >= 3000, 2] = 1.0
    rate[:, 3] = 1.0
    d_past = np.array([0.0, 0.0, 0.0, 3 * 3600.0])
    val, D = survival_value(rate, lead, t_now, t_end2, d_past)
    assert np.allclose(val[:, 0], 0)
    assert val[0, 1] > 0.9 and val[0, 1] > val[6, 1]
    assert val[[0, 3, 6, 12], 1] == pytest.approx([1.00, 0.94, 0.64, 0.30], abs=0.02)
    assert val[0, 2] > 0.9 and abs(val[0, 2] - val[6, 2]) < 1e-6 and val[12, 2] == pytest.approx(0.73, abs=0.02)
    assert val[0, 3] < 0.02


def test_prediction_mass_and_no_motion(cfg):
    w = get_world(1, cfg)
    from floodsim.sim import StaticMaps
    maps = StaticMaps.from_terrain(w.terrain, cfg.world.domain_m)
    from floodsim.planner.belief import component_maps
    b = ComponentBelief(component_maps(maps, w.forecast, 0.0), 15.0, PlannerParams())
    for t in np.arange(60.0, 3601.0, 60.0):
        before = b.lam[3].sum()
        lost_before = b.lost_mass[3]
        mob0 = b.mob.copy()
        b.predict(60.0, w.forecast, t, maps)
        assert b.lam[3].sum() + (b.lost_mass[3] - lost_before) == pytest.approx(before, rel=1e-9, abs=1e-9)
        dry = w.forecast.prob_flooded(t, 0.0) <= 0.5
        # mass sitting in dry cells only changes by what arrives from flooded cells (it never leaves)
        assert np.all(b.mob[dry] >= mob0[dry] - 1e-12)


def test_bias_learning(cfg):
    from floodsim.scenario import build_world, sample_spec
    found = None
    for seed in (2, 3, 4, 5, 11):
        spec = sample_spec(seed, cfg=cfg, fc_bias_m=0.3, fc_sigma0_m=0.15, fc_growth_mph=0.2)
        w = build_world(spec, cache_dir="cache/", cfg=cfg)
        bb = BiasBelief(PlannerParams())
        for t in np.arange(600.0, 3601.0, 600.0):
            bb.update(w.forecast, t)
        if bb.history:
            found = bb.mean()
            assert abs(found - (-0.3)) <= 0.1, (seed, found)
            break
    assert found is not None


def _run(seed, ctrl, n=8, steps=None, cfg=None, **kw):
    from floodsim.runner import get_controller, make_config
    from floodsim.sim import Sim
    cfg = cfg or make_config()
    w = get_world(seed, cfg)
    c = get_controller(ctrl, cfg, **kw)
    sim = Sim(w, c, SimConfig(fast=True, max_steps=steps), n_drones=n)
    return sim, c, sim.run()


def test_equivalence_value_one(cfg):
    from floodsim.planner.controllers import SurvivalPlanner
    from floodsim.runner import make_config
    from floodsim.sim import Sim
    for seed in (1, 2, 3):
        _, cb, rb = _run(seed, "bayes", n=4, steps=2400)
        w = get_world(seed, cfg)
        cs = SurvivalPlanner()
        cs.brain.mode = "custom"
        cs.brain.custom_value = lambda brain, t: (np.ones((2, 1) + brain.cover100.shape, np.float32), np.array([0.0]))
        rs = Sim(w, cs, SimConfig(fast=True, max_steps=2400), n_drones=4).run()
        gb = [(d["drone"], d["goal"], d["block"], d["h"]) for d in rb.decision_log]
        gs = [(d["drone"], d["goal"], d["block"], d["h"]) for d in rs.decision_log]
        assert gb == gs


def test_claims_unique_and_feasible(cfg):
    from floodsim.planner import scoring
    from floodsim.planner.fleet import FleetBrain
    orig = FleetBrain._decide_one
    problems = []

    def checked(self, i, v, t):
        g = orig(self, i, v, t)
        blocks = [c["block"] for c in self.current.values() if c["kind"] == "search"]
        if len(blocks) != len(set(blocks)):
            problems.append(("dup", t, blocks))
        if g is not None and type(g).__name__ == "Search":
            wind10 = self.fleet_wind10()
            f150 = float(scoring.log_factor(150.0))
            wt = (wind10[0] * f150, wind10[1] * f150)
            cx, cy = 0.5 * (g.box[0] + g.box[2]), 0.5 * (g.box[1] + g.box[3])
            d, T_tr, cruise = scoring.travel(np.array(cx), np.array(cy), v.own.x, v.own.y, wt, self.ap)
            T_se, _ = scoring.search_time(self.p, g.agl, g.speed, self.p.block_m)
            E_cost = (scoring.level_power(self.gp, cruise) * T_tr + scoring.level_power(self.gp, g.speed) * T_se) / 3600
            E_home = scoring.e_return(self.gp, self.ap, self.base[0], self.base[1], cx, cy, wt)
            if E_cost + E_home > v.own.E + 1e-6:
                problems.append(("infeasible", t, float(E_cost + E_home), v.own.E))
        return g

    FleetBrain._decide_one = checked
    try:
        for seed in range(20):
            _run(seed, "survival", n=8, steps=1800)
    finally:
        FleetBrain._decide_one = orig
    assert not problems, problems[:5]


def test_information_boundary_planner(cfg):
    from test_sim import _walk
    from floodsim.planner.fleet import FleetBrain
    seen_views = []
    orig = FleetBrain.observe

    def spy(self, views):
        if len(seen_views) < 20:
            seen_views.append(views)
        return orig(self, views)

    FleetBrain.observe = spy
    try:
        _run(2, "survival", n=3, steps=600)
    finally:
        FleetBrain.observe = orig
    for vs in seen_views:
        for v in vs.values():
            _walk(v, set())


@pytest.mark.xfail(strict=False, reason="known limitation (ASSUMPTIONS.md): the HAND/extent forecast underestimates "
                   "depth in backwater and breach areas, so the planner's danger estimate lags the truth there")
def test_value_aligns_with_scoring(cfg):
    """Planner's expected survival secured at each confirmed find vs truth-side survival frozen at that find."""
    from floodsim.planner.fleet import FleetBrain
    est, truth = [], []
    for seed in range(5):
        records = {}
        orig = FleetBrain.decide

        def rec(self, views_due, _records=records, _orig=orig):
            for i, v in views_due.items():
                for (tv, cell, confirmed) in v.new_verifications:
                    if confirmed and self.ready and self.urg.d_past is not None:
                        iy, ix = divmod(int(cell), self.nn)
                        lam = self.belief.lam[:, iy, ix] * self.belief.w_eff()
                        frac_sh = lam[:3].sum() / max(lam.sum(), 1e-12)
                        dp = self.urg.d_past[:, iy // 2, ix // 2] + self.urg.rate0[:, iy // 2, ix // 2] * (v.t - self.urg.t_last)
                        tc = self.p.t_crit_hat_s
                        _records[(int(cell), round(tv))] = frac_sh * math.exp(-dp[0] / tc) + (1 - frac_sh) * math.exp(-dp[1] / tc)
            return _orig(self, views_due)

        FleetBrain.decide = rec
        try:
            sim, c, res = _run(seed, "survival", n=8)
        finally:
            FleetBrain.decide = orig
        for v in res.verification_log:
            key = (int(v["cell"]), round(v["t"]))
            if v["confirmed"] and v["true_person"] and key in records:
                est.append(records[key])
                truth.append(float(sim.people.surv_frozen[v["person"]]))
    est, truth = np.array(est), np.array(truth)
    assert len(est) >= 8
    if truth.std() < 1e-6 or est.std() < 1e-6:
        assert np.allclose(est, truth, atol=0.1)
    else:
        assert np.corrcoef(est, truth)[0, 1] > 0.5, (est.round(3), truth.round(3))


def test_determinism_planner(cfg):
    _, _, a = _run(3, "survival", n=4, steps=2400)
    _, _, b = _run(3, "survival", n=4, steps=2400)
    assert a.metrics == b.metrics and np.array_equal(a.trajectory, b.trajectory)
    assert [d["text"] for d in a.decision_log] == [d["text"] for d in b.decision_log]


@pytest.mark.skipif(skip_perf(), reason="FLOODSIM_SKIP_PERF=1")
def test_planner_overhead(cfg):
    import time
    _run(5, "scripted", n=8)
    t0 = time.perf_counter(); _run(5, "scripted", n=8); base = time.perf_counter() - t0
    t0 = time.perf_counter(); _run(5, "survival", n=8); plan = time.perf_counter() - t0
    assert plan - base < 3.0, (plan, base)
