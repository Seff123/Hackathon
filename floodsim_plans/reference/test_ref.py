"""Synthetic checks of the reference planner maths."""
import numpy as np
from scipy.ndimage import gaussian_filter

import ref

rng = np.random.default_rng(0)
NY = NX = 60


def blobs(n, sigma, seed):
    r = np.random.default_rng(seed)
    g = np.zeros((NY, NX))
    for _ in range(n):
        g[r.integers(0, NY), r.integers(0, NX)] += r.uniform(0.5, 1.5)
    g = gaussian_filter(g, sigma) + 1e-6
    return g / g.sum()


def test_pd_curve():
    R = 140.0
    for z, exp in [(70, 0.90), (105, 0.70), (140, 0.48), (210, 0.21)]:
        assert abs(ref.pd_range(z, 0.5, R) - exp) < 0.02, (z, ref.pd_range(z, 0.5, R))
    assert abs(ref.pd_range(140, 0.3, R) - 0.16) < 0.02
    # a dry town is almost invisible, a flooded one is not
    dry = ref.pd_hat(84, R, 0.6, "home", np.array(0.0))
    wet = ref.pd_hat(84, R, 0.6, "home", np.array(1.0))
    print(f"pd at 84 m over town: dry {float(dry):.3f}, flooded (people on roofs) {float(wet):.3f}")
    assert dry < 0.02 and wet > 0.4


def test_scenario_posterior_recovers_mixture():
    comps = np.stack([blobs(6, 3, 1), blobs(4, 1.5, 2), blobs(10, 4, 3), blobs(5, 2, 4), np.ones((NY, NX))])
    comps = comps / comps.sum(axis=(1, 2), keepdims=True)
    K = comps.shape[0]
    prior_mean = np.array([0.45, 0.2, 0.2, 0.1, 0.05])
    W, S, LP = ref.scenario_grid(prior_mean)
    n_hat = 20.0
    truth_w = np.array([0.25, 0.0, 0.25, 0.5, 0.0])   # many people went to high ground
    truth_scale = 1.4
    lam_true = n_hat * truth_scale * np.tensordot(truth_w, comps, axes=1)
    people = rng.poisson(lam_true)                    # true counts per cell
    b = ref.ComponentBelief(comps, n_hat, W, S, LP)
    pd = 0.7
    order = rng.permutation(NY * NX)[: int(0.6 * NY * NX)]   # look at 60% of cells once
    found = 0
    for idx in order:
        iy, ix = divmod(int(idx), NX)
        lam_before = b.lam[:, iy, ix].copy()
        pdk = np.full((K, 1), pd)
        b.look(np.array([iy]), np.array([ix]), pdk)
        n_det = rng.binomial(people[iy, ix], pd)
        people[iy, ix] -= n_det
        for _ in range(n_det):
            b.confirmed_find(lam_before, pdk[:, 0])
            found += 1
    post = b.post()
    w_mean = post @ W
    s_mean = post @ S
    print("found", found, "posterior mean weights", np.round(w_mean, 2), "scale", round(float(s_mean), 2))
    print("truth weights", truth_w, "scale", truth_scale)
    assert w_mean[3] > 0.3 and w_mean[1] < 0.15
    assert 1.0 < s_mean < 2.0
    # remaining expected people vs truth remaining
    rem_belief = b.intensity().sum()
    print("remaining: belief", round(float(rem_belief), 1), "truth", int(people.sum()))


def test_value_function_shapes():
    lead = np.arange(0, 3601, 300.0)  # 0..60 min
    L = len(lead)
    t_now, t_end2 = 3600.0, 3600.0 + 2.5 * 3600 + 7200
    # 4 cells: dry forever; dangerous from 15 min; dangerous from 50 min; dangerous since long ago
    rate = np.zeros((L, 4))
    rate[lead >= 900, 1] = 1.0
    rate[lead >= 3000, 2] = 1.0
    rate[:, 3] = 1.0
    d_past = np.array([0.0, 0.0, 0.0, 3 * 3600.0])
    val, D = ref.survival_value(rate, lead, t_now, t_end2, d_past)
    np.set_printoptions(precision=3, suppress=True)
    print("value at leads 0,15,30,60 min:\n", val[[0, 3, 6, 12]])
    assert np.allclose(val[:, 0], 0)                     # dry: no value
    assert val[0, 1] > 0.9 and val[0, 1] > val[6, 1]     # soon-dangerous: high now, falls with delay
    assert val[0, 2] > 0.9 and abs(val[0, 2] - val[6, 2]) < 1e-6  # still safe for 50 min: no rush yet
    assert val[0, 3] < 0.02                               # already beyond help: little value


if __name__ == "__main__":
    test_pd_curve()
    test_scenario_posterior_recovers_mixture()
    test_value_function_shapes()
    print("ALL OK")
