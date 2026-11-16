import numpy as np
import pytest

from floodsim.config import Config
from floodsim.forecast import brier_score
from floodsim.scenario import sample_spec
from floodsim.world import build_world

from helpers import reachable_truth

CFG = Config()


@pytest.fixture(scope="module")
def world():
    return build_world(sample_spec(1, cfg=CFG), CFG)


def test_probabilities_valid(world):
    fc = world.forecast
    for t in (0.0, 1800.0, world.episode_s):
        for L in (0.0, 900.0, 3600.0):
            p = fc.prob_flooded(t, L)
            assert p.shape == world.terrain.z.shape
            assert np.all((p >= 0) & (p <= 1))
            d = fc.expected_depth(t, L)
            assert np.all(d >= 0) and np.isfinite(d).all()
            assert np.isfinite(fc.expected_hazard(t, L)).all()


def test_no_truth_reference(world):
    assert reachable_truth(world.forecast) is None


def test_smooth_errors(world):
    fc, terr, fl = world.forecast, world.terrain, world.flood
    lookup = {int(d): k for k, d in enumerate(fc.drains)}
    cells = [int(iy * terr.n + ix) for iy, ix in terr.river_order]
    cells = [c for c in cells if c in lookup]
    depth = fl.depth_grid(0.0).ravel()
    z, zf = terr.z.ravel(), terr.z_fill.ravel()
    errs = []
    for i in range(fc.stage_fc.shape[0]):
        d = fl.depth_grid(i * fc.issue_dt).ravel()
        true = np.where(d[cells] > 0, d[cells] + z[cells] - zf[cells], CFG.forecast.dry_stage_m)
        errs.append(fc.stage_fc[i, 0, [lookup[c] for c in cells]].astype(np.float64) - true)
    e = np.mean(errs, axis=0)  # mean error profile along the river
    lag = 20  # 20 cells of 50 m ~ 1 km along the river
    a, b = e[:-lag], e[lag:]
    # remove the scenario-wide bias before correlating
    assert np.corrcoef(a - a.mean(), b - b.mean())[0, 1] > 0.5


@pytest.mark.slow
def test_brier_informative_but_imperfect():
    scores = []
    for seed in range(20):
        w = build_world(sample_spec(seed, cfg=CFG), CFG)
        scores.append(brier_score(w, 1800.0))
    assert all(0.02 <= s <= 0.2 for s in scores), scores
