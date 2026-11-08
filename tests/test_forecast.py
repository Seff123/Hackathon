import numpy as np
import pytest

from floodsim.flood import FloodTimeline
from floodsim.forecast import _ar1_profile

from conftest import get_world, slow_enabled


def _reachable_types(obj, seen=None, depth=0):
    seen = seen if seen is not None else set()
    if id(obj) in seen or depth > 6:
        return set()
    seen.add(id(obj))
    out = {type(obj)}
    if isinstance(obj, dict):
        for v in obj.values():
            out |= _reachable_types(v, seen, depth + 1)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            out |= _reachable_types(v, seen, depth + 1)
    elif hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            out |= _reachable_types(v, seen, depth + 1)
    return out


def test_no_truth_reference(cfg):
    w = get_world(1, cfg)
    types = _reachable_types(w.forecast)
    assert FloodTimeline not in types
    assert type(w.weather) not in types and type(w.people) not in types


def test_probabilities_valid(cfg):
    w = get_world(1, cfg)
    for t in (0.0, 3000.0, 9000.0):
        for L in (0.0, 1200.0, 3600.0):
            p = w.forecast.prob_flooded(t, L)
            assert p.shape == (200, 200) and np.all((p >= 0) & (p <= 1))
            assert np.all(w.forecast.expected_depth(t, L) >= 0)
            assert np.isfinite(w.forecast.expected_hazard(t, L)).all()
    assert w.forecast.observed_extent(100.0).dtype == bool
    p0 = w.forecast.prob_flooded(600.0, 1200.0)
    p1 = w.forecast.prob_flooded(600.0, 1200.0, stage_offset=0.3)
    assert p1.mean() >= p0.mean()


@pytest.mark.skipif(not slow_enabled(), reason="slow")
def test_brier_informative_but_imperfect(cfg):
    for seed in range(20):
        w = get_world(seed, cfg)
        b = w.forecast.brier[30]
        assert 0.02 <= b <= 0.2, (seed, b)


def test_smooth_errors_along_river(cfg):
    """Correlation (across error draws) of the stage error at river cells ~1 km apart along the river."""
    w = get_world(2, cfg)
    t = w.terrain
    didx = np.flatnonzero(t.drainage_mask.reshape(-1))
    pos = {c: k for k, c in enumerate(didx.tolist())}
    rc = [pos[c] for c in t.river_cells.tolist()]
    E = np.stack([_ar1_profile(t, didx, np.random.default_rng(s), cfg.forecast.ar_corr_m) for s in range(300)])
    corrs = [np.corrcoef(E[:, rc[k]], E[:, rc[k + 20]])[0, 1] for k in range(0, len(rc) - 20, 15)]
    assert np.mean(corrs) > 0.5, np.mean(corrs)
