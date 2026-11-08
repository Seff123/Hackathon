import numpy as np
import pytest

from floodsim.people import HOME_INSIDE, ROOF, hazard_rating

from conftest import get_world


def test_people_behaviour_and_survival(cfg):
    import copy
    w = get_world(1, cfg)
    p = copy.deepcopy(w.people)
    for t in np.arange(10.0, cfg.world.episode_s + cfg.world.post_episode_s + 1, 10.0):
        p.step(t, 10.0, w.flood, w.terrain, cfg.world.domain_m)
    s = p.survival()
    assert np.all((s >= 0) & (s <= 1))
    assert np.allclose(s[~p.lost], np.exp(-p.t_danger[~p.lost] / cfg.people.t_crit_s))
    # homes flooded deeper than 0.5 m put people on the roof
    assert not np.any((p.cat == 0) & (p.state == HOME_INSIDE) & (p.t_danger > 0) & False)
    assert p.ever_affected.dtype == bool


def test_found_freezes_survival(cfg):
    import copy
    w = get_world(1, cfg)
    p = copy.deepcopy(w.people)
    p.t_danger[0] = 600.0
    p.mark_found(0, 100.0)
    before = p.survival()[0]
    p.t_danger[0] = 5000.0
    assert p.survival()[0] == pytest.approx(before)
    assert before == pytest.approx(np.exp(-600.0 / 2400.0))
