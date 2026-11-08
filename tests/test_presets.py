import numpy as np
import pytest

from floodsim.presets import PRESETS
from floodsim.scenario import build_world, sample_spec


def test_ordinary_spec_hash_unchanged_by_preset_field():
    spec = sample_spec(3)
    assert spec.preset is None
    assert "preset" not in spec.to_dict()     # existing flood caches keep their keys


@pytest.mark.parametrize("key", list(PRESETS))
def test_preset_samples_deterministically(key):
    a = sample_spec(1, preset=key)
    b = sample_spec(1, preset=key)
    assert a.spec_hash == b.spec_hash
    assert a.preset["name"] == key
    assert a.spec_hash != sample_spec(1).spec_hash
    for f, bnd in PRESETS[key].bounds.items():
        if hasattr(bnd, "contains") and not isinstance(getattr(a, f), list):
            assert bnd.contains(getattr(a, f)), (f, getattr(a, f))


def test_town_far_places_town_far_from_river():
    spec = sample_spec(2, preset="town_far")
    w = build_world(spec, cache_dir="cache/")
    ter = w.terrain
    c = ter.cell_size
    lo, hi = PRESETS["town_far"].knobs["town_river_dist_m"]
    for t in ter.towns:
        d = ter.dist_to_river[int(t["y"] // c), int(t["x"] // c)]
        assert lo - c <= d <= hi + c


def test_hills_escape_shorter_than_valley():
    from floodsim.vizpage.showcase import flood_features
    esc = {}
    for key in ("escape_hills", "trapped_valley"):
        vals = []
        for s in (1, 2, 3):
            f = flood_features(build_world(sample_spec(s, preset=key), cache_dir="cache/"))
            if np.isfinite(f["escape_m"]):
                vals.append(f["escape_m"])
        esc[key] = np.median(vals)
    assert esc["escape_hills"] < esc["trapped_valley"]
