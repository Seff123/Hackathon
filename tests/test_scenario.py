import dataclasses
import hashlib
import json

import numpy as np
import pytest

from floodsim.config import Bounds, ScenarioSpec, load_config
from floodsim.scenario import _sample_raw, sample_seeds, sample_spec, spec_from_split, write_seed_split, strata_of
from floodsim.terrain import generate_terrain
from floodsim import rng as R


def test_sampler_respects_bounds_over_200_seeds(cfg):
    b = cfg.bounds
    for seed in range(200):
        spec = _sample_raw(seed, 0, b, cfg.team, {})
        d = spec.to_dict()
        for f in dataclasses.fields(b):
            v = getattr(b, f.name)
            if isinstance(v, Bounds) and f.name in d and not isinstance(d[f.name], list):
                assert v.contains(d[f.name]), (seed, f.name, d[f.name])
        assert all(b.town_radius_m.contains(r) for r in spec.town_radius_m)
        assert len(spec.town_radius_m) == spec.n_towns
        assert spec.flood_type in b.flood_type and spec.outlet_edge in "NESW"
        assert abs(sum(spec.category_weights) - 1) < 1e-9
        assert 0 <= spec.wstar_ms <= 2.5 * 1.4 * (1 - spec.cloud_cover) + 1e-9
        assert b.peak_rain_mmh.contains(spec.peak_rain_mmh)
        assert 2 <= spec.n_drones <= 8 and 3 <= spec.n_targets <= 25


def _hash(*arrays):
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def test_determinism_spec_and_terrain(cfg):
    a = sample_spec(3, cfg=cfg)
    b = sample_spec(3, cfg=cfg)
    assert a.spec_hash == b.spec_hash
    ta = generate_terrain(a, cfg.world, R.streams(3, a.attempt)["terrain"])
    tb = generate_terrain(b, cfg.world, R.streams(3, b.attempt)["terrain"])
    assert _hash(ta.z, ta.hand, ta.landcover, ta.surface) == _hash(tb.z, tb.hand, tb.landcover, tb.surface)


def test_spec_json_roundtrip(cfg):
    a = sample_spec(5, cfg=cfg)
    b = ScenarioSpec.from_dict(json.loads(json.dumps(a.to_dict())))
    assert a.spec_hash == b.spec_hash


def test_streams_independent():
    s1 = R.streams(1, 0)
    s2 = R.streams(1, 0)
    s2["terrain"].standard_normal(1000)    # consuming one stream must not change another
    assert s1["people"].uniform() == s2["people"].uniform()


def test_reproducible_seed_splits(cfg, tmp_path):
    seeds = [1, 2, 7]
    rec = write_seed_split(tmp_path / "seeds_test.json", seeds, cfg)
    split = json.loads((tmp_path / "seeds_test.json").read_text())
    for s in seeds:
        assert spec_from_split(s, split, cfg).spec_hash == rec[str(s)]["spec_hash"]


def test_sample_seeds_balanced():
    seeds = sample_seeds(56)
    strata = [strata_of(s) for s in seeds]
    types = {t for t, _ in strata}
    sizes = {n for _, n in strata}
    assert len(types) == 4 and len(sizes) == 7
    assert len(set(seeds)) == len(seeds)


def test_rejection_checks_hold(cfg):
    from floodsim.people import place_people
    from floodsim.scenario import rejection_checks
    for seed in (0, 1, 2):
        spec = sample_spec(seed, cfg=cfg)
        st = R.streams(seed, spec.attempt)
        ter = generate_terrain(spec, cfg.world, st["terrain"])
        ppl = place_people(spec, ter, st["people"], cfg.people)
        assert rejection_checks(spec, ter, ppl, cfg.world, cfg) == []
