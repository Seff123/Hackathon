import os
from pathlib import Path

import pytest

from floodsim.config import load_config

CACHE = Path(__file__).resolve().parent.parent / "cache"


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def cache_dir():
    CACHE.mkdir(exist_ok=True)
    return str(CACHE)


_WORLDS = {}


def get_world(seed, cfg=None, **overrides):
    from floodsim.scenario import build_world, sample_spec
    cfg = cfg or load_config()
    key = (seed, tuple(sorted(overrides.items())))
    if key not in _WORLDS:
        spec = sample_spec(seed, cfg=cfg, **overrides)
        _WORLDS[key] = build_world(spec, cache_dir=str(CACHE), cfg=cfg)
    return _WORLDS[key]


def skip_perf():
    return os.environ.get("FLOODSIM_SKIP_PERF") == "1"


def slow_enabled():
    return os.environ.get("FLOODSIM_SLOW", "1") == "1"
