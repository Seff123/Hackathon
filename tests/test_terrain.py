import numpy as np

from floodsim import hydrology
from floodsim import rng as R
from floodsim.scenario import sample_spec
from floodsim.terrain import generate_terrain, URBAN, FOREST, WATER


def _terrain(cfg, seed):
    spec = sample_spec(seed, cfg=cfg)
    return spec, generate_terrain(spec, cfg.world, R.streams(seed, spec.attempt)["terrain"])


def test_drainage_to_outlet(cfg):
    for seed in (0, 1, 2, 3):
        _, t = _terrain(cfg, seed)
        assert t.drain_outlet_frac >= 0.8


def test_filled_dem_no_pits_and_hand(cfg):
    _, t = _terrain(cfg, 1)
    zf = t.z_fill.astype(np.float64)
    inner = zf[1:-1, 1:-1]
    nb_min = np.min(np.stack([zf[1 + dy:zf.shape[0] - 1 + dy, 1 + dx:zf.shape[1] - 1 + dx]
                              for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]), axis=0)
    assert np.all(nb_min < inner), "interior pit in filled DEM"
    assert np.all(t.z_fill >= t.z - 1e-4)
    assert np.all(t.hand >= 0)
    assert np.all(t.hand[t.drainage_mask] == 0)


def test_priority_flood_small():
    z = np.array([[5, 5, 5, 5], [5, 1, 2, 5], [5, 2, 1, 5], [5, 5, 0, 5]], dtype=float)
    zf = hydrology.priority_flood(z)
    assert zf[1, 1] >= 1 and zf[2, 2] > 0
    down = hydrology.d8_directions(zf)
    acc, order = hydrology.flow_accumulation(zf, down)
    assert acc.max() >= 4


def test_landcover_and_features(cfg):
    spec, t = _terrain(cfg, 2)
    assert (t.landcover == WATER).sum() == t.river_mask.sum()
    assert len(t.towns) >= 1 and (t.landcover == URBAN).any()
    land = t.landcover != WATER
    ff = (t.landcover == FOREST).sum() / land.sum()
    assert abs(ff - spec.forest_fraction) < 0.08
    bx, by = t.base_xy
    assert t.road_mask[int(by // 50), int(bx // 50)]
    assert np.all(t.surface >= t.z)
    # river path is 4-connected and runs inflow -> outlet edge
    rc = t.river_cells
    iy, ix = np.divmod(rc, t.n)
    steps = np.abs(np.diff(iy)) + np.abs(np.diff(ix))
    assert np.all(steps == 1)
    assert t.outlet_edge_mask()[iy[-1], ix[-1]]
