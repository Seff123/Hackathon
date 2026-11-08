"""Named scenario presets: situations chosen to show where a survival-aware planner and standard Bayesian search
should differ (big town in or far from the flood, easy escape to hills or trapped in a wide valley, sudden
breach or slow rise, few gliders for many people at risk).

A preset narrows the scenario sampling ranges and sets a few generator knobs; each seed still gives a different
random world inside the preset, and every world must pass the usual rejection checks (with any relaxation the
preset states). The knobs are stored on the spec (`spec.preset`), so they are covered by the spec hash and the
flood cache.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .config import Bounds, ScenarioBounds


@dataclass(frozen=True)
class Preset:
    key: str
    title: str
    blurb: str          # what the situation is
    expect: str         # what we expect to see between the planners, stated before running
    bounds: dict = field(default_factory=dict)     # ScenarioBounds fields replaced
    knobs: dict = field(default_factory=dict)      # generator / rejection knobs (see terrain, scenario)


FLUVIAL = {"fluvial": 1.0}
# a small channel overtops sooner, so the same inflow puts more water on the floodplain
SMALL_CHANNEL = dict(channel_width_m=Bounds(20.0, 35.0), channel_depth_m=Bounds(2.0, 2.5))

PRESETS: dict[str, Preset] = {p.key: p for p in [
    Preset("town_in_flood", "Big town in the flood",
           "A large town sits on the floodplain next to the river and the river floods it. Most people live "
           "where the water goes.",
           "Little difference: where people probably are and where lives are at stake are the same place.",
           bounds=dict(flood_type={"fluvial": 1.0, "fluvial+pluvial": 1.0}, n_towns=Bounds(1, 2, "int"),
                       town_radius_m=Bounds(900.0, 1200.0), levee_height_m=Bounds(0.0, 0.5),
                       peak_inflow_m3s=Bounds(1000.0, 1500.0, "loguniform"), valley_width_m=Bounds(600.0, 1000.0),
                       valley_depth_m=Bounds(10.0, 25.0), channel_width_m=Bounds(20.0, 35.0),
                       channel_depth_m=Bounds(2.0, 2.5), n_targets=Bounds(18, 25, "int"),
                       category_mean=(0.75, 0.15, 0.1)),
           knobs=dict(town_river_dist_m=(0.0, 300.0), extent_frac=(0.03, 0.5), min_targets_in_flood=5)),
    Preset("town_far", "Big town far from the flood",
           "The large town is on dry ground 2.5-4 km from the river. The people the flood reaches are in "
           "scattered homes, cars on low roads and people outdoors near the river.",
           "Survival-aware ahead: Bayesian search is drawn to the dry town, where most people probably are.",
           bounds=dict(flood_type=FLUVIAL, n_towns=Bounds(1, 1, "int"), town_radius_m=Bounds(900.0, 1200.0),
                       peak_inflow_m3s=Bounds(1000.0, 1500.0, "loguniform"), **SMALL_CHANNEL,
                       n_targets=Bounds(18, 25, "int"), category_mean=(0.25, 0.4, 0.35)),
           knobs=dict(town_river_dist_m=(2500.0, 4000.0), targets_near_flood_frac=0.15, min_targets_in_flood=3,
                      extent_frac=(0.03, 0.5), min_towns=1)),
    Preset("escape_hills", "Easy escape to high ground",
           "A narrow, deep valley between steep hills. Dry ground is a short walk from anywhere the water goes, "
           "and most people can walk.",
           "Little difference: most people reach the hills by themselves, so fewer depend on being found.",
           bounds=dict(flood_type=FLUVIAL, valley_width_m=Bounds(300.0, 500.0), valley_depth_m=Bounds(45.0, 60.0),
                       relief_m=Bounds(150.0, 300.0, "loguniform"), immobile_fraction=Bounds(0.1, 0.2),
                       peak_inflow_m3s=Bounds(800.0, 1500.0, "loguniform"), **SMALL_CHANNEL,
                       n_targets=Bounds(15, 25, "int")),
           knobs=dict(extent_frac=(0.02, 0.4), town_river_dist_m=(0.0, 400.0), min_targets_in_flood=4)),
    Preset("trapped_valley", "Trapped in a wide valley",
           "A wide, flat valley floor with low hills far away. When the river floods, dry ground is a long walk "
           "through water.",
           "Survival-aware ahead: people cannot walk out, so who is reached first decides who survives.",
           bounds=dict(flood_type=FLUVIAL, valley_width_m=Bounds(2500.0, 3500.0), valley_depth_m=Bounds(8.0, 15.0),
                       relief_m=Bounds(20.0, 40.0, "loguniform"), regional_slope=Bounds(0.001, 0.002),
                       immobile_fraction=Bounds(0.1, 0.2), peak_inflow_m3s=Bounds(1000.0, 1500.0, "loguniform"),
                       **SMALL_CHANNEL, floodplain_noise_m=Bounds(0.5, 0.8), n_targets=Bounds(15, 25, "int")),
           knobs=dict(extent_frac=(0.03, 0.6), town_river_dist_m=(0.0, 800.0), min_targets_in_flood=4)),
    Preset("sudden_breach", "Sudden levee breach",
           "A levee protects a town near the river and fails early in the sortie. Deep, fast water reaches homes "
           "with almost no warning.",
           "Survival-aware ahead, if its forecast sees the breach area: urgency is highest just behind the breach.",
           bounds=dict(flood_type={"breach": 1.0}, levee_height_m=Bounds(2.0, 3.0), breach_time_h=Bounds(0.1, 0.4),
                       breach_width_m=Bounds(60.0, 100.0), breach_head_m=Bounds(2.5, 4.0),
                       town_radius_m=Bounds(700.0, 1100.0), n_targets=Bounds(15, 25, "int"),
                       category_mean=(0.7, 0.15, 0.15)),
           knobs=dict(town_river_dist_m=(300.0, 900.0), min_targets_in_flood=4)),
    Preset("slow_rise", "Slow-rising river",
           "The river is still rising at launch and peaks late in the sortie. At first few people are in danger; "
           "by the end many are.",
           "Survival-aware ahead early: the forecast says where the water will go, so it can search there first.",
           bounds=dict(flood_type=FLUVIAL, time_to_peak_h=Bounds(3.0, 4.5), start_offset_h=Bounds(0.0, 0.5),
                       peak_inflow_m3s=Bounds(1000.0, 1500.0, "loguniform"), **SMALL_CHANNEL,
                       n_targets=Bounds(15, 25, "int")),
           knobs=dict(extent_frac=(0.02, 0.5), town_river_dist_m=(0.0, 600.0), min_targets_in_flood=4)),
    Preset("few_gliders", "Two gliders, many people at risk",
           "Only two gliders for a flood that reaches a lot of people. The fleet cannot search everywhere, so it "
           "has to choose.",
           "Survival-aware ahead: with little search time to spend, where it is spent matters most.",
           bounds=dict(flood_type={"fluvial": 1.0, "fluvial+pluvial": 1.0}, n_drones=Bounds(2, 2, "int"),
                       town_radius_m=Bounds(700.0, 1100.0), peak_inflow_m3s=Bounds(600.0, 1500.0, "loguniform"),
                       n_targets=Bounds(20, 25, "int"), category_mean=(0.65, 0.2, 0.15)),
           knobs=dict(town_river_dist_m=(100.0, 700.0), min_targets_in_flood=4)),
    Preset("extreme_2076", "Extreme 2076 flood",
           "The top of the 2076 climate range: storm rain and river flow both 1.4 times today's extremes, river "
           "and surface flooding at once, low levees overtopped, full cloud so no thermals, and a strong wind. "
           "A large part of the valley and its towns goes under water during the sortie.",
           "Survival-aware ahead: with more people at risk than the fleet can reach, choosing who is most urgent "
           "matters most; but the forecast's depth errors also grow, which may cancel the gain.",
           bounds=dict(flood_type={"fluvial+pluvial": 1.0}, climate_factor=Bounds(1.4, 1.4),
                       peak_rain_mmh=Bounds(40.0, 60.0), storm_radius_km=Bounds(6.0, 8.0),
                       storm_duration_h=Bounds(2.0, 4.0),
                       peak_inflow_m3s=Bounds(1400.0, 2100.0, "loguniform"),   # 1000-1500 x 1.4
                       **SMALL_CHANNEL, levee_height_m=Bounds(0.5, 1.5), time_to_peak_h=Bounds(1.0, 2.0),
                       start_offset_h=Bounds(0.5, 1.5), valley_width_m=Bounds(1000.0, 2000.0),
                       town_radius_m=Bounds(800.0, 1200.0), n_towns=Bounds(2, 3, "int"),
                       n_targets=Bounds(20, 25, "int"), cloud_cover=Bounds(0.95, 1.0), wind10_ms=Bounds(5.0, 8.0)),
           knobs=dict(town_river_dist_m=(0.0, 900.0), extent_frac=(0.08, 0.75), min_targets_in_flood=8)),
]}


def preset_bounds(key: str, base: ScenarioBounds) -> ScenarioBounds:
    return replace(base, **PRESETS[key].bounds)


def preset_tag(key: str) -> dict:
    """What goes on the spec: the preset name and its knobs (lists, so the spec stays JSON-like)."""
    k = {a: (list(v) if isinstance(v, tuple) else v) for a, v in PRESETS[key].knobs.items()}
    return {"name": key, **k}


def knob(spec, name, default=None):
    p = getattr(spec, "preset", None)
    return default if not p else p.get(name, default)
