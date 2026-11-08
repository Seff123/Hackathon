import base64
import json
import re
import shutil
import subprocess
import zlib
from pathlib import Path

import numpy as np
import pytest

from floodsim.vizpage.export import build_bundle, bundle_json, callouts, write_page

PAGE = Path(__file__).resolve().parents[2] / "floodsim" / "vizpage" / "page.html"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


@pytest.fixture(scope="module")
def recorded():
    from floodsim.vizpage.record import record_episode
    runs = [record_episode(1, c, 3) for c in ("bayes", "survival")]
    return runs


@pytest.fixture(scope="module")
def bundles(recorded):
    return [build_bundle(r, w) for r, w in recorded]


def _decode(b, key):
    g = b["grids"][key]
    raw = np.frombuffer(zlib.decompress(base64.b64decode(g["data"])), dtype=np.uint8)
    T = len(b["frames"])
    return raw.reshape(T, b["meta"]["grid"]["ny"], b["meta"]["grid"]["nx"])


def test_bundle_shape(bundles, recorded):
    for b, (res, world) in zip(bundles, recorded):
        T = len(b["frames"])
        assert T == int(world.cfg.world.episode_s / 30) + 1
        for key in ("depth", "lam", "stake", "pflood20", "seen"):
            arr = _decode(b, key)
            assert arr.shape == (T, 100, 100)
        depth_true = np.stack([f["depth"].astype(float) for f in res.recording["frames"]])
        assert b["grids"]["depth"]["max"] >= depth_true.max() - 1e-6 or depth_true.max() <= 3.0
        assert not np.isnan(depth_true).any()
        for key in ("lam", "stake"):
            dbg = [np.asarray(d[key], float) for d in res.recording["debug"] if d and key in d]
            assert b["grids"][key]["max"] >= max(x.max() for x in dbg) - 1e-4   # max is stored to 4 decimals


def test_sizes(bundles, tmp_path, recorded):
    for b in bundles:
        assert len(bundle_json(b)) <= 4e6
    p = write_page(bundles, tmp_path / "compare.html", "t", "s", callouts(bundles[0], bundles[1], recorded[1][1]))
    assert p.stat().st_size <= 10e6
    from floodsim.vizpage.demo3d import demo3d
    res, world = recorded[1]
    p3 = demo3d(res, world, tmp_path / "f3d.html")
    assert Path(p3).stat().st_size <= 10e6


def test_orientation_grid_to_canvas():
    src = PAGE.read_text()
    fn = re.search(r"function gridToCanvas\(.*?\n", src).group(0)
    if shutil.which("node"):
        out = subprocess.run(["node", "-e", fn + "console.log(JSON.stringify(gridToCanvas(5, 80, 100, 8)))"],
                             capture_output=True, text=True, check=True).stdout
        x, y = json.loads(out)
    else:  # pragma: no cover
        x, y = 80 * 8, (100 - 1 - 5) * 8
    W = 800
    assert x > W / 2 and y > W / 2          # hot cell (iy=5, ix=80) is in the lower right of the canvas


def test_truth_labelling():
    src = PAGE.read_text()
    # the hidden-truth toggle, the flood-depth legend and the tooltip depth line all carry the truth tag
    assert re.search(r'id="truth"[^>]*>\s*Show hidden truth <span class="tag truth"[^>]*>truth, hidden from the fleet', src)
    assert re.search(r'Flood depth <span class="tag truth"[^>]*>truth, hidden from the fleet', src)
    assert re.search(r'Depth now .*?class="tag truth"', src)
    for m in re.finditer(r'<label><input type="radio" name="heat" value="(stake|lam)"[^\n]*', src):
        assert 'class="tag belief"' in m.group(0)


def test_callouts_hand_calc(bundles, recorded):
    world = recorded[1][1]
    calls = callouts(bundles[0], bundles[1], world)
    ma, mb = bundles[0]["meta"]["metrics"], bundles[1]["meta"]["metrics"]
    for c in calls:
        if c["kind"] == "at_risk_found":
            assert (c["a"], c["b"]) == (ma["at_risk_found"], mb["at_risk_found"])
        if c["kind"] == "survival":
            assert c["a"] == pytest.approx(ma["survival_at_risk"]) and c["b"] == pytest.approx(mb["survival_at_risk"])
            assert abs(c["a"] - c["b"]) >= 0.03
        if c["kind"] == "town":
            # hand calculation: first frame with a searching drone inside the town radius below 0.8 R
            name = c["town"]
            tw = [t for t in bundles[1]["static"]["towns"] if t[0] == name][0]
            first = None
            for f in bundles[1]["frames"]:
                for d in f["drones"]:
                    if d[4] == "search" and np.hypot(d[0] - tw[1], d[1] - tw[2]) <= tw[3]:
                        iy, ix = int(d[1] // 50), int(d[0] // 50)
                        if d[2] - world.terrain.z[iy, ix] < 0.8 * bundles[1]["meta"]["camera_range_m"]:
                            first = f["t"]
                            break
                if first is not None:
                    break
            assert first == c["b"]
    if ma["at_risk_found"] != mb["at_risk_found"]:
        assert any(c["kind"] == "at_risk_found" for c in calls)


def test_determinism(recorded):
    res, world = recorded[0]
    assert bundle_json(build_bundle(res, world)) == bundle_json(build_bundle(res, world))


def test_charts_match_summary(tmp_path):
    from floodsim.planner.experiments import summarize
    from floodsim.runner import write_rows
    from floodsim.vizpage.charts import make_charts
    rng = np.random.default_rng(0)
    by = {}
    for c in ("scripted", "bayes", "heuristic", "survival"):
        rows = []
        for s in range(10):
            rows.append({"seed": s, "controller": c, "survival_at_risk": float(rng.uniform(0.3, 0.9)),
                         "found_fraction": 0.3, "time_to_first_find_s": 600.0, "at_risk_found": 2, "crashes": 0,
                         "forced_landings": 0, "separation_breaches": 0, "high_risk_coverage": 0.3,
                         "fa_verify_time_s": 100.0, "n_at_risk": 3, "flood_type": ["fluvial", "pluvial"][s % 2],
                         "n_drones": 4 + s % 2, "secured_curve": list(np.linspace(0, 0.5, 31))})
        by[c] = rows
    summ = summarize(by)
    d = tmp_path / "compare_x"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps(summ))
    write_rows([r for rs in by.values() for r in rs], d / "episodes.csv")
    out = make_charts(d, tmp_path / "site" / "charts")
    data = json.loads((out / "charts_data.json").read_text())
    for row in data["1_headline"]:
        p = summ["paired"][f"survival-{row['vs']}"]
        assert row["mean"] == round(p["mean"], 3) and row["ci_lo"] == round(p["ci95"][0], 3)
    for c, ms in data["4_by_flood_type"].items():
        if c == "types":
            continue
        for t, m in zip(data["4_by_flood_type"]["types"], ms):
            assert m == round(summ["by_flood_type"][t][c]["mean"], 3)
    for row in data["8_safety"]:
        assert row["crashes"] == round(summ["controllers"][row["controller"]]["crashes"]["mean"], 3)
    assert (tmp_path / "site" / "results.html").exists()
    assert len(list(out.glob("*.png"))) >= 3


@pytest.mark.skipif(not Path(CHROME).exists(), reason="Chrome not available")
def test_smoke_headless(bundles, recorded, tmp_path):
    p = write_page(bundles, tmp_path / "compare.html", "smoke", "", callouts(bundles[0], bundles[1], recorded[1][1]))
    site = Path(__file__).resolve().parents[2] / "site"
    site.mkdir(exist_ok=True)
    shot = site / "smoke.png"
    r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--window-size=1500,1400", "--virtual-time-budget=8000",
                        "--enable-logging=stderr", "--v=0", f"--screenshot={shot}", f"file://{p}"],
                       capture_output=True, text=True, timeout=120)
    errs = [l for l in r.stderr.splitlines() if "CONSOLE" in l and ("Uncaught" in l or "Error" in l)]
    assert not errs, errs
    assert shot.exists() and shot.stat().st_size > 50_000


def test_recording_does_not_change_planner():
    """The debug snapshot (lives-at-stake map) must not change what the survival planner does."""
    from floodsim.runner import run_episode
    # seed and tuned parameters where the bug showed up (recorded survival 0.638 vs 0.550 unrecorded)
    kw = {"params": {'beta_travel': 0.8828, 'mu_energy': 0.3118, 'overlap': 0.3338, 'hysteresis': 0.2856, 'replan_s': 111.0325, 'nu_verify': 0.137, 'mobile_fraction': 0.3413, 'scenario_concentration': 13.3834, 't_crit_hat_s': 3832.9131, 'sigma_hr': 0.541, 'tail_factor': 0.8807, 'eps_welfare': 0.015}}
    a, _ = run_episode(20013, "survival", None, None, None, "cache/", fast=True, record=False, controller_kw=kw)
    b, _ = run_episode(20013, "survival", None, None, None, "cache/", fast=False, record=True, controller_kw=kw)
    assert a.metrics["survival_at_risk"] == b.metrics["survival_at_risk"]
    assert [(f["person"], f["t"]) for f in a.finds] == [(f["person"], f["t"]) for f in b.finds]
