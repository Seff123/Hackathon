"""V1: replay bundle (one JSON object per run) and the self-contained replay / compare HTML pages."""
from __future__ import annotations

import base64
import json
import math
import zlib
from pathlib import Path

import numpy as np

from ..autopilot import MODE_NAMES
from ..people import STATE_NAMES
from ..planner.params import COMPONENTS

LABELS = {"survival": "Survival-aware (ours)", "bayes": "Standard Bayesian search", "heuristic": "Flood-bonus heuristic",
          "scripted": "Scripted lawnmower"}
FRAME_DT = 30.0


def b64z(a: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(np.ascontiguousarray(a).tobytes(), 9)).decode("ascii")


def quantise(a: np.ndarray, scale: str, vmax: float | None = None):
    a = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0)
    if scale == "count":
        mx = float(vmax or 3)
        return np.clip(np.round(a), 0, mx).astype(np.uint8), mx
    mx = float(vmax) if vmax is not None else float(max(a.max(), 1e-9))
    mx = max(mx, float(a.max()), 1e-9)
    x = np.clip(a / mx, 0, 1)
    if scale == "sqrt":
        x = np.sqrt(x)
    return np.round(x * 255).astype(np.uint8), mx


def coarse(a, f=2, fn=np.mean):
    n = a.shape[-1]
    return fn(a.reshape(a.shape[:-2] + (n // f, f, n // f, f)), axis=(-3, -1))


def _static(world):
    t = world.terrain
    z100 = coarse(t.z.astype(np.float64))
    zmin, zmax = float(z100.min()), float(z100.max())
    elev = np.round((z100 - zmin) / max(zmax - zmin, 1e-6) * 255).astype(np.uint8)
    lc = t.landcover
    counts = np.stack([coarse((lc == k).astype(np.float64), fn=np.sum) for k in range(4)])
    lc100 = counts.argmax(0).astype(np.uint8)
    river = coarse(t.river_mask.astype(np.float64), fn=np.max).astype(np.uint8)
    rd = lambda pts, k=8: [[round(float(x), 1), round(float(y), 1)] for x, y in np.asarray(pts)[::k]]
    return {
        "elev_u8": b64z(elev), "elev_min": round(zmin, 2), "elev_max": round(zmax, 2),
        "landcover_u8": b64z(lc100), "river_u8": b64z(river),
        "river_line": rd(t.river_xy, 20),
        "roads": [rd(r, 5) for r in t.roads], "towns": [[tw["name"], round(tw["x"], 1), round(tw["y"], 1), round(tw["r"], 1)] for tw in t.towns],
        "base": [round(t.base_xy[0], 1), round(t.base_xy[1], 1)],
        "thermals": [[round(th.x, 1), round(th.y, 1), round(th.r_t, 1), round(th.w_core, 2)] for th in world.weather.thermals()],
        "levees": [rd(l, 1) for l in t.levee_lines],
    }


def _event_text(e):
    g = f"G{e['drone'] + 1}"
    k = e["kind"]
    if k == "detect":
        return f"{g} heat signature in cell {e.get('cell')}"
    if k == "confirm":
        return f"{g} confirmed a survivor" if e.get("true") else f"{g} confirmed (false alarm)"
    if k == "reject":
        return f"{g} verification: nobody there"
    if k == "rtb":
        return f"{g} returning to base ({e.get('reason', 'energy')})"
    if k == "landed":
        return f"{g} landed, battery swap"
    if k == "thermal_enter":
        return f"{g} circling in a thermal"
    if k == "thermal_exit":
        return f"{g} leaves the thermal"
    if k == "terrain_warning":
        return f"{g} terrain warning: climbing away"
    return f"{g} {k}"


def build_bundle(result, world, label: str | None = None) -> dict:
    rec = result.recording
    assert rec is not None and rec["frames"], "needs a run with record_grids=True"
    frames = rec["frames"]
    T = len(frames)
    n = world.terrain.n
    cap = result.energy_log["capacity_wh"]
    dbg = [d for d in rec.get("debug", []) if d]
    dbg_by_t = {round(d["t"]): d for d in dbg}

    def dbg_grid(key, k):
        d = dbg_by_t.get(round(frames[k]["t"]))
        if d is None or key not in d:
            return np.zeros((n // 2, n // 2))
        return np.asarray(d[key], dtype=np.float64)

    depth = np.stack([f["depth"].astype(np.float64) for f in frames])
    lam = np.stack([dbg_grid("lam", k) for k in range(T)])
    stake = np.stack([dbg_grid("stake", k) for k in range(T)])
    pf20 = np.stack([dbg_grid("pflood20", k) for k in range(T)])
    seen = np.stack([coarse(f["seen"].reshape(n, n).astype(np.float64), fn=np.sum) for f in frames])
    grids = {}
    for key, arr, scale, vmax in (("depth", depth, "linear", 3.0), ("lam", lam, "sqrt", None), ("stake", stake, "sqrt", None),
                                  ("pflood20", pf20, "linear", 1.0), ("seen", seen, "count", 3)):
        q, mx = quantise(arr, scale, vmax)
        grids[key] = {"data": b64z(q), "scale": scale, "max": round(float(mx), 6)}
    # people truth keyframes and the secured-survival race line
    P = np.stack([f["people"] for f in frames])          # [T, N, 6] x, y, state, found, surv, affected
    pf = {d["id"]: d for d in result.people_final}
    people = []
    t_found = {f["person"]: f for f in result.finds}
    for k in range(P.shape[1]):
        track = []
        last = None
        for j in range(T):
            x, y, st = float(P[j, k, 0]), float(P[j, k, 1]), int(P[j, k, 2])
            cur = (round(x), round(y), st)
            if cur != last:
                track.append([frames[j]["t"], round(x, 1), round(y, 1), STATE_NAMES[st]])
                last = cur
        fd = t_found.get(k)
        people.append({"id": k, "cat": ["home", "vehicle", "outdoor"][int(world.people.cat[k])],
                       "at_risk": bool(pf[k]["at_risk"]) if k in pf else bool(P[-1, k, 5] > 0.5),
                       "t_found": fd["t"] if fd else None,
                       "surv_final": round(float(pf[k]["surv"]), 4) if k in pf else None, "track": track})
    # at-risk flags from the final metrics (truth): people ever affected by the end of the scoring horizon
    n_at_risk = int(result.metrics.get("n_at_risk", 0))
    frames_out = []
    nd = result.n_drones
    for j, f in enumerate(frames):
        t = f["t"]
        found = [fd for fd in result.finds if fd["t"] <= t]
        secured = sum(fd["surv"] for fd in found if fd["at_risk"]) / max(n_at_risk, 1)
        drones = []
        for d in f["drones"]:
            x, y, z, ef, mode, goal, res = d
            drones.append([round(x, 1), round(y, 1), round(z, 1), round(ef, 4), mode, goal, round(res, 4)])
        db = dbg_by_t.get(round(t))
        belief = None
        if db is not None:
            belief = {"w": [round(float(v), 4) for v in db["w"]], "count_scale": db["count_scale"], "bias_m": db["bias_m"],
                      "wind": [round(db["wind"][0], 3), round(db["wind"][1], 3), round(1.0 / math.sqrt(max(nd, 1)), 3)]}
        frames_out.append({"t": t, "drones": drones, "belief": belief,
                           "kpi": {"secured": round(secured, 4), "found": len(found),
                                   "at_risk_found": sum(1 for fd in found if fd["at_risk"]),
                                   "airborne": sum(1 for d in drones if d[4] not in ("ground", "down")),
                                   "lost": 0}})
    lost_t = {e["drone"]: e["t"] for e in result.safety_events if e["kind"] == "down" and e.get("why") in ("crash", "midair", "geofence", "forced_landing")}
    for fo in frames_out:
        fo["kpi"]["lost"] = sum(1 for d, t in lost_t.items() if t <= fo["t"])
    events = []
    for e in result.events:
        if e["kind"] in ("arrived", "search_done", "cannot_progress"):
            continue
        events.append({"t": e["t"], "drone": e["drone"], "kind": e["kind"], "text": _event_text(e)})
    for d in result.decision_log:
        events.append({"t": d["t"], "drone": d["drone"], "kind": "decision", "text": d["text"]})
    events.sort(key=lambda e: (e["t"], e["drone"], e["kind"]))
    spec = world.spec
    m = {k: v for k, v in result.metrics.items() if isinstance(v, (int, float, str))}
    meta = {"seed": int(spec.seed), "controller": result.controller, "label": label or LABELS.get(result.controller, result.controller),
            "flood_type": spec.flood_type, "n_drones": int(nd), "wind_ms": round(float(spec.wind10_ms), 2),
            "wind_dir_deg": round(float(spec.wind_from_deg), 1),
            "wind_true": [round(float(v), 3) for v in world.weather.mean_wind(10.0)],
            "episode_s": float(world.cfg.world.episode_s), "frame_dt_s": FRAME_DT,
            "grid": {"nx": n // 2, "ny": n // 2, "cell_m": 2 * world.terrain.cell_size},
            "camera_range_m": float(spec.camera_range_m), "capacity_wh": cap, "metrics": m,
            "components": list(COMPONENTS), "n_at_risk": n_at_risk}
    return {"meta": meta, "static": _static(world), "grids": grids, "frames": frames_out, "people": people,
            "events": events}


def bundle_json(bundle: dict) -> str:
    return json.dumps(bundle, separators=(",", ":"), sort_keys=True)


# ----------------------------------------------------------------------------- callouts
def _town_arrivals(bundle, world, R):
    out = {}
    ter = world.terrain
    for name, tx, ty, tr in bundle["static"]["towns"]:
        first = None
        for f in bundle["frames"]:
            for d in f["drones"]:
                x, y, z, _, mode = d[:5]
                if mode != "search":
                    continue
                if math.hypot(x - tx, y - ty) <= tr:
                    iy, ix = min(int(y // ter.cell_size), ter.n - 1), min(int(x // ter.cell_size), ter.n - 1)
                    if z - float(ter.z[iy, ix]) < 0.8 * R:
                        first = f["t"]
                        break
            if first is not None:
                break
        out[name] = first
    return out


def _dry_minutes(bundle, world, t_max=3600.0):
    fc = world.forecast
    ter = world.terrain
    total = 0.0
    for f in bundle["frames"]:
        if f["t"] > t_max:
            break
        p = fc.prob_flooded(f["t"], 0.0)
        for d in f["drones"]:
            if d[4] != "search":
                continue
            iy, ix = min(int(d[1] // ter.cell_size), ter.n - 1), min(int(d[0] // ter.cell_size), ter.n - 1)
            if p[iy, ix] < 0.05:
                total += FRAME_DT / 60.0
    return total


def callouts(bA, bB, world) -> list:
    """Auto callouts comparing run A (baseline) and run B (ours). Each favours one side by a clear margin."""
    R = bA["meta"]["camera_range_m"]
    la, lb = bA["meta"]["label"], bB["meta"]["label"]
    short = lambda b: "Ours" if b["meta"]["controller"] == "survival" else b["meta"]["label"]
    out = []
    ta, tb = _town_arrivals(bA, world, R), _town_arrivals(bB, world, R)
    for town in ta:
        a, b = ta[town], tb[town]
        if a is None and b is None:
            continue
        if b is not None and (a is None or a - b >= 300):
            diff = "" if a is None else f" {(a - b) / 60:.0f} min earlier"
            out.append({"kind": "town", "town": town, "a": a, "b": b, "favours": "B",
                        "text": f"{short(bB)} reached {town}{diff}" + (f" ({la} never searched it low)" if a is None else "")})
        elif a is not None and (b is None or b - a >= 300):
            diff = "" if b is None else f" {(b - a) / 60:.0f} min earlier"
            out.append({"kind": "town", "town": town, "a": a, "b": b, "favours": "A",
                        "text": f"{short(bA)} reached {town}{diff}" + (f" ({lb} never searched it low)" if b is None else "")})
    fa, fb = bA["meta"]["metrics"]["at_risk_found"], bB["meta"]["metrics"]["at_risk_found"]
    if fa != fb:
        out.append({"kind": "at_risk_found", "a": fa, "b": fb, "favours": "B" if fb > fa else "A",
                    "text": f"At-risk people found: {fb} vs {fa}" if fb > fa else f"At-risk people found: {fa} vs {fb} ({short(bA)} ahead)"})
    sa, sb = bA["meta"]["metrics"]["survival_at_risk"], bB["meta"]["metrics"]["survival_at_risk"]
    if abs(sb - sa) >= 0.03:
        out.append({"kind": "survival", "a": sa, "b": sb, "favours": "B" if sb > sa else "A",
                    "text": f"Survival of at-risk people: {sb:.2f} ({short(bB)}) vs {sa:.2f} ({short(bA)})"})
    da, db = _dry_minutes(bA, world), _dry_minutes(bB, world)
    if abs(da - db) >= 10:
        fav = "B" if db < da else "A"
        lo, hi = (bB, bA) if fav == "B" else (bA, bB)
        out.append({"kind": "dry", "a": round(da, 1), "b": round(db, 1), "favours": fav,
                    "text": f"In the first hour {short(hi)} spent {max(da, db):.0f} drone-minutes searching dry ground "
                            f"(forecast flood chance < 5%), {short(lo)} {min(da, db):.0f}"})
    return out


# ----------------------------------------------------------------------------- pages
TEMPLATE = Path(__file__).with_name("page.html")


def write_page(bundles: list, path, title: str, subtitle: str = "", calls: list | None = None,
               results: dict | None = None) -> Path:
    html = TEMPLATE.read_text()
    esc = lambda s: s.replace("</", "<\\/")
    scripts = "\n".join('<script type="application/json" class="bundle">' + esc(bundle_json(b)) + "</script>"
                        for b in bundles)
    extra = json.dumps({"title": title, "subtitle": subtitle, "callouts": calls or [], "results": results or {}},
                       separators=(",", ":"), sort_keys=True).replace("</", "<\\/")
    html = html.replace("<!--BUNDLES-->", scripts).replace("/*PAGE*/{}", extra).replace("<!--TITLE-->", title)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html)
    return p
