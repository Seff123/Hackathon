"""V0: recording hooks. Run an episode with record_grids=True and save raw recordings."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import to_jsonable


def record_episode(seed: int, controller: str, n_drones=None, cache="cache/", team=None, config=None,
                   controller_kw=None):
    from ..runner import run_episode
    return run_episode(seed, controller, n_drones, config, team, cache, fast=False, record=True,
                       controller_kw=controller_kw)


def save_recording(result, world, out_dir) -> Path:
    """out/rec_<seed>_<controller>.npz (grids) + events.jsonl (events and decisions)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = result.recording
    tag = f"{result.seed}_{result.controller}"
    frames = rec["frames"]
    arrays = {"t": np.array([f["t"] for f in frames]),
              "depth": np.stack([f["depth"] for f in frames]),
              "hazard": np.stack([f["hazard"] for f in frames]),
              "seen": np.stack([f["seen"] for f in frames]),
              "people": np.stack([f["people"] for f in frames])}
    dbg = [d for d in rec.get("debug", []) if d]
    if dbg:
        for key in ("lam", "stake", "pflood20"):
            if all(key in d for d in dbg):
                arrays[f"dbg_{key}"] = np.stack([d[key] for d in dbg])
        arrays["dbg_t"] = np.array([d["t"] for d in dbg])
    np.savez_compressed(out / f"rec_{tag}.npz", **arrays)
    with open(out / f"events_{tag}.jsonl", "w") as fh:
        for f in frames:
            fh.write(json.dumps({"kind": "frame", "t": f["t"], "drones": to_jsonable(f["drones"])}) + "\n")
        for d in dbg:
            fh.write(json.dumps({"kind": "belief", "t": d["t"], "w": d["w"], "count_scale": d["count_scale"],
                                 "bias_m": d["bias_m"], "wind": d["wind"], "claims": d["claims"]}) + "\n")
        for e in result.events:
            fh.write(json.dumps(to_jsonable({"kind": "event", **e})) + "\n")
        for d in result.decision_log:
            fh.write(json.dumps(to_jsonable({"kind": "decision", **d})) + "\n")
    return out / f"rec_{tag}.npz"
