"""V6: pitch backup assets. A 40 s compare clip rendered with matplotlib (MP4 if ffmpeg is present, plus GIF)
and key-frame PNGs. A live screen recording of the HTML page has to be made by hand (QuickTime/OBS)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..viz import DRONE_COLOURS, _maybe_mp4, _plt, base_image


def _secured(result, t):
    n = max(result.metrics.get("n_at_risk", 0), 1)
    return sum(f["surv"] for f in result.finds if f["at_risk"] and f["t"] <= t) / n


def make_pitch(runs, out_dir, seconds: int = 40, fps: int = 8, keyframes=(5, 30, 60, 120)) -> Path:
    plt = _plt()
    from PIL import Image
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    world = runs[-1][1]
    L = world.cfg.world.domain_m
    base = base_image(world.terrain)
    frames_all = [r.recording["frames"] for r, _ in runs]
    n_fr = min(len(f) for f in frames_all)
    idx = np.linspace(0, n_fr - 1, seconds * fps).round().astype(int)
    labels = ["Standard Bayesian search" if r.controller == "bayes" else
              "Survival-aware (ours)" if r.controller == "survival" else r.controller for r, _ in runs]
    ep = world.cfg.world.episode_s
    curves = [[_secured(r, f["t"]) for f in frames_all[k][:n_fr]] for k, (r, _) in enumerate(runs)]
    ts = [f["t"] / 60 for f in frames_all[0][:n_fr]]
    fig = plt.figure(figsize=(11, 7.2), dpi=72)
    imgs = []

    def draw(j):
        fig.clf()
        t = frames_all[0][j]["t"]
        for k, (r, w) in enumerate(runs):
            ax = fig.add_axes([0.02 + 0.49 * k, 0.33, 0.46, 0.6])
            ax.imshow(base, origin="lower", extent=(0, L, 0, L))
            d = frames_all[k][j]["depth"].astype(np.float32)
            rgba = np.zeros(d.shape + (4,)); rgba[..., 1] = 0.42; rgba[..., 2] = 0.78
            rgba[..., 3] = np.where(d > 0.1, np.clip(0.25 + d / 3, 0, 0.85), 0)
            ax.imshow(rgba, origin="lower", extent=(0, L, 0, L))
            dbg = [x for x in r.recording.get("debug", []) if x and abs(x["t"] - t) < 1]
            if dbg and "stake" in dbg[0]:
                st = np.asarray(dbg[0]["stake"], float)
                mx = max(st.max(), 1e-9)
                h = np.zeros(st.shape + (4,)); h[..., 0] = 0.79; h[..., 1] = 0.47; h[..., 2] = 0.1
                h[..., 3] = np.clip(np.sqrt(st / mx) * 0.8, 0, 0.8) * (st > 0.02 * mx)
                ax.imshow(h, origin="lower", extent=(0, L, 0, L))
            for i, dr in enumerate(frames_all[k][j]["drones"]):
                if dr[4] not in ("ground", "down"):
                    ax.plot(dr[0], dr[1], "^", color=DRONE_COLOURS[i % 8], ms=7, mec="k")
            for fnd in r.finds:
                if fnd["t"] <= t:
                    P = frames_all[k][j]["people"]
                    ax.plot(P[fnd["person"], 0], P[fnd["person"], 1], "o", color="#23845a", ms=6)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, L); ax.set_ylim(0, L)
            ax.set_title(f"{labels[k]}   survival secured {curves[k][j]:.2f}", fontsize=11)
        ax = fig.add_axes([0.07, 0.06, 0.88, 0.2])
        cols = ["#5a6f8f", "#d9861a"]
        for k in range(len(runs)):
            ax.plot(ts, curves[k], color=cols[k % 2], lw=2.5 if k else 1.8, label=labels[k])
        ax.axvline(t / 60, color="k", lw=1)
        ax.set_xlim(0, ep / 60)
        ax.set_xlabel("minutes after launch"); ax.set_ylabel("survival secured")
        ax.legend(frameon=False, loc="upper left", fontsize=9)
        fig.suptitle(f"Same flood, same drones, same sensors · t = {t / 60:.0f} min (amber: lives at stake, as estimated by our planner; "
                     f"blue: flood, truth)", fontsize=10)
        fig.canvas.draw()
        return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    for j in idx:
        imgs.append(draw(int(j)))
    imgs[0].save(out / "compare_clip.gif", save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0,
                 optimize=True)
    _maybe_mp4(imgs, out / "compare_clip.mp4", fps)
    for m in keyframes:
        j = min(int(m * 60 / 30), n_fr - 1)
        draw(j).save(out / f"keyframe_{m:03d}min.png")
    plt.close(fig)
    return out
