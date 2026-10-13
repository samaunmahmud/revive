"""Score ReVive against known clean originals.

For each clean image x damage profile:
  damaged  -> no restoration (lower bound)
  fixed    -> every tool applied once, default settings, no verification (naive pipeline)
  revive   -> the agent
Metrics vs the clean original: PSNR (dB, higher better) and SSIM (0-1, higher better).

Usage: python -m eval.evaluate [--out eval/results]
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revive import agent, damage, perception, tools  # noqa: E402


def test_images():
    """Public-domain scikit-image samples; greyscale ones stand in for old B&W prints."""
    imgs = {
        "astronaut": data.astronaut(), "coffee": data.coffee(), "chelsea": data.chelsea(),
        "rocket": data.rocket(), "camera_bw": data.camera(),
    }
    out = {}
    for name, im in imgs.items():
        if im.ndim == 2:
            im = np.stack([im] * 3, axis=-1)
        out[name] = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return out


def fixed_pipeline(img):
    m = perception.scratch_mask(img)
    img = tools.inpaint_scratches(img, m, 3)
    img = tools.denoise(img, 10)
    img = tools.correct_colour(img, 1.0)
    img = tools.enhance_contrast(img, 2.0)
    return tools.sharpen(img, 0.6)


def score(clean, other):
    # identical images give infinite PSNR; cap at 60 dB (visually lossless) so averages stay finite
    p = 60.0 if np.array_equal(clean, other) else min(60.0, psnr(clean, other, data_range=255))
    return (round(p, 2),
            round(ssim(clean, other, channel_axis=2, data_range=255), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    rows, traces = [], {}
    for name, clean in test_images().items():
        for prof in damage.PROFILES:
            dmg = damage.age(clean, prof, seed=42)
            t0 = time.perf_counter()
            fixed = fixed_pipeline(dmg)
            t_fixed = time.perf_counter() - t0
            t0 = time.perf_counter()
            restored, trace = agent.restore(dmg, log=lambda *_: None)
            t_agent = time.perf_counter() - t0

            # simulated human who answers truthfully: approves repair masks, and says
            # "ageing" to the colour question only if this profile really added a tint
            truth = {"inpaint_scratches": True, "correct_colour": damage.PROFILES[prof]["cast"] > 0}
            hitl, hitl_trace = agent.restore(dmg, log=lambda *_: None, answers=truth)

            d, f, r, h = score(clean, dmg), score(clean, fixed), score(clean, restored), score(clean, hitl)
            rows.append({"image": name, "damage": prof,
                         "psnr_damaged": d[0], "psnr_fixed": f[0], "psnr_revive": r[0],
                         "psnr_revive_hitl": h[0],
                         "ssim_damaged": d[1], "ssim_fixed": f[1], "ssim_revive": r[1], "ssim_revive_hitl": h[1],
                         "status": trace["status"], "steps_kept": sum(s["kept"] for s in trace["steps"]),
                         "steps_rejected": sum(not s["kept"] for s in trace["steps"]),
                         "sec_fixed": round(t_fixed, 2), "sec_revive": round(t_agent, 2)})
            traces[f"{name}/{prof}"] = trace
            traces[f"{name}/{prof}/with_human"] = hitl_trace
            cv2.imwrite(os.path.join(args.out, "images", f"{name}_{prof}.jpg"),
                        np.hstack([dmg, restored, clean]))
            print(f"{name:10s} {prof:9s} SSIM {d[1]:.3f} → fixed {f[1]:.3f} / revive {r[1]:.3f}  "
                  f"[{trace['status']}]  with human {h[1]:.3f}")

    with open(os.path.join(args.out, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out, "traces.json"), "w") as fh:
        json.dump(traces, fh, indent=2)

    auto = [r for r in rows if r["status"] != "needs_human_review"]
    print("\n=== summary ===")
    for label, subset in (("all", rows), ("auto-completed", auto)):
        if not subset:
            continue
        mean = lambda k: np.mean([r[k] for r in subset])
        print(f"{label:15s} n={len(subset):2d}  SSIM damaged {mean('ssim_damaged'):.3f} | "
              f"fixed {mean('ssim_fixed'):.3f} | revive {mean('ssim_revive'):.3f}   "
              f"PSNR damaged {mean('psnr_damaged'):.1f} | fixed {mean('psnr_fixed'):.1f} | "
              f"revive {mean('psnr_revive'):.1f}")
        print(f"{'':15s}       with human in the loop: SSIM {mean('ssim_revive_hitl'):.3f}  PSNR {mean('psnr_revive_hitl'):.1f}")
    esc = len(rows) - len(auto)
    print(f"escalated to human: {esc}/{len(rows)}")


if __name__ == "__main__":
    main()
