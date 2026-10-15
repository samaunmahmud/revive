"""Human mask corrections: how much does a person's brush add to the agent?

    python -m eval.human_edits        # -> eval/results/human_edits.json, docs/figures/human-edit-tovey.png

Three checks, all with the same agent and verifier as the main evaluation:

1. Missed damage (synthetic, scored against the clean original). Each of the 5
   test photos gets the "scratched" profile plus one 6 px wide scratch, wider
   than the detector's 2.5 px limit, so the agent cannot see it. A person paints
   it with a 9 px stroke. We measure the error vs the clean photo inside that
   scratch, automatic vs painted.
2. Protected detail (synthetic, simulated person). On the "scratched" profile we
   know where the real marks are (damaged - clean). Parts of the agent's repair
   mask far from any real mark are false detections; a simulated person protects
   them. We measure the error vs clean on those pixels. This is an upper bound:
   a real person protects only what they notice.
3. A real scan (tovey-1942: one wide, curved white scratch the agent misses).
   The stroke follows the scratch the way a brush would (brightest pixel per row
   inside a band drawn around it). No clean original exists, so this one is a
   figure plus the verifier's own before/after anomaly.
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revive import agent, damage, perception  # noqa: E402
from eval.evaluate import score, test_images  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
quiet = lambda *_: None  # noqa: E731


def err(img, clean, region):
    """Mean absolute error vs the clean photo inside a region (0-255 scale)."""
    return float(np.abs(img.astype(np.float32) - clean.astype(np.float32))[region > 0].mean())


def inpaint_step(trace):
    return next((s for s in trace["steps"] if s["tool"] == "inpaint_scratches" and s["kept"]), None)


def missed_damage(images):
    rows = []
    for name, clean in images.items():
        h, w = clean.shape[:2]
        dmg = damage.age(clean, "scratched", seed=42)
        p1, p2 = (int(w * 0.62), int(h * 0.08)), (int(w * 0.66), int(h * 0.92))
        cv2.line(dmg, p1, p2, (235, 235, 235), 6)
        scratch, stroke = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
        cv2.line(scratch, p1, p2, 255, 6)
        cv2.line(stroke, p1, p2, 255, 9)
        found = float((perception.analyse(dmg)["_mask"][scratch > 0] > 0).mean())
        auto, _ = agent.restore(dmg, log=quiet)
        painted, trace = agent.restore(dmg, log=quiet, mask_edits={"add": stroke})
        step = inpaint_step(trace)
        rows.append({"image": name, "detector_found_pct": round(found * 100, 1),
                     "error_damaged": round(err(dmg, clean, scratch), 1),
                     "error_auto": round(err(auto, clean, scratch), 1),
                     "error_painted": round(err(painted, clean, scratch), 1),
                     "ssim_auto": score(clean, auto)[1], "ssim_painted": score(clean, painted)[1],
                     "painted_anomaly": step["checks"].get("painted_anomaly") if step else None,
                     "kept": bool(step)})
    return rows


def protected_detail(images):
    rows = []
    for name, clean in images.items():
        dmg = damage.age(clean, "scratched", seed=42)
        truth = (np.abs(dmg.astype(np.int16) - clean.astype(np.int16)).max(axis=2) > 25).astype(np.uint8) * 255
        near_truth = cv2.dilate(truth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        mask = perception.analyse(dmg)["_mask"]
        false = cv2.bitwise_and(mask, cv2.bitwise_not(near_truth))
        auto, t_auto = agent.restore(dmg, log=quiet)
        row = {"image": name, "false_detection_pct": round(float((false > 0).mean()) * 100, 3),
               "auto_inpainted": bool(inpaint_step(t_auto))}
        if false.any() and row["auto_inpainted"]:
            protect = cv2.dilate(false, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            protected, _ = agent.restore(dmg, log=quiet, mask_edits={"erase": protect})
            row.update({"error_auto": round(err(auto, clean, false), 1),
                        "error_protected": round(err(protected, clean, false), 1),
                        "ssim_auto": score(clean, auto)[1], "ssim_protected": score(clean, protected)[1]})
        rows.append(row)
    return rows


def real_scan(out_png):
    im = cv2.imread(os.path.join(ROOT, "eval", "real_photos", "tovey-1942.jpg"))
    h, w = im.shape[:2]
    x0, x1, y0, y1 = 540, 600, 88, 385          # band drawn around the scratch
    band = im[y0:y1, x0:x1].min(axis=2)
    pts = [(x0 + int(band[r].argmax()), y0 + r) for r in range(0, y1 - y0, 6) if band[r].max() > 200]
    stroke = np.zeros((h, w), np.uint8)
    cv2.polylines(stroke, [np.array(pts, np.int32)], False, 255, 9)
    found = float((perception.analyse(im)["_mask"][stroke > 0] > 0).mean())
    auto, t_auto = agent.restore(im, log=quiet)
    painted, trace = agent.restore(im, log=quiet, mask_edits={"add": stroke})
    step = inpaint_step(trace)

    # figure: input · the painted stroke · automatic · with the person's stroke (crop around the scratch)
    shown = im.copy()
    shown[stroke > 0] = (0.45 * shown[stroke > 0] + 0.55 * np.array([102, 209, 255])).astype(np.uint8)
    crop = lambda a: cv2.resize(a[60:420, 470:680], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # noqa: E731
    panels = [crop(a) for a in (im, shown, auto, painted)]
    labels = ["input (1942 scan)", "a person paints it", "automatic: missed", "with the painted stroke"]
    for p, text in zip(panels, labels):
        cv2.rectangle(p, (0, p.shape[0] - 34), (p.shape[1], p.shape[0]), (24, 20, 16), -1)
        cv2.putText(p, text, (10, p.shape[0] - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 228, 214), 1, cv2.LINE_AA)
    gap = np.full((panels[0].shape[0], 8, 3), 255, np.uint8)
    cv2.imwrite(out_png, np.hstack([x for p in panels for x in (p, gap)][:-1]))
    return {"photo": "tovey-1942", "detector_found_pct": round(found * 100, 1),
            "auto_steps": [s["tool"] for s in t_auto["steps"] if s["kept"]],
            "painted_steps": [s["tool"] for s in trace["steps"] if s["kept"]],
            "painted_anomaly": step["checks"].get("painted_anomaly") if step else None,
            "structure": step["structure"] if step else None, "figure": os.path.relpath(out_png, ROOT)}


def main():
    images = test_images()
    result = {"missed_damage": missed_damage(images), "protected_detail": protected_detail(images),
              "real_scan": real_scan(os.path.join(ROOT, "docs", "figures", "human-edit-tovey.png"))}
    os.makedirs(os.path.join(ROOT, "eval", "results"), exist_ok=True)
    with open(os.path.join(ROOT, "eval", "results", "human_edits.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    md = result["missed_damage"]
    print("1. missed wide scratch (error vs clean inside the scratch, 0-255)")
    print("| image | detector found | damaged | automatic | painted | SSIM auto → painted |\n|---|---|---|---|---|---|")
    for r in md:
        print(f"| {r['image']} | {r['detector_found_pct']}% | {r['error_damaged']} | {r['error_auto']} | "
              f"{r['error_painted']} | {r['ssim_auto']} → {r['ssim_painted']} |")
    m = lambda k, rows: round(float(np.mean([r[k] for r in rows])), 3)  # noqa: E731
    print(f"| **mean** | | {m('error_damaged', md)} | {m('error_auto', md)} | {m('error_painted', md)} | "
          f"{m('ssim_auto', md)} → {m('ssim_painted', md)} |")
    print(f"painted repairs kept by the verifier: {sum(r['kept'] for r in md)}/{len(md)}")

    pd = [r for r in result["protected_detail"] if "error_protected" in r]
    print("\n2. protecting false detections (error vs clean on falsely detected pixels)")
    print("| image | false detections | automatic | protected | SSIM auto → protected |\n|---|---|---|---|---|")
    for r in result["protected_detail"]:
        if "error_protected" in r:
            print(f"| {r['image']} | {r['false_detection_pct']}% | {r['error_auto']} | {r['error_protected']} | "
                  f"{r['ssim_auto']} → {r['ssim_protected']} |")
        else:
            print(f"| {r['image']} | {r['false_detection_pct']}% | (nothing to protect) | | |")

    print("\n3. real scan:", json.dumps(result["real_scan"]))


if __name__ == "__main__":
    main()
