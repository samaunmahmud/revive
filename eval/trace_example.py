"""The annotated trace for the report: OpenCV re-measurement changes what gets repaired.

    python -m eval.trace_example     # writes docs/figures/trace-astronaut-aged.png + prints the numbers

Uses the bundled "Faded portrait" sample (astronaut, 'aged' damage). Because the
damage is synthetic and seeded, the exact positions of the real scratches and
dust are known: the same ageing is re-run with marks switched off and the
difference is the ground truth. That lets us check that the scratches "revealed"
by the tone step are real marks, not image detail.
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revive import agent, damage, perception  # noqa: E402
from eval.evaluate import test_images  # noqa: E402

SAMPLE = os.path.join("server", "samples", "astronaut-aged.png")
OUT = os.path.join("docs", "figures", "trace-astronaut-aged.png")


def true_marks(clean):
    p = damage.PROFILES["aged"]
    saved = dict(p)
    p.update(scratches=0, dust=0)
    no_marks = damage.age(clean, "aged", seed=42)
    p.update(saved)
    with_marks = damage.age(clean, "aged", seed=42)
    return (cv2.absdiff(with_marks, no_marks).max(axis=2) > 20).astype(np.uint8) * 255


def mask_quality(mask, truth):
    near = lambda m: cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0
    m, t = mask > 0, truth > 0
    return {"coverage_pct": round(float(m.mean() * 100), 2),
            "precision_pct": round(float((m & near(truth)).sum() / max(m.sum(), 1) * 100), 1),
            "recall_pct": round(float((t & near(mask)).sum() / max(t.sum(), 1) * 100), 1)}


def main():
    clean = test_images()["astronaut"]
    img = cv2.imread(SAMPLE)
    truth = true_marks(clean)

    frames = {}
    def on(kind, payload, image=None, mask=None):
        frames[kind if kind == "start" else payload.get("step")] = (None if image is None else image.copy(), mask, payload)

    _, trace = agent.restore(img, log=lambda *_: None, on_event=on)
    tone = next(s for s in trace["steps"] if s["tool"] == "enhance_contrast")
    after_tone = frames[tone["step"]][0]
    start_mask = frames["start"][1]
    tone_mask = perception.analyse(after_tone)["_mask"]

    numbers = {"steps": [{k: s[k] for k in ("step", "tool", "params", "reason", "structure", "checks", "revealed", "verdict")}
                         for s in trace["steps"]],
               "mask_at_start": mask_quality(start_mask, truth),
               "mask_after_tone_step": mask_quality(tone_mask, truth),
               "score": [trace["initial_score"], trace["final_score"]], "status": trace["status"]}
    print(json.dumps(numbers, indent=2))

    def overlay(im, m):
        o = im.copy()
        o[cv2.dilate(truth, np.ones((2, 2), np.uint8)) > 0] = (60, 200, 60)
        o[cv2.dilate(m, np.ones((2, 2), np.uint8)) > 0] = (40, 60, 230)
        return o

    q0, q1 = numbers["mask_at_start"], numbers["mask_after_tone_step"]
    final = frames[max(k for k in frames if k != "start")][0]
    panels = [(img, "1. input: tinted, faded, grainy, scratched"),
              (overlay(img, start_mask), f"2. damage found at start: {q0['recall_pct']:.0f}% of true marks"),
              (overlay(after_tone, tone_mask), f"3. after tone step: {q1['recall_pct']:.0f}% of true marks"),
              (final, "4. after inpainting + denoise")]
    h, tiles = 300, []
    for im, cap in panels:
        s = h / im.shape[0]
        im = cv2.resize(im, (round(im.shape[1] * s), h), interpolation=cv2.INTER_AREA)
        bar = np.full((44, im.shape[1], 3), 252, np.uint8)
        cv2.putText(bar, cap, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 25, 20), 1, cv2.LINE_AA)
        tiles += [np.vstack([im, bar]), np.full((h + 44, 12, 3), 252, np.uint8)]
    legend = np.full((30, sum(t.shape[1] for t in tiles[:-1]), 3), 252, np.uint8)
    cv2.putText(legend, "red = mask ReVive measured with OpenCV 5     green = true scratch/dust positions (known because the damage is synthetic)",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 75, 70), 1, cv2.LINE_AA)
    cv2.imwrite(OUT, np.vstack([np.hstack(tiles[:-1]), legend]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
