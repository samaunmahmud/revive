"""Qualitative check on real, public-domain damaged photos (no clean original exists).

    python -m eval.real_photos     # docs/figures/real-photos.png + a summary table

For each photo in eval/real_photos (sources and licences in credits.json):
input | full repair mask (yellow) with confident marks (red) | automatic result,
plus what the agent decided and what it asked. There is no ground truth, so
these are judged by eye; the summary prints the measurements behind each decision.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revive import agent, perception  # noqa: E402
from server.jobs import prepare  # noqa: E402

HERE = os.path.join("eval", "real_photos")
OUT = os.path.join("docs", "figures", "real-photos.png")


def main():
    credits = {c["file"]: c for c in json.load(open(os.path.join(HERE, "credits.json")))}
    rows, summary = [], []
    for path in sorted(glob.glob(os.path.join(HERE, "*.jpg"))):
        name = os.path.basename(path)
        img = prepare(cv2.imread(path))
        report = perception.analyse(img)
        confident = perception.confident_marks(img, report["_mask"])
        restored, trace = agent.restore(img, log=lambda *_: None)

        overlay = img.copy()
        overlay[cv2.dilate(report["_mask"], np.ones((3, 3), np.uint8)) > 0] = (0, 200, 255)
        overlay[cv2.dilate(confident, np.ones((3, 3), np.uint8)) > 0] = (40, 40, 230)
        h = 300
        fit = lambda x: cv2.resize(x, (round(x.shape[1] * h / x.shape[0]), h), interpolation=cv2.INTER_AREA)
        gap = np.full((h, 8, 3), 252, np.uint8)
        row = np.hstack([fit(img), gap, fit(overlay), gap, fit(restored)])
        label = f"{name[:-4]}  ({credits.get(name, {}).get('date', '')})"
        bar = np.full((34, row.shape[1], 3), 252, np.uint8)
        did = ", ".join(s["tool"] for s in trace["steps"] if s["kept"]) or "no repairs"
        asked = ", ".join(q["about"] for q in trace["questions"] if q["about"] != "residual_damage")
        cv2.putText(bar, f"{label}: {did}" + (f"; asked about {asked}" if asked else ""),
                    (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 25, 20), 1, cv2.LINE_AA)
        rows.append(np.vstack([bar, row]))
        summary.append({"photo": name, "measured": perception.public(report),
                        "kept": [s["tool"] for s in trace["steps"] if s["kept"]],
                        "rejected": [s["tool"] for s in trace["steps"] if not s["kept"]],
                        "questions": [q["question"] for q in trace["questions"]], "status": trace["status"]})

    width = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.hstack([r, np.full((r.shape[0], width - r.shape[1], 3), 252, np.uint8)]) for r in rows])
    cv2.imwrite(OUT, sheet, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(json.dumps(summary, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
