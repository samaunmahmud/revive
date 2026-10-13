"""Restore a photo from the command line.

    python restore.py old_photo.jpg                 # heuristic planner, offline
    python restore.py old_photo.jpg --planner bedrock
    python restore.py old_photo.jpg --approve inpaint_scratches --decline correct_colour

Writes <name>_revived.jpg, <name>_compare.jpg (before | after) and <name>_trace.json.
"""
import argparse
import json
import os

import cv2
import numpy as np

from revive import agent


def main():
    ap = argparse.ArgumentParser(description="ReVive: agentic photo restoration with OpenCV 5")
    ap.add_argument("image")
    ap.add_argument("--planner", choices=["heuristic", "bedrock"], default="heuristic")
    ap.add_argument("--approve", nargs="*", default=[], help="questions to answer 'yes' to")
    ap.add_argument("--decline", nargs="*", default=[], help="questions to answer 'no' to")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read {args.image}")

    planner = agent.BedrockPlanner() if args.planner == "bedrock" else agent.HeuristicPlanner()
    answers = {**{a: True for a in args.approve}, **{d: False for d in args.decline}}
    print(f"ReVive · planner={planner.name} · {img.shape[1]}x{img.shape[0]}")
    restored, trace = agent.restore(img, planner=planner, answers=answers)

    stem = os.path.splitext(args.image)[0]
    cv2.imwrite(f"{stem}_revived.jpg", restored)
    cv2.imwrite(f"{stem}_compare.jpg", np.hstack([img, restored]))
    with open(f"{stem}_trace.json", "w") as fh:
        json.dump(trace, fh, indent=2)

    print(f"\nstatus: {trace['status']}  ·  damage score {trace['initial_score']} → {trace['final_score']}")
    for q in trace["questions"]:
        print(f"  ❓ {q['about']}: {q['question']}")
    print(f"saved {stem}_revived.jpg, {stem}_compare.jpg, {stem}_trace.json")


if __name__ == "__main__":
    main()
