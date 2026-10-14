"""Reproducible OpenCV benchmark of ReVive's real workload: stock OpenCV 5 vs COOL.

Run the SAME script with each OpenCV build on the same machine, then compare:

    python -m bench.benchmark run --label stock --price 0.0918     # pip opencv-python-headless
    python -m bench.benchmark run --label cool  --price 0.1018     # COOL AMI venv (instance + software $/h)
    python -m bench.benchmark compare bench/results/*.json > bench/results/RESULTS.md

What is measured (all on the bundled sample photos, at native ~512 px and at a
1600 px "phone upload" size):
  - per-step latency (median / p95 ms): perception.analyse and each repair tool
  - full agent restore latency (heuristic planner, approvals given, so the path is fixed)
  - throughput: full restores per second using all cores (one restore per thread)
  - cost: $ per 1,000 restores from --price (instance + software, $/hour)
Environment details (OpenCV version, file, KleidiCV/HAL, CPU, instance type) are recorded.
"""
import argparse
import glob
import json
import os
import platform
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revive import agent, perception, tools  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "server", "samples")
APPROVE_ALL = {"inpaint_scratches": True, "correct_colour": True}


# ---------- environment ----------

def instance_type() -> str:
    """EC2 instance type via IMDSv2, or 'local' off EC2."""
    try:
        tok = urllib.request.urlopen(urllib.request.Request(
            "http://169.254.169.254/latest/api/token", method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}), timeout=0.3).read().decode()
        return urllib.request.urlopen(urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-type",
            headers={"X-aws-ec2-metadata-token": tok}), timeout=0.3).read().decode()
    except Exception:
        return "local"


def environment(label: str, price: float | None) -> dict:
    info = cv2.getBuildInformation()
    hal = [ln.strip() for ln in info.splitlines() if "HAL" in ln or "Baseline" in ln or "Dispatched" in ln]
    return {
        "label": label, "instance_type": instance_type(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(), "opencv_version": cv2.__version__, "opencv_file": cv2.__file__,
        "opencv_threads": cv2.getNumThreads(), "kleidicv": "kleidicv" in info.lower(),
        "build_lines": hal, "numpy": np.__version__, "python": platform.python_version(),
        "price_per_hour": price, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------- workload ----------

def images(size: int | None):
    out = {}
    for path in sorted(glob.glob(os.path.join(SAMPLES, "*.png"))):
        img = cv2.imread(path)
        if size:
            h, w = img.shape[:2]
            s = size / max(h, w)
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_CUBIC)
        out[os.path.splitext(os.path.basename(path))[0]] = img
    return out


def steps(img):
    """The operations the agent actually runs, with the agent's typical settings."""
    report = perception.analyse(img)
    mask = report["_mask"]
    return {
        "perception.analyse": lambda: perception.analyse(img),
        "correct_colour": lambda: tools.correct_colour(img, 0.9),
        "enhance_contrast": lambda: tools.enhance_contrast(img, 0.0),
        "inpaint_scratches": lambda: tools.inpaint_scratches(img, mask, 3, "telea"),
        "denoise": lambda: tools.denoise(img, 10.0),
        "sharpen": lambda: tools.sharpen(img, 0.6),
        "agent.restore (full)": lambda: agent.restore(img, answers=APPROVE_ALL, log=lambda *_: None),
    }


def timed(fn, repeats: int) -> list[float]:
    fn()  # warm-up (allocations, thread pools, caches)
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(0.95 * (len(xs) - 1)))]


def run(args):
    env = environment(args.label, args.price)
    print(f"# {args.label}: OpenCV {env['opencv_version']} ({env['opencv_file']}) · "
          f"{env['instance_type']} · {env['cpu_count']} vCPU · KleidiCV={env['kleidicv']}", file=sys.stderr)
    result = {"env": env, "latency": {}, "throughput": {}}

    for size_label, size in (("native", None), ("1600px", 1600)):
        imgs = images(size)
        per_step: dict[str, list[float]] = {}
        for name, img in imgs.items():
            for step, fn in steps(img).items():
                per_step.setdefault(step, []).extend(timed(fn, args.repeats))
            print(f"  {size_label} {name} done", file=sys.stderr)
        result["latency"][size_label] = {
            s: {"median_ms": round(statistics.median(v), 2), "p95_ms": round(p95(v), 2), "n": len(v)}
            for s, v in per_step.items()}

        # throughput: many restores at once, one per thread (how the server runs under load)
        work = list(imgs.values()) * args.rounds
        workers = os.cpu_count() or 1
        restore = lambda im: agent.restore(im, answers=APPROVE_ALL, log=lambda *_: None)
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(restore, list(imgs.values())))  # warm-up
            t0 = time.perf_counter()
            list(ex.map(restore, work))
            secs = time.perf_counter() - t0
        rps = len(work) / secs
        tp = {"restores": len(work), "workers": workers, "seconds": round(secs, 2),
              "restores_per_sec": round(rps, 3)}
        if args.price:
            tp["usd_per_1000_restores"] = round(args.price / 3600 / rps * 1000, 4)
        result["throughput"][size_label] = tp
        print(f"  {size_label} throughput {rps:.2f} restores/s", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.label}-{env['instance_type']}.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(path)


# ---------- comparison table ----------

def compare(args):
    runs = [json.load(open(p)) for p in args.files]
    base = next((r for r in runs if r["env"]["label"] == args.baseline), runs[0])
    print("# ReVive OpenCV benchmark\n")
    print("| run | instance | vCPU | OpenCV | KleidiCV | $/hour |")
    print("|---|---|---|---|---|---|")
    for r in runs:
        e = r["env"]
        print(f"| {e['label']} | {e['instance_type']} ({e['machine']}) | {e['cpu_count']} | "
              f"{e['opencv_version']} | {'yes' if e['kleidicv'] else 'no'} | {e['price_per_hour'] or '-'} |")
    for size in base["latency"]:
        print(f"\n## Latency, {size} (median ms; speed-up vs {base['env']['label']})\n")
        print("| step | " + " | ".join(r["env"]["label"] for r in runs) + " |")
        print("|---|" + "---|" * len(runs))
        for step, b in base["latency"][size].items():
            cells = []
            for r in runs:
                m = r["latency"][size][step]["median_ms"]
                cells.append(f"{m:.1f}" + ("" if r is base else f" ({b['median_ms'] / m:.2f}×)"))
            print(f"| {step} | " + " | ".join(cells) + " |")
        print(f"\n**Throughput, {size}:** " + "; ".join(
            f"{r['env']['label']} {r['throughput'][size]['restores_per_sec']} restores/s"
            + (f" (${r['throughput'][size]['usd_per_1000_restores']} per 1,000)"
               if "usd_per_1000_restores" in r["throughput"][size] else "")
            for r in runs))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True, help="e.g. stock, cool, x86-stock")
    r.add_argument("--price", type=float, help="$/hour for instance + software, for cost per restore")
    r.add_argument("--repeats", type=int, default=5)
    r.add_argument("--rounds", type=int, default=4, help="throughput: passes over the sample set")
    r.add_argument("--out", default=os.path.join(ROOT, "bench", "results"))
    c = sub.add_parser("compare")
    c.add_argument("files", nargs="+")
    c.add_argument("--baseline", default="stock")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    main()
