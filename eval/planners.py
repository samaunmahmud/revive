"""Bedrock LLM planner vs heuristic planner on the same 35 cases.

    python -m eval.planners                          # model from REVIVE_MODEL_ID, region from AWS_REGION
    python -m eval.planners --model <id> --limit 7   # quick check on fewer cases

Both planners see the same OpenCV measurements and go through the same verifier,
so differences come only from the decisions. Recorded per planner: SSIM vs clean,
escalations, kept/rejected steps, steps that fell back to the heuristic, and for
Bedrock the per-call latency and token usage (Converse reports it).
Writes eval/results/planners.json and prints a summary table.
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revive import agent, damage  # noqa: E402
from eval.evaluate import score, test_images  # noqa: E402


class MeteredClient:
    """Wraps the boto3 bedrock-runtime client to record latency and tokens per call."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def converse(self, **kw):
        t0 = time.perf_counter()
        try:
            resp = self.inner.converse(**kw)
        except Exception as e:
            self.calls.append({"ms": (time.perf_counter() - t0) * 1000, "error": type(e).__name__})
            raise
        u = resp.get("usage", {})
        self.calls.append({"ms": (time.perf_counter() - t0) * 1000,
                           "in": u.get("inputTokens", 0), "out": u.get("outputTokens", 0)})
        return resp


def run(planner_factory, cases):
    rows = []
    for name, prof, clean, dmg in cases:
        planner = planner_factory()
        t0 = time.perf_counter()
        restored, trace = agent.restore(dmg, planner=planner, log=lambda *_: None)
        steps = trace["steps"]
        rows.append({
            "case": f"{name}/{prof}", "ssim": float(score(clean, restored)[1]),
            "status": trace["status"], "questions": [q["about"] for q in trace["questions"]],
            "kept": sum(s["kept"] for s in steps), "rejected": sum(not s["kept"] for s in steps),
            "fallback_steps": sum(s["reason"].startswith("[fallback") for s in steps),
            "tools": [s["tool"] for s in steps if s["kept"]], "seconds": time.perf_counter() - t0,
            "calls": getattr(getattr(planner, "client", None), "calls", []),
        })
        print(f"  {rows[-1]['case']:22s} SSIM {rows[-1]['ssim']:.3f} {trace['status']:18s} "
              f"tools {rows[-1]['tools']} fallbacks {rows[-1]['fallback_steps']}")
    return rows


def summary(rows):
    calls = [c for r in rows for c in r["calls"]]
    ok = [c for c in calls if "error" not in c]
    return {
        "mean_ssim": round(statistics.mean(r["ssim"] for r in rows), 4),
        "escalation_rate": round(sum(r["status"] != "complete" for r in rows) / len(rows), 3),
        "steps_kept": sum(r["kept"] for r in rows), "steps_rejected": sum(r["rejected"] for r in rows),
        "fallback_steps": sum(r["fallback_steps"] for r in rows),
        "llm_calls": len(calls), "llm_errors": len(calls) - len(ok),
        "llm_median_ms": round(statistics.median(c["ms"] for c in ok), 1) if ok else None,
        "tokens_in": sum(c["in"] for c in ok), "tokens_out": sum(c["out"] for c in ok),
        "mean_seconds_per_photo": round(statistics.mean(r["seconds"] for r in rows), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("REVIVE_MODEL_ID"))
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-west-2"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases")
    ap.add_argument("--out", default="eval/results")
    args = ap.parse_args()

    cases = [(n, p, c, damage.age(c, p, seed=42)) for n, c in test_images().items() for p in damage.PROFILES]
    cases = cases[: args.limit] if args.limit else cases

    def bedrock():
        p = agent.BedrockPlanner(model_id=args.model, region=args.region)
        p.client = MeteredClient(p.client)
        return p

    print("heuristic planner")
    heuristic = run(agent.HeuristicPlanner, cases)
    print(f"bedrock planner ({args.model or 'default model'}, {args.region})")
    llm = run(bedrock, cases)

    same = sum(a["tools"] == b["tools"] for a, b in zip(heuristic, llm))
    result = {"model": agent.BedrockPlanner(model_id=args.model, region=args.region).model_id, "region": args.region,
              "cases": len(cases), "heuristic": summary(heuristic), "bedrock": summary(llm),
              "same_tool_sequence": same,
              "bedrock_better": sum(b["ssim"] > a["ssim"] + 0.005 for a, b in zip(heuristic, llm)),
              "bedrock_worse": sum(b["ssim"] < a["ssim"] - 0.005 for a, b in zip(heuristic, llm)),
              "rows": {"heuristic": heuristic, "bedrock": llm}}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "planners.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    print(f"\n| | heuristic | bedrock |\n|---|---|---|")
    for k in result["heuristic"]:
        print(f"| {k} | {result['heuristic'][k]} | {result['bedrock'][k]} |")
    print(f"\nsame tool sequence in {same}/{len(cases)} cases; bedrock better in "
          f"{result['bedrock_better']}, worse in {result['bedrock_worse']}")


if __name__ == "__main__":
    main()
