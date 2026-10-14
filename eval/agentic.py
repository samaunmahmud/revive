"""Agentic evaluation: task success, human control, failure handling, observability.

Three experiments on the same 35 cases as eval/evaluate.py (5 images x 7 profiles):

1. Baseline agent (heuristic planner): task success, escalation rate,
   kept/rejected steps, retries, "revealed damage" events, per-tool latency.
2. Verifier ablation: a NoisyPlanner makes seeded over-strong choices on a share
   of its steps (the kind of mistake an LLM planner can make). The same runs
   with the verifier ON vs OFF show how much damage verification prevents.
3. Planner failure: the real BedrockPlanner talks to a FlakyBedrockClient that
   returns broken JSON, unknown or blocked tools, or times out. Every run must
   still finish, through the heuristic fallback, and the trace must say so.

Usage: python -m eval.agentic [--out eval/results] [--mistake-rate 0.5]
"""
import argparse
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revive import agent, damage  # noqa: E402
from eval.evaluate import score, test_images  # noqa: E402

TRUTH = lambda prof: {"inpaint_scratches": True, "correct_colour": damage.PROFILES[prof]["cast"] > 0}


# ---------- 2. a planner that sometimes makes over-strong choices ----------

OVERSTRONG = {
    "denoise": lambda p: {"strength": round(min(30.0, p.get("strength", 8) * 3), 1)},
    "enhance_contrast": lambda p: {"clip": 4.0},
    "inpaint_scratches": lambda p: {"radius": 9, "method": "ns"},
    "correct_colour": lambda p: {"strength": 1.0},
}


class NoisyPlanner:
    """Heuristic decisions, but a seeded share of first attempts use over-strong settings."""

    name = "noisy"

    def __init__(self, rate: float, seed: int):
        self.inner = agent.HeuristicPlanner()
        self.rate = rate
        self.rng = random.Random(seed)
        self.mistakes = 0

    def next_action(self, r, history, blocked):
        a = self.inner.next_action(r, history, blocked)
        tool = a.get("tool")
        first_try = not any(h["tool"] == tool for h in history)
        if tool in OVERSTRONG and first_try and self.rng.random() < self.rate:
            a = {**a, "params": OVERSTRONG[tool](a.get("params", {})),
                 "reason": a["reason"] + " [injected over-strong settings]"}
            self.mistakes += 1
        return a


# ---------- 3. a Bedrock client that fails in every way it can ----------

class FlakyBedrockClient:
    """Stands in for boto3's bedrock-runtime client. Cycles through failure modes."""

    MODES = ["broken_json", "unknown_tool", "blocked_tool", "timeout", "prose_no_json"]

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.calls = Counter()

    def converse(self, **kwargs):
        mode = self.rng.choice(self.MODES)
        self.calls[mode] += 1
        text = {
            "broken_json": '{"tool": "denoise", "params": {"strength": ',
            "unknown_tool": '{"tool": "make_it_pretty", "params": {}, "reason": "?"}',
            "blocked_tool": None,  # filled in below from the prompt
            "prose_no_json": "I think the photo looks fine overall.",
        }.get(mode)
        if mode == "timeout":
            raise TimeoutError("simulated Bedrock timeout")
        if mode == "blocked_tool":
            prompt = kwargs["messages"][0]["content"][0]["text"]
            blocked = prompt.split("waiting for a human): ")[1].split("\n")[0]
            tool = next((t for t in agent.TOOLS if t in blocked), "denoise")
            text = json.dumps({"tool": tool, "params": {}, "reason": "ignores the blocked list"})
            if tool not in blocked:  # nothing blocked yet: fall back to broken JSON
                text = '{"tool": '
        return {"output": {"message": {"content": [{"text": text}]}}}


def flaky_bedrock_planner(seed):
    p = agent.BedrockPlanner.__new__(agent.BedrockPlanner)  # skip boto3 client creation
    p.model_id, p.client, p.fallback = "simulated", FlakyBedrockClient(seed), agent.HeuristicPlanner()
    return p


# ---------- helpers ----------

def summarise(trace, clean, restored, damaged):
    steps = trace["steps"]
    ssim_r, ssim_d = score(clean, restored)[1], score(clean, damaged)[1]
    return {
        "status": trace["status"],
        "escalated": trace["status"] == "needs_human_review",
        "questions": [q["about"] for q in trace["questions"]],
        "kept": sum(s["kept"] for s in steps),
        "rejected": sum(not s["kept"] for s in steps),
        "gave_up": sum("giving up" in s["verdict"] for s in steps),
        "revealed": sum(bool(s["revealed"]) for s in steps),
        "fallbacks": sum(s["reason"].startswith("[fallback") for s in steps),
        "ssim": float(ssim_r), "ssim_damaged": float(ssim_d),
        "no_harm": bool(ssim_r >= ssim_d - 0.005),
        "score_before": trace["initial_score"], "score_after": trace["final_score"],
        "ms": {s["tool"]: s["ms"] for s in steps},
    }


def mean(xs):
    return round(float(np.mean(xs)), 4) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--mistake-rate", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    quiet = lambda *_: None

    cases = [(name, prof, clean, damage.age(clean, prof, seed=42))
             for name, clean in test_images().items() for prof in damage.PROFILES]

    base, human, on, off, flaky = [], [], [], [], []
    latency = defaultdict(list)
    noisy_mistakes, flaky_calls = 0, Counter()
    example_trace = None

    for i, (name, prof, clean, dmg) in enumerate(cases):
        # 1. baseline agent, and the same case with a (simulated, truthful) human
        r, t = agent.restore(dmg, log=quiet)
        row = {"case": f"{name}/{prof}", "profile": prof, **summarise(t, clean, r, dmg)}
        base.append(row)
        for s in t["steps"]:
            latency[s["tool"]].append(s["ms"])
        rh, th = agent.restore(dmg, log=quiet, answers=TRUTH(prof))
        human.append({"case": row["case"], "profile": prof, **summarise(th, clean, rh, dmg)})

        # 2. verifier ablation: identical seeded mistakes, verifier on vs off
        p_on, p_off = NoisyPlanner(args.mistake_rate, seed=i), NoisyPlanner(args.mistake_rate, seed=i)
        r_on, t_on = agent.restore(dmg, planner=p_on, log=quiet, answers=TRUTH(prof))
        r_off, t_off = agent.restore(dmg, planner=p_off, log=quiet, answers=TRUTH(prof), verify=False)
        noisy_mistakes += p_on.mistakes
        on.append({"case": row["case"], "mistakes": p_on.mistakes, **summarise(t_on, clean, r_on, dmg)})
        off.append({"case": row["case"], "mistakes": p_off.mistakes, **summarise(t_off, clean, r_off, dmg)})
        if example_trace is None and any(not s["kept"] for s in t_on["steps"]) and \
                any(s["kept"] for s in t_on["steps"] if "gentler" in s["reason"]):
            example_trace = {"case": row["case"], "trace": t_on}

        # 3. planner failure: real BedrockPlanner, every model call fails somehow
        pf = flaky_bedrock_planner(seed=i)
        r_f, t_f = agent.restore(dmg, planner=pf, log=quiet)
        flaky_calls.update(pf.client.calls)
        same = np.array_equal(r_f, r)
        flaky.append({"case": row["case"], "identical_to_heuristic": bool(same), **summarise(t_f, clean, r_f, dmg)})
        print(f"{row['case']:22s} {row['status']:18s} kept {row['kept']} rej {row['rejected']} | "
              f"noisy: {p_on.mistakes} mistakes, SSIM verify-on {on[-1]['ssim']:.3f} vs off {off[-1]['ssim']:.3f} | "
              f"flaky bedrock: {'same result' if same else 'DIFFERENT'}")

    n = len(cases)
    control = [b for b in base if b["profile"] == "none"]
    summary = {
        "cases": n,
        "baseline": {
            "auto_complete_rate": round(sum(not b["escalated"] for b in base) / n, 3),
            "escalation_rate": round(sum(b["escalated"] for b in base) / n, 3),
            "escalations_by_reason": dict(Counter(q for b in base for q in b["questions"])),
            "no_harm_rate": round(sum(b["no_harm"] for b in base) / n, 3),
            "task_success_rate": round(sum(b["no_harm"] and b["score_after"] <= 1.0 for b in base) / n, 3),
            "steps_kept": sum(b["kept"] for b in base), "steps_rejected": sum(b["rejected"] for b in base),
            "runs_with_revealed_damage": sum(b["revealed"] > 0 for b in base),
            "mean_ssim": mean([b["ssim"] for b in base]),
            "control_mean_ssim": mean([b["ssim"] for b in control]),
            "control_steps_applied": sum(b["kept"] for b in control),
        },
        "with_human": {
            "task_success_rate": round(sum(h["no_harm"] and h["score_after"] <= 1.0 for h in human) / n, 3),
            "mean_ssim": mean([h["ssim"] for h in human]),
        },
        "verifier_ablation": {
            "mistake_rate": args.mistake_rate, "mistakes_injected": noisy_mistakes,
            "verifier_on": {"mean_ssim": mean([o["ssim"] for o in on]),
                            "no_harm_rate": round(sum(o["no_harm"] for o in on) / n, 3),
                            "rejected_steps": sum(o["rejected"] for o in on),
                            "gave_up": sum(o["gave_up"] for o in on)},
            "verifier_off": {"mean_ssim": mean([o["ssim"] for o in off]),
                             "no_harm_rate": round(sum(o["no_harm"] for o in off) / n, 3)},
            "cases_where_verifier_helped": int(sum(a["ssim"] > b["ssim"] + 0.005 for a, b in zip(on, off))),
            "cases_where_verifier_hurt": int(sum(a["ssim"] < b["ssim"] - 0.005 for a, b in zip(on, off))),
        },
        "planner_failure": {
            "simulated_failures": dict(flaky_calls),
            "runs_completed": n,
            "fallback_steps": sum(f["fallbacks"] for f in flaky),
            "identical_to_heuristic": sum(f["identical_to_heuristic"] for f in flaky),
        },
        "latency_ms": {tool: {"median": round(statistics.median(v), 1), "p95": round(float(np.percentile(v, 95)), 1),
                              "n": len(v)} for tool, v in sorted(latency.items())},
    }

    with open(os.path.join(args.out, "agentic.json"), "w") as fh:
        json.dump({"summary": summary, "baseline": base, "with_human": human,
                   "verifier_on": on, "verifier_off": off, "flaky_bedrock": flaky}, fh, indent=2)
    if example_trace:
        with open(os.path.join(args.out, "example_rejection_trace.json"), "w") as fh:
            json.dump(example_trace, fh, indent=2)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
