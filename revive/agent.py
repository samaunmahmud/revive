"""The ReVive agent: perceive -> plan -> act -> verify, until the photo is restored.

The planner (an LLM on Amazon Bedrock, or the built-in heuristic) picks the next
tool from OpenCV 5 measurements. After every action OpenCV 5 re-measures the
photo, and those numbers decide whether the step is kept, retried with gentler
settings, or abandoned. Anything the agent can't decide safely becomes a
question for a human instead of a guess.
"""
import json
import os
import time

import cv2
import numpy as np

from . import perception
from .tools import TOOLS

MAX_STEPS = 10
MAX_RETRIES = 2

# thresholds (tuned on clean vs damaged samples; see eval/)
SCRATCH_MIN = 0.0045      # below this, "scratches" are probably real thin detail
SCRATCH_REVIEW = 0.03     # repair_area above this means inventing large areas -> human approves the mask
NOISE_MIN = 6.0
FADED_BLACK_POINT = 0.15
CAST_MIN = 8.0
AMBIGUOUS_CAST = 15.0     # strong tint but blacks intact: could be the real scene colour
STRUCTURE_FLOOR = {"denoise": 0.90, "sharpen": 0.90, "inpaint_scratches": 0.85}

# What each tool is supposed to fix, and the minimum improvement that counts.
# A step is judged on its own target; side effects on other metrics are logged,
# because fixing one thing often *reveals* other damage (stretching faded tones
# makes grain and scratches visible), which later steps then repair.
TARGET = {
    "correct_colour":    ("highlight_cast", 1.0),
    "enhance_contrast":  ("black_point", 0.03),
    "inpaint_scratches": ("scratch_coverage", 0.0005),
    "denoise":           ("noise_sigma", 0.5),
    "sharpen":           ("sharpness", -1e9),  # judged on structure only
}
REVEALABLE = ("noise_sigma", "scratch_coverage")

# Side-effect checks: catch repairs that "work" on their own metric but do harm.
# Both use only the before/after images (no clean reference), so they apply to real photos.
NOISE_GROWTH_MAX = 1.15   # enhance_contrast: grain may grow with the tone stretch (revealed), not faster (created)
METHOD_NOISE_MAX = 1.30   # denoise: may remove about the estimated noise; much more means real detail went too


# ---------- verification ----------

def structure_similarity(a, b) -> float:
    """SSIM on blurred greyscale: did the step keep the photo's structure?

    Blurring first means removing grain or thin scratches isn't punished,
    but smearing faces and edges is. It compares blurred, low-frequency
    structure, so very large photos are compared on a <= 2048 px copy, and in
    float32: at 16 MP, float64 working arrays alone would need about 1 GB.
    """
    side = max(a.shape[:2])
    if side > 2048:
        size = (round(a.shape[1] * 2048 / side), round(a.shape[0] * 2048 / side))
        a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, size, interpolation=cv2.INTER_AREA)
    ga = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (0, 0), 2).astype(np.float32)
    gb = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (0, 0), 2).astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = g(ga), g(gb)
    va, vb, cov = g(ga * ga) - mu_a ** 2, g(gb * gb) - mu_b ** 2, g(ga * gb) - mu_a * mu_b
    ssim = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(ssim.mean())


def side_effects(tool, before_img, after_img, before: dict, after: dict):
    """Returns (ok, why, measured) for repairs whose own metric can't see the harm they do.

    enhance_contrast: a pure tone stretch scales noise and tonal range together, so
      noise/contrast stays ~1.0x. Local contrast (CLAHE) pushed too far amplifies
      grain beyond the stretch: that is created noise, not revealed noise.
    denoise ("method noise", Buades et al.): what NL-means removes should look like
      the noise it estimated. Removing much more than that means it smoothed away detail.
    """
    if tool == "enhance_contrast":
        growth = (after["noise_sigma"] / max(after["contrast"], 1e-3)) / \
                 max(before["noise_sigma"] / max(before["contrast"], 1e-3), 1e-6)
        return (growth <= NOISE_GROWTH_MAX,
                f"created grain: noise grew {growth:.2f}x relative to the tones (max {NOISE_GROWTH_MAX})",
                {"noise_growth": round(float(growth), 3)})
    if tool == "denoise":
        gray = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ratio = float((gray(before_img) - gray(after_img)).std()) / max(before["noise_sigma"], 1e-6)
        return (ratio <= METHOD_NOISE_MAX,
                f"removed detail: took away {ratio:.2f}x the estimated grain (max {METHOD_NOISE_MAX})",
                {"method_noise": round(ratio, 3)})
    return True, "", {}


def damage_score(r: dict) -> float:
    """One number for 'how damaged does this look'. Lower is better; ~0 = no visible damage.

    A colour cast only counts when the blacks are also faded (the ageing signature),
    so a naturally warm scene isn't scored as damaged.
    """
    faded = r["black_point"] > FADED_BLACK_POINT
    return round(
        300.0 * max(0.0, r["scratch_coverage"] - 0.002)
        + max(0.0, r["noise_sigma"] - 4.0) / 4.0
        + max(0.0, r["black_point"] - 0.10) * 10.0
        + (max(0.0, r["highlight_cast"] - 5.0) / 6.0 if faded else 0.0),
        3,
    )


# ---------- planners ----------

def _gentle(history, tool):
    """Each earlier rejection of a tool makes the next attempt 40% gentler."""
    n = sum(1 for h in history if h["tool"] == tool and not h["kept"])
    return 0.6 ** n, (" (gentler retry)" if n else "")


class HeuristicPlanner:
    """Deterministic planner: runs offline, and is the fallback for the LLM planner.

    Order: colour -> tones -> scratches -> grain. Restoring tones first makes
    faint scratches stand out, so the re-measurement after each step finds damage
    that was invisible at the start.
    """

    name = "heuristic"

    def next_action(self, r: dict, history: list, blocked: set) -> dict:
        faded = r["black_point"] > FADED_BLACK_POINT

        if "correct_colour" not in blocked:
            if faded and r["highlight_cast"] > CAST_MIN:
                f, note = _gentle(history, "correct_colour")
                return {"tool": "correct_colour", "params": {"strength": round(0.9 * f, 2)},
                        "reason": f"highlight_cast {r['highlight_cast']} with faded blacks "
                                  f"({r['black_point']}) = ageing tint{note}"}
            if not faded and r["highlight_cast"] > AMBIGUOUS_CAST:
                return {"tool": "ask_human", "about": "correct_colour",
                        "question": f"The highlights are tinted (cast {r['highlight_cast']}) but the "
                                    f"blacks are intact. Is this ageing, or the real colour of the scene?",
                        "reason": "ambiguous colour cast"}

        if faded and "enhance_contrast" not in blocked:
            f, note = _gentle(history, "enhance_contrast")
            return {"tool": "enhance_contrast", "params": {"clip": round(1.0 * f, 2) if f < 1 else 0.0},
                    "reason": f"black_point {r['black_point']} > {FADED_BLACK_POINT}: blacks faded{note}"}

        if r["scratch_coverage"] > SCRATCH_MIN and "inpaint_scratches" not in blocked:
            # judged on what would actually be repainted (the full mask), not the confident subset
            if r.get("repair_area", r["scratch_coverage"]) > SCRATCH_REVIEW:
                return {"tool": "ask_human", "about": "inpaint_scratches",
                        "question": f"Repairing would repaint {r.get('repair_area', r['scratch_coverage'])*100:.1f}% of the photo. "
                                    "Approve the repair mask before I fill it in?",
                        "reason": "large damaged area"}
            f, note = _gentle(history, "inpaint_scratches")
            return {"tool": "inpaint_scratches",
                    "params": {"radius": 3, "method": "telea" if f == 1.0 else "ns"},
                    "reason": f"scratch_coverage {r['scratch_coverage']:.4f} > {SCRATCH_MIN}{note}"}

        if r["noise_sigma"] > NOISE_MIN and "denoise" not in blocked:
            f, note = _gentle(history, "denoise")
            return {"tool": "denoise", "params": {"strength": round(float(np.clip(r['noise_sigma'] * f, 3, 15)), 1)},
                    "reason": f"noise_sigma {r['noise_sigma']} > {NOISE_MIN}{note}"}

        return {"tool": "done", "params": {}, "reason": "no remaining measurement above its threshold"}


# Allowed parameters per tool: (min, max) for numbers, a tuple of choices for strings.
# LLM replies are clamped to these, so a bad value can never crash or overdo a tool.
PARAM_RANGES = {
    "correct_colour": {"strength": (0.0, 1.0)},
    "enhance_contrast": {"clip": (0.0, 4.0)},
    "inpaint_scratches": {"radius": (2, 7), "method": ("telea", "ns")},
    "denoise": {"strength": (3.0, 15.0)},
    "sharpen": {"amount": (0.2, 1.5)},
}


def sanitize_params(tool: str, params) -> dict:
    """Keep only known parameters, clamped to their allowed range; fill gaps with defaults."""
    out = dict(DEFAULT_PARAMS.get(tool, {}))
    for key, rule in PARAM_RANGES.get(tool, {}).items():
        if not isinstance(params, dict) or key not in params:
            continue
        value = params[key]
        if isinstance(rule[0], str):
            if value in rule:
                out[key] = value
        else:
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            v = min(max(v, rule[0]), rule[1])
            out[key] = int(round(v)) if isinstance(rule[0], int) else round(v, 2)
    return out


def planner_rules() -> str:
    """The planner's rules, generated from the same constants the heuristic and verifier use."""
    return (
        "You are the planner in ReVive, an agent that restores old photos the way a careful human "
        "restorer would. Each turn you get OpenCV measurements of the photo and choose ONE next step. "
        "You never see the pixels.\n\n"
        "Measurements: scratch_coverage = share of the photo with confident scratch/dust marks; "
        "repair_area = share inpainting would actually repaint; noise_sigma = grain; black_point = darkest "
        "tones 0-1 (fading lifts it); highlight_cast = tint of the brightest pixels; contrast; sharpness.\n\n"
        "Rules (same thresholds as the built-in planner):\n"
        f"- Order: colour, then tones, then scratches, then grain.\n"
        f"- correct_colour only if highlight_cast > {CAST_MIN} AND black_point > {FADED_BLACK_POINT} (an ageing "
        f"tint fades the blacks too). If highlight_cast > {AMBIGUOUS_CAST} but blacks are intact, it may be the "
        "real scene colour: ask_human about correct_colour.\n"
        f"- enhance_contrast if black_point > {FADED_BLACK_POINT}; start with clip 0 (pure tone stretch).\n"
        f"- inpaint_scratches if scratch_coverage > {SCRATCH_MIN}. If repair_area > {SCRATCH_REVIEW}, it would "
        "invent a large area: ask_human about inpaint_scratches first.\n"
        f"- denoise if noise_sigma > {NOISE_MIN}; strength about equal to noise_sigma (3-15).\n"
        "- Never call a tool listed as blocked.\n\n"
        "After every step a verifier re-measures and REJECTS the step unless: its own target metric improved; "
        f"structure similarity stays above its floor {json.dumps(STRUCTURE_FLOOR)}; contrast did not create "
        f"grain (noise growth <= {NOISE_GROWTH_MAX}x); denoise removed no more than {METHOD_NOISE_MAX}x the "
        "estimated grain. After a rejection, retry about 40% gentler.\n\n"
        "Reply with ONE JSON object and nothing else, one of:\n"
        '{"tool": "<tool>", "params": {...}, "reason": "<one sentence citing the numbers>"}\n'
        '{"tool": "ask_human", "about": "<tool>", "question": "<question for the person>", "reason": "..."}\n'
        '{"tool": "done", "reason": "..."}\n'
        f"Tools and parameter ranges: {json.dumps({k: v['params'] for k, v in TOOLS.items()})}"
    )


class BedrockPlanner:
    """LLM planner on Amazon Bedrock (Converse API).

    It sees only OpenCV's measurements and the verifier's verdicts, never the
    pixels, so every decision is grounded in the vision output. Its reply is
    validated: unknown or blocked tools and unparsable replies fall back to the
    heuristic for that step, and parameters are clamped to safe ranges.
    """

    name = "bedrock"
    MAX_TOKENS = 2000  # room for models that think before answering

    def __init__(self, model_id=None, region=None):
        import boto3
        self.model_id = model_id or os.environ.get("REVIVE_MODEL_ID", "anthropic.claude-haiku-4-5-20251001-v1:0")
        self.client = boto3.client("bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "eu-west-2"))
        self.fallback = HeuristicPlanner()

    def next_action(self, r: dict, history: list, blocked: set) -> dict:
        prompt = (
            f"OpenCV measurements now: {json.dumps(perception.public(r))}\n"
            f"Steps so far (kept=false means the verifier rejected it): {json.dumps(history)}\n"
            f"Blocked tools (finished, given up on, or waiting for a human): {sorted(blocked)}\n"
            "Choose the next step."
        )
        try:
            # No temperature/top_p: current Claude models reject non-default sampling settings.
            resp = self.client.converse(
                modelId=self.model_id,
                system=[{"text": planner_rules()}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": self.MAX_TOKENS},
            )
            # newer models may put a reasoning block first; use the first text block
            text = next(c["text"] for c in resp["output"]["message"]["content"] if "text" in c)
            action = json.loads(text[text.find("{"): text.rfind("}") + 1])
            tool = action.get("tool")
            if tool not in TOOLS and tool not in ("done", "ask_human"):
                raise ValueError(f"unknown tool {tool}")
            if tool in blocked:
                raise ValueError(f"blocked tool {tool}")
            if tool == "ask_human" and action.get("about") not in TOOLS:
                raise ValueError(f"question about unknown tool {action.get('about')}")
            if tool in TOOLS:
                action["params"] = sanitize_params(tool, action.get("params"))
            action["reason"] = str(action.get("reason", ""))[:300]
            return action
        except Exception as e:  # a model hiccup must never stop a restoration
            a = self.fallback.next_action(r, history, blocked)
            a["reason"] = f"[fallback: {type(e).__name__}] " + a["reason"]
            if str(e):   # keep the model's error in the trace so a fallback is explainable, not just counted
                a["reason"] += f" (LLM error: {str(e)[:160]})"
            return a


# ---------- the loop ----------

DEFAULT_PARAMS = {"inpaint_scratches": {"radius": 3, "method": "telea"},
                  "correct_colour": {"strength": 0.9}}


def _approved_params(tool, history):
    """Settings for a human-approved repair; gentler after each verifier rejection."""
    f, note = _gentle(history, tool)
    params = dict(DEFAULT_PARAMS.get(tool, {}))
    if f < 1:
        if tool == "inpaint_scratches":
            params = {"radius": 2, "method": "ns"}
        elif tool == "correct_colour":
            params = {"strength": round(0.9 * f, 2)}
    return params, note


def restore(img, planner=None, log=print, answers=None, on_event=None, verify=True):
    """Run the agent on one image. Returns (restored_image, trace).

    answers: optional {tool: bool} — a human's replies to questions the agent
    would otherwise ask (True = go ahead, False = leave it). This is how the web
    UI resumes a paused restoration after the person clicks approve/decline.

    on_event: optional callback(kind, payload, image=None, mask=None) for live
    observers (the web UI streams these). kinds: "start", "step", "human",
    "question". It only observes; it never changes what the agent does.

    verify: set False only for ablation experiments (eval/agentic.py): every
    step is kept regardless of the re-measurement, to show what the verifier
    prevents.
    """
    answers = answers or {}
    emit = on_event or (lambda *a, **k: None)
    planner = planner or HeuristicPlanner()
    current = img.copy()
    report = perception.analyse(current)
    trace = {"planner": planner.name, "initial": perception.public(report),
             "initial_score": damage_score(report), "steps": [], "questions": []}
    history, blocked, retries = [], set(), {}
    approved = set()  # repairs a human signed off on
    emit("start", {"initial": trace["initial"], "initial_score": trace["initial_score"],
                   "planner": planner.name}, image=current, mask=report["_mask"])

    for step in range(1, MAX_STEPS + 1):
        action = planner.next_action(report, history, blocked)
        tool = action["tool"]

        if tool == "done":
            trace["final_reason"] = action.get("reason", "")
            break
        if tool == "ask_human" and action.get("about") in answers:
            about = action["about"]
            trace.setdefault("human_answers", []).append({"step": step, "about": about, "approved": answers[about]})
            emit("human", trace["human_answers"][-1])
            if not answers[about]:
                blocked.add(about)
                log(f"  step {step}: 👤 human declined {about}")
                continue
            log(f"  step {step}: 👤 human approved {about}")
            params, note = _approved_params(about, history)
            action = {"tool": about, "params": params, "reason": "approved by human" + note}
            tool = about
            approved.add(about)
        elif tool == "ask_human":
            about = action.get("about", "unknown")
            blocked.add(about)  # don't apply it; carry on with the other repairs
            trace["questions"].append({"step": step, "about": about, "question": action.get("question", ""),
                                       "measurements": perception.public(report)})
            emit("question", trace["questions"][-1], image=current, mask=report["_mask"])
            log(f"  step {step}: ❓ ask human about {about}: {action.get('question', '')}")
            continue

        t0 = time.perf_counter()
        params = action.get("params", {})
        mask_used = report["_mask"]
        fn = TOOLS[tool]["fn"]
        candidate = fn(current, report["_mask"], **params) if tool == "inpaint_scratches" else fn(current, **params)
        new_report = perception.analyse(candidate)
        ms = round((time.perf_counter() - t0) * 1000, 1)

        before, after = damage_score(report), damage_score(new_report)
        structure = structure_similarity(current, candidate)
        floor = STRUCTURE_FLOOR.get(tool, 0.0)
        if tool == "inpaint_scratches" and tool in approved:
            # the person approved repainting this much of the photo, so expect
            # structure to change by about that much; scratches must still drop
            floor = round(floor - float((mask_used > 0).mean()), 3)
        metric, min_gain = TARGET[tool]
        gain = report[metric] - new_report[metric]
        target_ok = gain >= min_gain
        side_ok, side_why, checks = side_effects(tool, current, candidate, report, new_report)
        kept = (target_ok and structure >= floor and side_ok) if verify else True
        revealed = {m: [report[m], new_report[m]] for m in REVEALABLE
                    if m != metric and new_report[m] > report[m] * 1.15 and new_report[m] - report[m] > 0.3 * (1 if m == "noise_sigma" else 0.003)}

        entry = {"step": step, "tool": tool, "params": params, "reason": action.get("reason", ""),
                 "measured_before": perception.public(report), "measured_after": perception.public(new_report),
                 "target": metric, "target_gain": round(float(gain), 4),
                 "score_before": before, "score_after": after, "structure": round(structure, 3),
                 "checks": checks, "revealed": revealed, "kept": kept, "ms": ms}
        if kept:
            current, report = candidate, new_report
            blocked.add(tool)
            entry["verdict"] = (f"kept: {metric} improved, structure preserved" if verify
                                else "kept (verifier disabled)")
            if revealed:
                entry["verdict"] += "; revealed hidden " + ", ".join(revealed)
        else:
            why = (f"{metric} did not improve enough" if not target_ok
                   else f"structure {structure:.3f} below {floor}" if structure < floor
                   else side_why)
            retries[tool] = retries.get(tool, 0) + 1
            if retries[tool] > MAX_RETRIES:
                blocked.add(tool)
                entry["verdict"] = f"rejected ({why}); giving up on {tool}"
            else:
                entry["verdict"] = f"rejected ({why}); retrying gentler"
        history.append({k: entry[k] for k in ("tool", "params", "kept", "verdict")})
        trace["steps"].append(entry)
        emit("step", entry, image=candidate, mask=mask_used if tool == "inpaint_scratches" else None)
        log(f"  step {step}: {tool}{params} → {entry['verdict']} (score {before}→{after}, structure {structure:.3f})")

    trace["final"] = perception.public(report)
    trace["final_score"] = damage_score(report)
    if trace["questions"]:
        trace["status"] = "needs_human_review"
    elif trace["final_score"] > 1.0:
        trace["status"] = "needs_human_review"
        trace["questions"].append({"step": None, "about": "residual_damage",
                                   "question": f"Damage score is still {trace['final_score']} after automatic "
                                               "repair. Review the result?"})
    else:
        trace["status"] = "complete"
    return current, trace
