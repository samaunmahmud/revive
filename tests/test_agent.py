"""The agent's promises: do no harm, verify every step, ask instead of guessing, survive LLM failures."""
import cv2
import numpy as np

from revive import agent, perception


class ScriptedPlanner:
    """Returns a fixed list of actions, then defers to the heuristic planner."""

    name = "scripted"

    def __init__(self, actions):
        self.actions = list(actions)
        self.inner = agent.HeuristicPlanner()

    def next_action(self, r, history, blocked):
        if self.actions:
            return self.actions.pop(0)
        return self.inner.next_action(r, history, blocked)


def test_undamaged_photo_is_returned_untouched(aged, quiet):
    img = aged("chelsea", "none")
    restored, trace = agent.restore(img, log=quiet)
    assert trace["steps"] == []
    assert trace["status"] == "complete"
    assert np.array_equal(restored, img)


def test_faded_photo_gets_its_tones_restored(aged, quiet):
    img = aged("rocket", "faded")
    restored, trace = agent.restore(img, log=quiet)
    kept = [s["tool"] for s in trace["steps"] if s["kept"]]
    assert kept == ["enhance_contrast"]
    assert perception.analyse(restored)["black_point"] < 0.15


def test_every_step_is_traced_with_evidence(aged, quiet):
    _, trace = agent.restore(aged("astronaut", "aged"), log=quiet)
    assert trace["steps"]
    for s in trace["steps"]:
        for key in ("tool", "params", "reason", "measured_before", "measured_after",
                    "target", "target_gain", "structure", "checks", "revealed", "kept", "verdict", "ms"):
            assert key in s


def test_verifier_rejects_contrast_that_creates_grain_then_retries_gentler(aged, quiet):
    img = aged("astronaut", "faded")
    planner = ScriptedPlanner([{"tool": "enhance_contrast", "params": {"clip": 4.0}, "reason": "test: over-strong"}])
    _, trace = agent.restore(img, planner=planner, log=quiet)
    first, second = trace["steps"][0], trace["steps"][1]
    assert first["tool"] == "enhance_contrast" and not first["kept"]
    assert "created grain" in first["verdict"]
    assert second["tool"] == "enhance_contrast" and second["kept"]


def test_verifier_off_keeps_the_harmful_step(aged, quiet):
    planner = ScriptedPlanner([{"tool": "enhance_contrast", "params": {"clip": 4.0}, "reason": "test"}])
    _, trace = agent.restore(aged("astronaut", "faded"), planner=planner, log=quiet, verify=False)
    assert trace["steps"][0]["kept"]


def test_cast_with_intact_blacks_becomes_a_question(aged, quiet):
    img = aged("coffee", "grainy")  # warm highlights, intact blacks
    _, trace = agent.restore(img, log=quiet)
    assert "correct_colour" in [q["about"] for q in trace["questions"]]
    assert "correct_colour" not in [s["tool"] for s in trace["steps"]]
    assert trace["status"] == "needs_human_review"


def test_human_answers_resume_the_restoration(aged, quiet):
    img = aged("coffee", "grainy")
    _, yes = agent.restore(img, log=quiet, answers={"correct_colour": True})
    _, no = agent.restore(img, log=quiet, answers={"correct_colour": False})
    assert "correct_colour" in [s["tool"] for s in yes["steps"] if s["kept"]]
    assert "correct_colour" not in [s["tool"] for s in no["steps"]]
    assert {"about": "correct_colour", "approved": True} in [
        {k: h[k] for k in ("about", "approved")} for h in yes["human_answers"]]


def test_large_damage_asks_before_repainting(aged, quiet):
    img = aged("chelsea", "heavy")
    _, trace = agent.restore(img, log=quiet)
    q = next(q for q in trace["questions"] if q["about"] == "inpaint_scratches")
    assert "repaint" in q["question"]
    assert "inpaint_scratches" not in [s["tool"] for s in trace["steps"]]
    _, approved = agent.restore(img, log=quiet, answers={"inpaint_scratches": True})
    assert any(s["tool"] == "inpaint_scratches" and s["kept"] for s in approved["steps"])
    assert approved["final_score"] < trace["final_score"]


class BrokenClient:
    def converse(self, **_):
        raise TimeoutError("simulated outage")


def test_bedrock_failure_falls_back_to_the_heuristic(aged, quiet):
    img = aged("astronaut", "aged")
    planner = agent.BedrockPlanner.__new__(agent.BedrockPlanner)
    planner.model_id, planner.client, planner.fallback = "test", BrokenClient(), agent.HeuristicPlanner()
    r_llm, t_llm = agent.restore(img, planner=planner, log=quiet)
    r_h, _ = agent.restore(img, log=quiet)
    assert np.array_equal(r_llm, r_h)
    assert all(s["reason"].startswith("[fallback: TimeoutError]") for s in t_llm["steps"])


def test_live_events_follow_the_trace(aged, quiet):
    events = []
    _, trace = agent.restore(aged("astronaut", "aged"), log=quiet,
                             on_event=lambda kind, payload, image=None, mask=None: events.append(kind))
    assert events[0] == "start"
    assert events.count("step") == len(trace["steps"])


class FakeBedrock:
    """Returns scripted replies and records each request."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    def converse(self, **kwargs):
        self.requests.append(kwargs)
        text = self.replies.pop(0) if self.replies else '{"tool": "done", "reason": "ok"}'
        return {"output": {"message": {"content": [{"reasoningContent": {}}, {"text": text}]}},
                "usage": {"inputTokens": 1, "outputTokens": 1}}


def _bedrock(client):
    p = agent.BedrockPlanner.__new__(agent.BedrockPlanner)
    p.model_id, p.client, p.fallback = "test", client, agent.HeuristicPlanner()
    return p


def test_llm_params_are_clamped_and_unknown_keys_dropped(aged, quiet):
    client = FakeBedrock(['Sure! {"tool": "enhance_contrast", "params": {"clip": 40, "foo": 1}, "reason": "blacks faded"}'])
    _, trace = agent.restore(aged("rocket", "faded"), planner=_bedrock(client), log=quiet)
    step = trace["steps"][0]
    assert step["tool"] == "enhance_contrast" and step["params"] == {"clip": 4.0}   # clamped, 'foo' dropped
    assert not step["kept"]                                  # and the verifier still rejects the over-strong clip


def test_llm_question_about_unknown_tool_falls_back(aged, quiet):
    client = FakeBedrock(['{"tool": "ask_human", "about": "make_pretty", "question": "?"}'])
    _, trace = agent.restore(aged("rocket", "faded"), planner=_bedrock(client), log=quiet)
    assert trace["steps"][0]["reason"].startswith("[fallback: ValueError]")


def test_llm_request_carries_rules_and_no_sampling_params(aged, quiet):
    client = FakeBedrock([])
    agent.restore(aged("rocket", "faded"), planner=_bedrock(client), log=quiet)
    req = client.requests[0]
    rules = req["system"][0]["text"]
    assert str(agent.SCRATCH_REVIEW) in rules and str(agent.NOISE_GROWTH_MAX) in rules and "repair_area" in rules
    assert "temperature" not in req["inferenceConfig"] and "topP" not in req["inferenceConfig"]
    assert all("image" not in c for c in req["messages"][0]["content"])


def test_sanitize_params_handles_garbage():
    assert agent.sanitize_params("denoise", {"strength": "lots"}) == {}
    assert agent.sanitize_params("denoise", {"strength": 99}) == {"strength": 15.0}
    assert agent.sanitize_params("inpaint_scratches", {"radius": 9.7, "method": "magic"}) == {"radius": 7, "method": "telea"}
    assert agent.sanitize_params("correct_colour", None) == {"strength": 0.9}


# ---------- human mask edits ----------

def _wide_scratch(clean_img):
    """A 6 px scratch: wider than the detector's 2.5 px limit, so it is missed."""
    img = clean_img.copy()
    h, w = img.shape[:2]
    p1, p2 = (w // 3, int(h * .1)), (w // 3 + 8, int(h * .9))
    cv2.line(img, p1, p2, (235, 235, 235), 6)
    stripe, brush = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    cv2.line(stripe, p1, p2, 255, 6)
    cv2.line(brush, p1, p2, 255, 10)
    return img, stripe, brush


def test_painted_damage_is_repaired_even_when_undetected(clean, quiet):
    img, stripe, brush = _wide_scratch(clean["astronaut"])
    _, auto = agent.restore(img, log=quiet)
    assert not any(s["tool"] == "inpaint_scratches" for s in auto["steps"])   # the detector misses it

    restored, trace = agent.restore(img, log=quiet, mask_edits={"add": brush})
    step = next(s for s in trace["steps"] if s["tool"] == "inpaint_scratches")
    assert step["kept"] and step["reason"].startswith("repairing the damage you marked")
    a0, a1 = step["checks"]["painted_anomaly"]
    assert a1 < a0 * (1 - agent.PAINTED_ANOMALY_DROP)
    err = lambda im: np.abs(im.astype(float) - clean["astronaut"].astype(float))[stripe > 0].mean()
    assert err(restored) < 0.25 * err(img)
    assert trace["mask_edits"]["added"] > 0


def test_protected_pixels_are_never_repainted(aged, quiet):
    img = aged("astronaut", "scratched")
    _, auto = agent.restore(img, log=quiet)
    assert any(s["tool"] == "inpaint_scratches" and s["kept"] for s in auto["steps"])
    protect = np.zeros(img.shape[:2], np.uint8)
    protect[:, : img.shape[1] // 2] = 255            # protect the left half
    restored, trace = agent.restore(img, log=quiet, mask_edits={"erase": protect})
    left = slice(None), slice(0, img.shape[1] // 2)
    # nothing in the left half may change except by global tools (none run on this profile)
    assert [s["tool"] for s in trace["steps"] if s["kept"]] == ["inpaint_scratches"]
    assert np.array_equal(restored[left], img[left])


def test_declining_the_repair_overrides_painted_marks(clean, quiet):
    img, _, brush = _wide_scratch(clean["chelsea"])
    restored, trace = agent.restore(img, log=quiet, answers={"inpaint_scratches": False},
                                    mask_edits={"add": brush})
    assert not any(s["tool"] == "inpaint_scratches" for s in trace["steps"])


def test_edits_drawn_on_the_preview_scale_to_full_size(clean, quiet):
    img, _, brush = _wide_scratch(clean["rocket"])
    small = cv2.resize(brush, (brush.shape[1] // 2, brush.shape[0] // 2), interpolation=cv2.INTER_AREA)
    _, trace = agent.restore(img, log=quiet, mask_edits={"add": small})
    assert any(s["tool"] == "inpaint_scratches" and s["kept"] for s in trace["steps"])
