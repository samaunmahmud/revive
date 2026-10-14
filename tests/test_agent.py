"""The agent's promises: do no harm, verify every step, ask instead of guessing, survive LLM failures."""
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
