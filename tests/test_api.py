"""Web API: upload -> live SSE trace -> answer -> files, plus input validation."""
import json

import cv2
import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def events(client, job_id):
    """Read the SSE stream to the end; returns [(kind, data)]."""
    out, kind = [], None
    with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        for line in resp.iter_lines():
            if line.startswith("event:"):
                kind = line[6:].strip()
            elif line.startswith("data:"):
                out.append((kind, json.loads(line[5:])))
    return out


def test_health_and_samples(client):
    assert client.get("/api/health").json() == {"ok": True}
    samples = client.get("/api/samples").json()
    assert len(samples) >= 4
    assert client.get(samples[0]["url"]).status_code == 200


def test_sample_job_streams_the_trace_live(client):
    job = client.post("/api/jobs", data={"sample": "astronaut-aged", "planner": "heuristic"}).json()["job_id"]
    ev = events(client, job)
    kinds = [k for k, _ in ev]
    assert kinds[0] == "queued" and "start" in kinds and "step" in kinds and kinds[-1] == "end"
    done = next(d for k, d in ev if k == "done")
    assert done["trace"]["status"] in ("complete", "needs_human_review")
    assert client.get(done["restored_png"]).status_code == 200
    trace = client.get(f"/api/jobs/{job}/files/trace.json").json()
    assert trace["steps"] == done["trace"]["steps"]


def test_question_shows_mask_and_answer_reruns(client):
    job = client.post("/api/jobs", data={"sample": "chelsea-heavy"}).json()["job_id"]
    question = next(d for k, d in events(client, job) if k == "question")
    assert question["about"] == "inpaint_scratches"
    assert client.get(question["mask"]).status_code == 200          # damage-mask overlay for the UI
    child = client.post(f"/api/jobs/{job}/answer", json={"answers": {"inpaint_scratches": True}}).json()["job_id"]
    done = next(d for k, d in events(client, child) if k == "done")
    assert done["trace"]["human_answers"][0]["approved"] is True
    assert client.get(f"/api/jobs/{child}").json()["parent"] == job


def test_upload_a_photo(client, aged):
    ok, buf = cv2.imencode(".jpg", aged("rocket", "faded"))
    job = client.post("/api/jobs", files={"file": ("old.jpg", buf.tobytes(), "image/jpeg")}).json()["job_id"]
    assert any(k == "done" for k, _ in events(client, job))


def test_rejects_bad_input(client):
    assert client.post("/api/jobs", files={"file": ("x.jpg", b"not an image", "image/jpeg")}).status_code == 400
    assert client.post("/api/jobs", data={"sample": "nope"}).status_code == 404
    assert client.post("/api/jobs", data={"sample": "astronaut-aged", "planner": "gpt"}).status_code == 400
    assert client.post("/api/jobs", data={}).status_code == 400


def test_answers_only_for_known_tools(client):
    job = client.post("/api/jobs", data={"sample": "chelsea-clean"}).json()["job_id"]
    events(client, job)
    assert client.post(f"/api/jobs/{job}/answer", json={"answers": {"rm_rf": True}}).status_code == 400


def test_files_cannot_escape_the_job_folder(client):
    job = client.post("/api/jobs", data={"sample": "chelsea-clean"}).json()["job_id"]
    events(client, job)
    for name in ("..%2F..%2Fapp.py", "..%2Fsamples.json", "trace.txt"):
        assert client.get(f"/api/jobs/{job}/files/{name}").status_code == 404
    assert client.get("/api/jobs/doesnotexist").status_code == 404
