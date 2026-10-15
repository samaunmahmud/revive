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


def _job_original_shape(client, data, name, mime):
    import os
    job = client.post("/api/jobs", files={"file": (name, data, mime)}).json()["job_id"]
    events(client, job)
    from server.jobs import DATA_DIR
    return cv2.imread(os.path.join(DATA_DIR, job, "original.png")).shape


def test_archive_scan_formats_are_accepted(client, aged):
    import numpy as np
    img = cv2.resize(aged("rocket", "faded"), (320, 240))
    ok, tiff16 = cv2.imencode(".tiff", img.astype(np.uint16) * 257)              # 16-bit archive scan
    assert _job_original_shape(client, tiff16.tobytes(), "scan.tiff", "image/tiff") == (240, 320, 3)
    ok, grey = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))       # greyscale
    assert _job_original_shape(client, grey.tobytes(), "bw.png", "image/png") == (240, 320, 3)
    ok, rgba = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_BGR2BGRA))       # transparency
    assert _job_original_shape(client, rgba.tobytes(), "a.png", "image/png") == (240, 320, 3)


def test_phone_photo_exif_rotation_is_applied(client, aged):
    import struct
    img = cv2.resize(aged("rocket", "faded"), (200, 320))                        # stored portrait 200x320
    ok, jpg = cv2.imencode(".jpg", img)
    jpg = jpg.tobytes()
    tiff = b"MM\x00*" + struct.pack(">I", 8) + struct.pack(">H", 1) + struct.pack(">HHIHH", 0x0112, 3, 1, 6, 0) + struct.pack(">I", 0)
    exif = b"\xff\xe1" + struct.pack(">H", 2 + 6 + len(tiff)) + b"Exif\x00\x00" + tiff  # Orientation = 6 (rotate 90)
    assert _job_original_shape(client, jpg[:2] + exif + jpg[2:], "phone.jpg", "image/jpeg") == (200, 320, 3)


def test_full_size_run_uses_the_large_upload_and_same_answers(client, aged):
    big = cv2.resize(aged("coffee", "grainy"), (2400, 1600), interpolation=cv2.INTER_CUBIC)
    ok, png = cv2.imencode(".png", big)
    job = client.post("/api/jobs", files={"file": ("big.png", png.tobytes(), "image/png")}).json()["job_id"]
    done = next(d for k, d in events(client, job) if k == "done")
    assert done["trace"]["size"] == [1600, 1067] and done["full_size"] == [2400, 1600]
    answered = client.post(f"/api/jobs/{job}/answer", json={"answers": {"correct_colour": False}}).json()["job_id"]
    events(client, answered)
    full = client.post(f"/api/jobs/{answered}/fullres").json()["job_id"]
    trace = next(d for k, d in events(client, full) if k == "done")["trace"]
    assert trace["full_res"] and trace["size"] == [2400, 1600]
    assert {"about": "correct_colour", "approved": False} in [{k: h[k] for k in ("about", "approved")} for h in trace.get("human_answers", [])]


def test_full_size_not_offered_for_small_photos(client):
    job = client.post("/api/jobs", data={"sample": "chelsea-clean"}).json()["job_id"]
    assert next(d for k, d in events(client, job) if k == "done")["full_size"] is None
    assert client.post(f"/api/jobs/{job}/fullres").status_code == 400


def test_display_images_are_capped_but_download_is_full_size(client, aged):
    big = cv2.resize(aged("rocket", "faded"), (2400, 1800), interpolation=cv2.INTER_CUBIC)
    ok, png = cv2.imencode(".png", big)
    job = client.post("/api/jobs", files={"file": ("big.png", png.tobytes(), "image/png")}).json()["job_id"]
    events(client, job)
    full = client.post(f"/api/jobs/{job}/fullres").json()["job_id"]
    ev = events(client, full)
    import numpy as np
    def shape(url):
        return cv2.imdecode(np.frombuffer(client.get(url).content, np.uint8), cv2.IMREAD_UNCHANGED).shape[:2]
    step = next(d for k, d in ev if k == "step")
    done = next(d for k, d in ev if k == "done")
    assert max(shape(step["image"])) == 1600 and max(shape(done["restored"])) == 1600   # browser copies
    assert shape(done["restored_png"]) == (1800, 2400)                                   # the download


def _png_data_url(mask):
    import base64
    ok, buf = cv2.imencode(".png", mask)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def test_mask_edits_reach_the_agent_and_carry_over(client):
    import numpy as np
    job = client.post("/api/jobs", data={"sample": "chelsea-clean"}).json()["job_id"]
    events(client, job)
    h, w = cv2.imread("server/samples/chelsea-clean.png").shape[:2]
    paint = np.zeros((h // 2, w // 2, 4), np.uint8)                   # drawn on a smaller canvas, RGBA
    cv2.line(paint, (w // 6, h // 20), (w // 6 + 4, h // 2 - h // 20), (255, 255, 255, 255), 4)
    child = client.post(f"/api/jobs/{job}/answer", json={"mask_add": _png_data_url(paint)}).json()["job_id"]
    done = next(d for k, d in events(client, child) if k == "done")
    assert done["trace"]["mask_edits"]["added"] > 0
    assert any(s["tool"] == "inpaint_scratches" for s in done["trace"]["steps"])
    meta = client.get(f"/api/jobs/{child}").json()
    assert meta["mask_edits"] and meta["answers"]["inpaint_scratches"] is True

    # a later answer without new edits keeps them
    grandchild = client.post(f"/api/jobs/{child}/answer", json={"answers": {"correct_colour": True}}).json()["job_id"]
    assert client.get(f"/api/jobs/{grandchild}").json()["mask_edits"]


def test_bad_masks_are_rejected(client):
    job = client.post("/api/jobs", data={"sample": "chelsea-clean"}).json()["job_id"]
    assert client.post(f"/api/jobs/{job}/answer", json={"mask_add": "data:text/plain;base64,aGk="}).status_code == 400
    assert client.post(f"/api/jobs/{job}/answer", json={"mask_add": "data:image/png;base64,bm90YXBuZw=="}).status_code == 400
    import numpy as np
    wrong_shape = _png_data_url(np.full((100, 400), 255, np.uint8))
    assert client.post(f"/api/jobs/{job}/answer", json={"mask_add": wrong_shape}).status_code == 400


def test_busy_server_says_so_instead_of_queueing_forever(client):
    from server import app as app_module
    from server import jobs as jobs_module

    class Stuck:  # an unfinished job from some client
        def __init__(self, who):
            self.id, self.client, self.finished, self.created = f"stuck-{who}-{id(self)}", who, False, 0.0

    store = app_module.jobs
    fakes = [Stuck("203.0.113.9") for _ in range(jobs_module.MAX_PENDING_PER_CLIENT)]
    with store._lock:
        store._jobs.update({f.id: f for f in fakes})
    try:
        # the same visitor (CloudFront appends the real address last; the first entry is forged)
        r = client.post("/api/jobs", data={"sample": "chelsea-clean"}, headers={"X-Forwarded-For": "1.2.3.4, 203.0.113.9"})
        assert r.status_code == 503 and "several restorations" in r.json()["detail"]
        # someone else still gets in
        r = client.post("/api/jobs", data={"sample": "chelsea-clean"}, headers={"X-Forwarded-For": "198.51.100.7"})
        assert r.status_code == 200
        events(client, r.json()["job_id"])
        # and when the whole server is full, everyone is told to wait
        extra = [Stuck(f"198.51.100.{i}") for i in range(jobs_module.MAX_PENDING)]
        with store._lock:
            store._jobs.update({f.id: f for f in extra})
        r = client.post("/api/jobs", data={"sample": "chelsea-clean"})
        assert r.status_code == 503 and r.headers["retry-after"] == "30"
    finally:
        with store._lock:
            for f in [*fakes, *locals().get("extra", [])]:
                store._jobs.pop(f.id, None)
