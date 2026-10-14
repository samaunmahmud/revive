"""ReVive web API: a thin layer over the existing agent.

    uvicorn server.app:app --reload          # dev, from the repo root

POST /api/jobs                 start a restoration (upload or sample) -> {job_id}
GET  /api/jobs/{id}/events     live trace as Server-Sent Events
POST /api/jobs/{id}/answer     answer the agent's questions -> new job re-run on the same photo
GET  /api/jobs/{id}            status + trace once finished
GET  /api/jobs/{id}/files/...  images, masks, restored photo, trace.json
GET  /api/samples              bundled demo photos
GET  /api/config               planner availability

If web/dist exists (npm run build), the built React app is served at /.
"""
import json
import os
import re

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from revive.tools import TOOLS
from .jobs import MAX_SIDE, JobStore, prepare

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT, "server", "samples")
WEB_DIST = os.path.join(ROOT, "web", "dist")
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
FILE_NAME = re.compile(r"^[a-z0-9_-]+\.(png|jpg|json)$")

app = FastAPI(title="ReVive", version="0.1.0")
jobs = JobStore()

with open(os.path.join(SAMPLES_DIR, "samples.json")) as fh:
    SAMPLES = {s["id"]: s for s in json.load(fh)}


def bedrock_status() -> dict:
    """Whether AWS credentials are visible. The planner still falls back per step if a call fails."""
    try:
        import boto3
        has_creds = boto3.Session().get_credentials() is not None
    except Exception:
        has_creds = False
    return {"available": has_creds,
            "model_id": os.environ.get("REVIVE_MODEL_ID", "anthropic.claude-haiku-4-5-20251001-v1:0"),
            "region": os.environ.get("AWS_REGION", "eu-west-2")}


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/config")
def config():
    return {"bedrock": bedrock_status(), "max_side": MAX_SIDE}


@app.get("/api/samples")
def samples():
    return [{**s, "url": f"/api/samples/{s['id']}.png"} for s in SAMPLES.values()]


app.mount("/api/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


@app.post("/api/jobs")
async def create_job(file: UploadFile | None = File(None), sample: str | None = Form(None),
                     planner: str = Form("heuristic")):
    if planner not in ("heuristic", "bedrock"):
        raise HTTPException(400, "planner must be 'heuristic' or 'bedrock'")
    if sample:
        if sample not in SAMPLES:
            raise HTTPException(404, f"unknown sample {sample}")
        img = cv2.imread(os.path.join(SAMPLES_DIR, f"{sample}.png"))
    elif file is not None:
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "image is larger than 15 MB")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "could not read that file as an image (JPEG, PNG, WebP, TIFF or BMP)")
    else:
        raise HTTPException(400, "send a file or a sample id")
    job = jobs.create(prepare(img), planner, answers={})
    return {"job_id": job.id}


def _job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found (it may have expired)")
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _job(job_id)
    return {"job_id": job.id, "finished": job.finished, "planner": job.planner,
            "answers": job.answers, "parent": job.parent, "trace": job.trace}


@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str, request: Request):
    job = _job(job_id)
    # On reconnect the browser sends the id of the last event it saw; resume after it
    last = request.headers.get("last-event-id", "")
    start = int(last) + 1 if last.isdigit() else 0

    def sse():
        i = start
        for e in job.stream(start):
            if e is None:
                yield ": keep-alive\n\n"   # comment line: stops proxies closing an idle stream
            else:
                yield f"id: {i}\nevent: {e['kind']}\ndata: {json.dumps({**e['data'], 't': e['t']}, default=float)}\n\n"
                i += 1
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class Answers(BaseModel):
    answers: dict[str, bool]


@app.post("/api/jobs/{job_id}/answer")
def answer(job_id: str, body: Answers):
    """Re-run the agent on the same original with the person's answers merged in.

    This uses the agent's own resume hook: restore(img, answers={tool: bool}).
    """
    job = _job(job_id)
    unknown = [k for k in body.answers if k not in TOOLS]
    if unknown:
        raise HTTPException(400, f"can only answer questions about tools, not {unknown}")
    child = jobs.create(job.original(), job.planner, {**job.answers, **body.answers}, parent=job.id)
    return {"job_id": child.id}


@app.get("/api/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str, download: bool = False):
    job = _job(job_id)
    path = os.path.join(job.dir, name)
    if not FILE_NAME.match(name) or not os.path.isfile(path):
        raise HTTPException(404, "file not found")
    filename = f"revive-{job.id}-{name}" if download else None
    return FileResponse(path, filename=filename)


if os.path.isdir(WEB_DIST):
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
