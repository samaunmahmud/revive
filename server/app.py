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
import base64
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
from . import storage
from .jobs import FULL_MAX_SIDE, MAX_SIDE, Busy, JobStore, prepare

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT, "server", "samples")
WEB_DIST = os.path.join(ROOT, "web", "dist")
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 40_000_000  # decoded size limit, protects server memory
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
    return {"bedrock": bedrock_status(), "max_side": MAX_SIDE, "full_max_side": FULL_MAX_SIDE,
            "storage": storage.describe()}


@app.get("/api/samples")
def samples():
    return [{**s, "url": f"/api/samples/{s['id']}.png"} for s in SAMPLES.values()]


app.mount("/api/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


@app.post("/api/jobs")
async def create_job(request: Request, file: UploadFile | None = File(None), sample: str | None = Form(None),
                     planner: str = Form("heuristic")):
    if planner not in ("heuristic", "bedrock"):
        raise HTTPException(400, "planner must be 'heuristic' or 'bedrock'")
    client = admit(request)
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
        if img.shape[0] * img.shape[1] > MAX_PIXELS:
            raise HTTPException(413, "image is larger than 40 megapixels")
    else:
        raise HTTPException(400, "send a file or a sample id")
    job = jobs.create(prepare(img), planner, answers={}, full=prepare(img, FULL_MAX_SIDE), client=client)
    return {"job_id": job.id}


def client_id(request: Request) -> str:
    """Who is asking, for the per-visitor job limit. Behind CloudFront the viewer's address is the
    LAST X-Forwarded-For entry (CloudFront appends it; earlier entries come from the client and can be forged)."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def admit(request: Request) -> str:
    client = client_id(request)
    try:
        jobs.admit(client)
    except Busy as e:
        raise HTTPException(503, str(e), headers={"Retry-After": "30"})
    return client


def _job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found (it may have expired)")
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _job(job_id)
    return {"job_id": job.id, "finished": job.finished, "planner": job.planner, "answers": job.answers,
            "parent": job.parent, "full_size": job.full_size, "mask_edits": bool(job.mask_edits()), "full_res": job.full_res, "trace": job.trace}


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


MAX_MASK_BYTES = 4 * 1024 * 1024


class Answers(BaseModel):
    answers: dict[str, bool] = {}
    mask_add: str | None = None     # PNG data URL: damage the person painted
    mask_erase: str | None = None   # PNG data URL: detail the person protected


def decode_mask(data_url: str | None, shape) -> np.ndarray | None:
    """Brush layer from the browser (PNG data URL, painted pixels opaque) -> uint8 mask."""
    if not data_url:
        return None
    head, _, b64 = data_url.partition(",")
    if not head.startswith("data:image/png;base64") or len(b64) > MAX_MASK_BYTES * 4 // 3:
        raise HTTPException(400, "mask must be a PNG data URL under 4 MB")
    try:
        m = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_UNCHANGED)
    except ValueError:
        m = None
    if m is None:
        raise HTTPException(400, "could not read the mask")
    m = m[..., 3] if m.ndim == 3 and m.shape[2] == 4 else (m.max(axis=2) if m.ndim == 3 else m)
    h, w = shape
    if abs(m.shape[1] / m.shape[0] - w / h) > 0.02:
        raise HTTPException(400, "mask does not match the photo's shape")
    return (cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) > 0).astype(np.uint8) * 255


@app.post("/api/jobs/{job_id}/answer")
def answer(job_id: str, body: Answers, request: Request):
    """Re-run the agent on the same original with the person's answers merged in.

    This uses the agent's own resume hook: restore(img, answers={tool: bool}),
    plus mask_edits when the person corrected the damage mask. Sending mask edits
    approves the scratch repair; edits not re-sent carry over from the parent job.
    """
    job = _job(job_id)
    unknown = [k for k in body.answers if k not in TOOLS]
    if unknown:
        raise HTTPException(400, f"can only answer questions about tools, not {unknown}")
    client = admit(request)
    original = job.original()
    add, erase = decode_mask(body.mask_add, original.shape[:2]), decode_mask(body.mask_erase, original.shape[:2])
    answers = {**job.answers, **body.answers}
    edits = job.mask_edits()
    if add is not None or erase is not None:
        edits = {"add": add, "erase": erase}
        answers["inpaint_scratches"] = True
    child = jobs.create(original, job.planner, answers, parent=job.id,
                        full=job.full_original(), full_res=job.full_res, edits=edits, client=client)
    return {"job_id": child.id}


@app.post("/api/jobs/{job_id}/fullres")
def full_resolution(job_id: str, request: Request):
    """Run the agent again on the full-size upload, with the same answers.

    The preview is planned on a downscaled copy, which averages away fine grain
    and thin scratches, so the full-size run re-measures rather than replaying
    the preview's steps. It streams like any other job and can still ask.
    """
    job = _job(job_id)
    full = job.full_original()
    if full is None:
        raise HTTPException(400, "this photo is already at full size")
    client = admit(request)
    child = jobs.create(full, job.planner, dict(job.answers), parent=job.id, full_res=True, edits=job.mask_edits(),
                        client=client)
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
