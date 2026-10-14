"""Restoration jobs: run the agent on a background thread and record its events.

The web layer never touches the agent loop. It starts a Job, and the Job calls
agent.restore(..., on_event=...) and appends each event to a list. SSE clients
read that list from any position, so a client that connects late (or
reconnects) still sees every step from the start.

Files live in data/jobs/<id>/ for now; phase 2 moves them to S3.
"""
import json
import os
import shutil
import threading
import time
import traceback
import uuid

import cv2
import numpy as np

from revive import agent

from . import storage

DATA_DIR = os.environ.get("REVIVE_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "jobs"))
MAX_SIDE = 1600           # live preview size: fast enough to watch the agent work
FULL_MAX_SIDE = 4096      # "restore at full size" cap: ~16 MP, tens of seconds on a small Graviton
MAX_JOBS_KEPT = 200       # oldest job folders are deleted beyond this
RUN_SLOTS = threading.BoundedSemaphore(int(os.environ.get("REVIVE_WORKERS", "2")))
FULL_SLOT = threading.BoundedSemaphore(1)  # full-size runs are memory-heavy: one at a time


def prepare(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def mask_overlay(mask: np.ndarray) -> np.ndarray:
    """Transparent PNG (BGRA) with damage pixels in red, laid over the photo by the UI.

    Dilated once more for display only, so 1-2 px scratches stay visible when
    the photo is shown smaller than its real size.
    """
    shown = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    out = np.zeros((*mask.shape, 4), np.uint8)
    out[shown > 0] = (40, 60, 255, 235)
    return out


class Job:
    def __init__(self, original: np.ndarray, planner: str, answers: dict, parent: str | None = None,
                 full: np.ndarray | None = None, full_res: bool = False):
        self.id = uuid.uuid4().hex[:12]
        self.planner = planner
        self.answers = dict(answers)
        self.parent = parent
        self.full_res = full_res  # this job IS the full-size run
        self.dir = os.path.join(DATA_DIR, self.id)
        self.created = time.time()
        self.events: list[dict] = []
        self.finished = False
        self.trace: dict | None = None
        self.cond = threading.Condition()
        os.makedirs(self.dir, exist_ok=True)
        cv2.imwrite(self._path("original.png"), original)        # lossless, re-used by answer re-runs
        cv2.imwrite(self._path("original.jpg"), original, [cv2.IMWRITE_JPEG_QUALITY, 92])
        # keep the larger upload, if any, for a later full-size run
        self.full_size = None
        if full is not None and max(full.shape[:2]) > max(original.shape[:2]):
            cv2.imwrite(self._path("upload.png"), full)
            self.full_size = [full.shape[1], full.shape[0]]

    # ---------- files ----------

    def _path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def url(self, name: str) -> str:
        return f"/api/jobs/{self.id}/files/{name}"

    def original(self) -> np.ndarray:
        return cv2.imread(self._path("original.png"))

    def full_original(self) -> np.ndarray | None:
        return cv2.imread(self._path("upload.png")) if self.full_size else None

    # ---------- events ----------

    def _push(self, kind: str, data: dict):
        with self.cond:
            self.events.append({"kind": kind, "data": data, "t": round(time.time() - self.created, 3)})
            self.cond.notify_all()

    def _on_event(self, kind, payload, image=None, mask=None):
        """Called by agent.restore() from the worker thread. Saves images, records the event."""
        data = dict(payload)
        tag = {"start": "0", "step": str(payload.get("step")), "question": f"q{payload.get('step')}"}.get(kind)
        if image is not None and tag:
            cv2.imwrite(self._path(f"img-{tag}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            data["image"] = self.url(f"img-{tag}.jpg")
        if mask is not None and tag:
            cv2.imwrite(self._path(f"mask-{tag}.png"), mask_overlay(mask))
            data["mask"] = self.url(f"mask-{tag}.png")
            data["mask_coverage"] = round(float((mask > 0).mean()), 5)
        self._push(kind, data)

    def stream(self, start: int = 0, heartbeat: float = 15.0):
        """Yield events from index `start` until the job finishes. None = heartbeat."""
        i = start
        while True:
            with self.cond:
                if i >= len(self.events) and not self.finished:
                    self.cond.wait(timeout=heartbeat)
                batch, finished = self.events[i:], self.finished
                i += len(batch)
            if not batch and not finished:
                yield None
            for e in batch:
                yield e
            if finished and i >= len(self.events):
                return

    # ---------- run ----------

    def _planner(self):
        if self.planner == "bedrock":
            try:
                return agent.BedrockPlanner()
            except Exception as e:  # e.g. boto3 can't build a client: fall back, and say so
                self._push("notice", {"message": f"Bedrock unavailable ({type(e).__name__}); using the heuristic planner."})
        return agent.HeuristicPlanner()

    def run(self):
        self._push("queued", {"planner": self.planner, "answers": self.answers, "parent": self.parent})
        with FULL_SLOT if self.full_res else RUN_SLOTS:
            try:
                t0 = time.perf_counter()
                restored, trace = agent.restore(self.original(), planner=self._planner(), answers=self.answers,
                                                log=lambda *_: None, on_event=self._on_event)
                trace["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                trace["job_id"], trace["parent_job"] = self.id, self.parent
                trace["size"], trace["full_res"] = [restored.shape[1], restored.shape[0]], self.full_res
                cv2.imwrite(self._path("restored.png"), restored)
                cv2.imwrite(self._path("restored.jpg"), restored, [cv2.IMWRITE_JPEG_QUALITY, 92])
                with open(self._path("trace.json"), "w") as fh:
                    json.dump(trace, fh, indent=2)
                self.trace = trace
                archived = storage.archive(self.id, self.dir)
                self._push("done", {"trace": trace, "restored": self.url("restored.jpg"), "archive": archived,
                                    "full_size": self.full_size,
                                    "restored_png": self.url("restored.png"), "original": self.url("original.jpg")})
            except Exception as e:
                traceback.print_exc()
                self._push("error", {"message": f"{type(e).__name__}: {e}"})
        with self.cond:
            self.finished = True
            self.cond.notify_all()

    def start(self):
        threading.Thread(target=self.run, daemon=True, name=f"job-{self.id}").start()
        return self


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, original, planner, answers, parent=None, full=None, full_res=False) -> Job:
        job = Job(original, planner, answers, parent, full=full, full_res=full_res)
        with self._lock:
            self._jobs[job.id] = job
            for old in sorted(self._jobs.values(), key=lambda j: j.created)[:-MAX_JOBS_KEPT]:
                if old.finished:
                    self._jobs.pop(old.id, None)
                    shutil.rmtree(old.dir, ignore_errors=True)
        return job.start()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)
