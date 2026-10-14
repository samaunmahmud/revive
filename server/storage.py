"""Durable archive of finished jobs in Amazon S3.

Enabled when REVIVE_S3_BUCKET is set (the CloudFormation stack sets it).
The app keeps serving files from local disk for speed; S3 holds the original,
the restored photo and the full agent trace for every restoration.
Uploads are best-effort: a storage hiccup never fails a restoration.
"""
import logging
import os

log = logging.getLogger("revive.storage")

BUCKET = os.environ.get("REVIVE_S3_BUCKET", "")
_client = None

CONTENT_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".json": "application/json"}


def enabled() -> bool:
    return bool(BUCKET)


def describe() -> str:
    return f"s3://{BUCKET}" if BUCKET else "local disk"


def _s3():
    global _client
    if _client is None:
        import boto3
        _client = boto3.client("s3")
    return _client


def archive(job_id: str, job_dir: str, names=("original.jpg", "restored.png", "trace.json")) -> str | None:
    """Upload a finished job's files. Returns the S3 prefix, or None if disabled/failed."""
    if not BUCKET:
        return None
    prefix = f"jobs/{job_id}/"
    try:
        for name in names:
            path = os.path.join(job_dir, name)
            if os.path.isfile(path):
                ext = os.path.splitext(name)[1]
                _s3().upload_file(path, BUCKET, prefix + name,
                                  ExtraArgs={"ContentType": CONTENT_TYPES.get(ext, "application/octet-stream")})
        return f"s3://{BUCKET}/{prefix}"
    except Exception:
        log.exception("S3 archive failed for job %s", job_id)
        return None
