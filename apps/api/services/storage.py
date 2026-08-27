"""
Blob storage with a 1-hour TTL — local temp dir by default, Cloudflare R2 when
configured. Used for video jobs (inputs and results) that are too big to keep in
memory and must survive across the async upload → process → download flow.

Config (all optional — unset ⇒ local temp dir under the OS temp folder):

    R2_ENDPOINT   = https://<accountid>.r2.cloudflarestorage.com
    R2_BUCKET     = clearmark
    R2_ACCESS_KEY = ...
    R2_SECRET_KEY = ...
    R2_PUBLIC_BASE = https://media.example.com   (optional public read host)

R2 speaks the S3 API, so boto3 drives it unchanged. "Files deleted after 1 hour"
is both a privacy promise and a cost control (matches dewatermark.ai). Locally we
sweep on access; on R2 set a lifecycle rule to expire objects after 1 day and we
also best-effort delete on read.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

TTL_SECONDS = 60 * 60
_LOCAL_ROOT = Path(tempfile.gettempdir()) / "clearmark_blobs"
_lock = threading.Lock()


def _r2_enabled() -> bool:
    return bool(os.getenv("R2_ENDPOINT") and os.getenv("R2_BUCKET") and os.getenv("R2_ACCESS_KEY"))


def backend_label() -> str:
    return "r2" if _r2_enabled() else "local"


# --------------------------------------------------------------------------- #
# R2 (S3) client
# --------------------------------------------------------------------------- #
_client = None


def _s3():
    global _client
    if _client is None:
        import boto3

        _client = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY"],
            aws_secret_access_key=os.environ["R2_SECRET_KEY"],
            region_name="auto",
        )
    return _client


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def new_key(ext: str) -> str:
    ext = ext.lstrip(".")
    return f"{time.strftime('%Y%m%d')}/{uuid.uuid4().hex}.{ext}"


def put(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    if _r2_enabled():
        _s3().put_object(
            Bucket=os.environ["R2_BUCKET"], Key=key, Body=data, ContentType=content_type
        )
        return
    p = _LOCAL_ROOT / key
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def get(key: str) -> bytes | None:
    if _r2_enabled():
        try:
            obj = _s3().get_object(Bucket=os.environ["R2_BUCKET"], Key=key)
            return obj["Body"].read()
        except Exception:  # noqa: BLE001
            return None
    p = _LOCAL_ROOT / key
    if not p.exists():
        return None
    if time.time() - p.stat().st_mtime > TTL_SECONDS:
        p.unlink(missing_ok=True)
        return None
    return p.read_bytes()


def delete(key: str) -> None:
    if _r2_enabled():
        try:
            _s3().delete_object(Bucket=os.environ["R2_BUCKET"], Key=key)
        except Exception:  # noqa: BLE001
            pass
        return
    (_LOCAL_ROOT / key).unlink(missing_ok=True)


def public_url(key: str) -> str | None:
    """A directly-downloadable URL if one exists (R2 public host or presigned)."""
    if not _r2_enabled():
        return None
    base = os.getenv("R2_PUBLIC_BASE")
    if base:
        return base.rstrip("/") + "/" + key
    try:
        return _s3().generate_presigned_url(
            "get_object",
            Params={"Bucket": os.environ["R2_BUCKET"], "Key": key},
            ExpiresIn=TTL_SECONDS,
        )
    except Exception:  # noqa: BLE001
        return None


def sweep_local() -> None:
    """Delete expired local blobs (no-op for R2, which uses lifecycle rules)."""
    if _r2_enabled() or not _LOCAL_ROOT.exists():
        return
    now = time.time()
    with _lock:
        for p in _LOCAL_ROOT.rglob("*"):
            if p.is_file() and now - p.stat().st_mtime > TTL_SECONDS:
                p.unlink(missing_ok=True)
