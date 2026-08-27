"""
Async video jobs — upload → task → poll, the model the dewatermark.ai teardown
confirmed (video needs GPU minutes, so it can't be a blocking request).

A job moves through:  CREATED → QUEUED → PROCESSING (progress 0→1) → DONE | FAILED

Processing runs in a background thread that hands the clip to the GPU worker's
ProPainter endpoint (temporal-consistent inpainting with one static mask for the
whole clip). Without a GPU worker there is no sane CPU path for video — the task
fails fast with a clear message rather than pretending. Inputs/outputs live in
storage.py with the same 1-hour TTL; failed tasks refund nothing to bill here,
but the status carries the reason.

This is an in-process registry (fine for a single self-host box). Swap for Redis
+ RQ when you run multiple API workers; the route code only touches the functions
below.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from services import storage

logger = logging.getLogger("clearmark.video")

TTL_SECONDS = 60 * 60
MAX_TASKS = 100


@dataclass
class Task:
    id: str
    status: str = "CREATED"          # CREATED|QUEUED|PROCESSING|DONE|FAILED
    progress: float = 0.0
    has_watermark: Optional[bool] = None
    input_key: Optional[str] = None
    mask_key: Optional[str] = None
    output_key: Optional[str] = None
    error: Optional[str] = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {
            "task_id": self.id,
            "status": self.status,
            "progress": round(self.progress, 3),
            "has_watermark": self.has_watermark,
            "download_url": (
                f"/api/video/tasks/{self.id}/download" if self.status == "DONE" else None
            ),
            "error": self.error,
        }


_tasks: dict[str, Task] = {}
_lock = threading.Lock()


def _sweep_locked() -> None:
    now = time.time()
    for k in [k for k, t in _tasks.items() if now - t.updated > TTL_SECONDS]:
        t = _tasks.pop(k, None)
        if t:
            for key in (t.input_key, t.mask_key, t.output_key):
                if key:
                    storage.delete(key)


def create(input_bytes: bytes, ext: str = "mp4") -> Task:
    tid = uuid.uuid4().hex
    key = storage.new_key(ext)
    storage.put(key, input_bytes, "video/mp4")
    with _lock:
        _sweep_locked()
        if len(_tasks) >= MAX_TASKS:
            oldest = min(_tasks.items(), key=lambda kv: kv[1].updated)[0]
            _tasks.pop(oldest, None)
        t = Task(id=tid, status="CREATED", input_key=key)
        _tasks[tid] = t
    return t


def get(tid: str) -> Optional[Task]:
    with _lock:
        return _tasks.get(tid)


def _update(tid: str, **kw) -> None:
    with _lock:
        t = _tasks.get(tid)
        if not t:
            return
        for k, v in kw.items():
            setattr(t, k, v)
        t.updated = time.time()


def start(tid: str, mask_bytes: Optional[bytes]) -> None:
    """Queue processing on a background thread."""
    t = get(tid)
    if not t or not t.input_key:
        raise KeyError(tid)
    mask_key = None
    if mask_bytes:
        mask_key = storage.new_key("png")
        storage.put(mask_key, mask_bytes, "image/png")
    _update(tid, status="QUEUED", mask_key=mask_key, progress=0.0)
    threading.Thread(target=_run, args=(tid,), daemon=True).start()


def _run(tid: str) -> None:
    from services import engine

    t = get(tid)
    if not t or not t.input_key:
        return
    if not engine.worker_url():
        _update(
            tid,
            status="FAILED",
            error="Xử lý video cần GPU worker (đặt GPU_WORKER_URL). CPU không đủ cho ProPainter.",
        )
        return
    try:
        _update(tid, status="PROCESSING", progress=0.05)
        video = storage.get(t.input_key)
        mask = storage.get(t.mask_key) if t.mask_key else None
        out_bytes, has_wm = _call_worker_video(video, mask)
        out_key = storage.new_key("mp4")
        storage.put(out_key, out_bytes, "video/mp4")
        _update(tid, status="DONE", progress=1.0, output_key=out_key, has_watermark=has_wm)
    except Exception as exc:  # noqa: BLE001
        logger.exception("video task %s failed", tid)
        _update(tid, status="FAILED", error=str(exc)[:300])


def _call_worker_video(video: bytes, mask: Optional[bytes]) -> tuple[bytes, Optional[bool]]:
    import httpx

    from services import engine

    url = engine.worker_url()
    files = {"video": ("in.mp4", video, "video/mp4")}
    if mask:
        files["mask"] = ("mask.png", mask, "image/png")
    with httpx.Client(timeout=float(os.getenv("VIDEO_TIMEOUT", "1800"))) as client:
        resp = client.post(
            url.rstrip("/") + "/video",
            files=files,
            headers=engine._headers(),  # reuse the same bearer auth
        )
        resp.raise_for_status()
        has_wm = resp.headers.get("X-Has-Watermark")
        return resp.content, (None if has_wm is None else has_wm == "1")
