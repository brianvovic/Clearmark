"""
Editing sessions with an accumulated mask.

The quality bug this fixes
--------------------------
If a user erases, sees a leftover, and erases again, the naive approach feeds
the *previous output* (a re-encoded JPEG/PNG that was already inpainted) back
into the model. Each round re-compresses and inpaints on top of inpaint →
cumulative blur. dewatermark.ai avoids this by keeping the full-res ORIGINAL on
the server and accumulating the mask; every refine re-runs from the pristine
original with mask_1 ∪ mask_2 ∪ … So detail never degrades across refines.

This is an in-process store (thread-safe, TTL-evicted) with the original bytes
kept on disk in a temp dir. It is deliberately swappable: Phase 2 can back it
with Redis + object storage without changing the route code, which only calls
the functions below.
"""

from __future__ import annotations

import io
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

TTL_SECONDS = 60 * 60          # match "files deleted after 1 hour"
MAX_SESSIONS = 200             # simple guard for a self-host box
_SWEEP_EVERY = 120


@dataclass
class _Session:
    original: bytes                       # pristine full-res upload bytes
    size: tuple[int, int]                 # (W, H)
    mask: Optional[np.ndarray] = None     # accumulated uint8 {0,255}, HxW
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)


_store: dict[str, _Session] = {}
_lock = threading.Lock()
_last_sweep = 0.0


def _sweep_locked() -> None:
    global _last_sweep
    now = time.time()
    if now - _last_sweep < _SWEEP_EVERY:
        return
    _last_sweep = now
    dead = [k for k, s in _store.items() if now - s.touched > TTL_SECONDS]
    for k in dead:
        _store.pop(k, None)


def create(original: bytes, size: tuple[int, int]) -> str:
    sid = uuid.uuid4().hex
    with _lock:
        _sweep_locked()
        if len(_store) >= MAX_SESSIONS:
            # drop the oldest to make room
            oldest = min(_store.items(), key=lambda kv: kv[1].touched)[0]
            _store.pop(oldest, None)
        _store[sid] = _Session(original=original, size=size)
    return sid


def _get(sid: str) -> _Session:
    with _lock:
        s = _store.get(sid)
        if s is None:
            raise KeyError(sid)
        s.touched = time.time()
        return s


def original_image(sid: str) -> Image.Image:
    s = _get(sid)
    img = Image.open(io.BytesIO(s.original))
    img.load()
    return img.convert("RGB")


def size(sid: str) -> tuple[int, int]:
    return _get(sid).size


def accumulate(sid: str, mask_bin: np.ndarray) -> np.ndarray:
    """OR a new mask (uint8 {0,255}, full-res) into the accumulated mask."""
    s = _get(sid)
    with _lock:
        if s.mask is None:
            s.mask = (mask_bin > 0).astype(np.uint8) * 255
        else:
            s.mask = np.maximum(s.mask, (mask_bin > 0).astype(np.uint8) * 255)
        s.touched = time.time()
        return s.mask.copy()


def get_mask(sid: str) -> Optional[np.ndarray]:
    m = _get(sid).mask
    return None if m is None else m.copy()


def reset_mask(sid: str) -> None:
    s = _get(sid)
    with _lock:
        s.mask = None
        s.touched = time.time()


def drop(sid: str) -> None:
    with _lock:
        _store.pop(sid, None)
