"""
Hard-negative bank — cases where detection/removal failed on eval.

Failed pairs are saved under training/_data/hard_neg/ as:
  *.png          watermarked image
  *_mask.png     ground-truth mask
  *_clean.png    clean target (for removal)

Train loops oversample these so the model focuses on what it actually misses
instead of only easy synthetic samples.
"""

from __future__ import annotations

import logging
import os
import time
import uuid

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("clearmark.hard_neg")

_BASE = os.path.join(os.path.dirname(__file__), "_data", "hard_neg")
os.makedirs(_BASE, exist_ok=True)


def hard_neg_dir() -> str:
    return _BASE


def count() -> int:
    return sum(1 for f in os.listdir(_BASE) if f.endswith(".png") and "_mask" not in f and "_clean" not in f)


def save_case(wm_rgb: np.ndarray, gt_mask: np.ndarray, clean_rgb: np.ndarray, *, reason: str = "") -> str:
    """Persist one hard case. Returns the stem id."""
    stem = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    if reason:
        stem = f"{stem}_{reason[:24]}"
    Image.fromarray(wm_rgb).save(os.path.join(_BASE, f"{stem}.png"))
    Image.fromarray(gt_mask.astype(np.uint8)).save(os.path.join(_BASE, f"{stem}_mask.png"))
    Image.fromarray(clean_rgb).save(os.path.join(_BASE, f"{stem}_clean.png"))
    logger.info("hard-neg saved %s (%s)", stem, reason or "fail")
    return stem


def list_cases(limit: int = 500) -> list[tuple[str, str, str]]:
    """Return list of (wm_path, mask_path, clean_path), newest first."""
    stems = []
    for name in os.listdir(_BASE):
        if name.endswith(".png") and "_mask" not in name and "_clean" not in name:
            stem = name[:-4]
            mp = os.path.join(_BASE, f"{stem}_mask.png")
            cp = os.path.join(_BASE, f"{stem}_clean.png")
            wp = os.path.join(_BASE, name)
            if os.path.exists(mp) and os.path.exists(cp):
                stems.append((os.path.getmtime(wp), wp, mp, cp))
    stems.sort(reverse=True)
    return [(w, m, c) for _, w, m, c in stems[:limit]]


def load_case(wm_path: str, mask_path: str, clean_path: str, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wm = np.array(Image.open(wm_path).convert("RGB").resize((size, size)))
    clean = np.array(Image.open(clean_path).convert("RGB").resize((size, size)))
    mask = np.array(Image.open(mask_path).convert("L").resize((size, size), Image.Resampling.NEAREST))
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return wm, mask, clean


def prune(max_keep: int = 400) -> None:
    """Keep only the newest N cases to bound disk use."""
    cases = list_cases(limit=10_000)
    for wm, mask, clean in cases[max_keep:]:
        for p in (wm, mask, clean):
            try:
                os.remove(p)
            except OSError:
                pass
