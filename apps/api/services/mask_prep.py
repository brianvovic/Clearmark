"""
Prepare detector masks for Removal / LaMa.

  1) Hard binarize → only 0 or 255 (no soft alpha / gradient)
  2) Edge-aware dilate 5–12px: thin strokes get more grow so fill bites into
     clean background; fat blobs grow less to avoid smearing faces/texture.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

DEFAULT_DILATE_PX = 8


def _stroke_thickness_px(binary: np.ndarray) -> float:
    """Estimate median stroke width via distance transform on the mask."""
    if binary.max() == 0:
        return 0.0
    # Distance to background inside the mask → ~ half-thickness
    dist = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 3)
    vals = dist[binary > 0]
    if vals.size == 0:
        return 0.0
    return float(np.median(vals) * 2.0)


def smart_dilate_px(binary: np.ndarray, *, base: int = DEFAULT_DILATE_PX,
                    mode: str = "smart") -> int:
    """
    Map stroke thickness → dilate radius.
      thin text (~2–4px) → grow more (10–12)
      fat logos (~20px+) → grow less (5–6)
    ``mode``: fast=smaller, smart=default, pro/aggressive=larger.
    """
    thick = _stroke_thickness_px(binary)
    if thick <= 0:
        px = base
    elif thick < 5:
        px = 11
    elif thick < 12:
        px = 8
    else:
        px = 5
    if mode in ("fast", "safe"):
        px = max(4, px - 2)
    elif mode in ("pro", "aggressive"):
        px = min(14, px + 2)
    return int(px)


def prepare_removal_mask(
    mask: Image.Image | np.ndarray,
    *,
    size: tuple[int, int] | None = None,
    dilate_px: int | None = None,
    mode: str = "smart",
) -> Image.Image:
    """Return an L-mode mask that is strictly binary {0,255} and dilated."""
    if isinstance(mask, Image.Image):
        m = mask.convert("L")
        if size is not None and m.size != size:
            m = m.resize(size, Image.Resampling.NEAREST)
        arr = np.array(m)
    else:
        arr = mask
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        if size is not None and (arr.shape[1], arr.shape[0]) != size:
            arr = cv2.resize(arr, size, interpolation=cv2.INTER_NEAREST)

    _, binary = cv2.threshold(arr.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

    px = dilate_px if dilate_px is not None else smart_dilate_px(binary, mode=mode)
    if px > 0 and binary.max() > 0:
        k = max(3, int(px) * 2 + 1)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.dilate(binary, kernel, iterations=1)

    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary, mode="L")
