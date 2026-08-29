"""
Prepare detector masks for Removal / LaMa.

  1) Hard binarize → only 0 or 255
  2) Conservative edge-aware dilate (small — big dilate eats skin/clothes)
  3) Subject guard: never touch faces; on body skin, shrink mask halo so
     only the watermark core is filled (keeps bikini / limbs intact)
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

DEFAULT_DILATE_PX = 4


def _stroke_thickness_px(binary: np.ndarray) -> float:
    if binary.max() == 0:
        return 0.0
    dist = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 3)
    vals = dist[binary > 0]
    if vals.size == 0:
        return 0.0
    return float(np.median(vals) * 2.0)


def smart_dilate_px(binary: np.ndarray, *, base: int = DEFAULT_DILATE_PX,
                    mode: str = "smart") -> int:
    """
    Conservative grow — thin text needs a little fringe, fat logos almost none.
    SDXL/pro uses the smallest grow so diffusion does not invent missing limbs.
    """
    thick = _stroke_thickness_px(binary)
    if thick <= 0:
        px = base
    elif thick < 5:
        px = 5          # was 11 — that ate clothing
    elif thick < 12:
        px = 4
    else:
        px = 3

    if mode in ("fast", "safe"):
        px = max(2, px - 1)
    elif mode in ("pro", "aggressive", "4.0"):
        px = max(2, min(4, px - 1))  # tightest for diffusion
    elif mode == "smart":
        px = max(2, min(5, px))
    return int(px)


def apply_subject_guard(
    mask: np.ndarray,
    rgb: np.ndarray,
    *,
    mode: str = "smart",
) -> np.ndarray:
    """
    Prevent inpaint from erasing people:
      • Face / head boxes → always cleared from the mask
      • Body skin → keep only a slightly eroded core (drop dilate halo on skin)
    Watermark text sitting ON skin is still removed (core stays); clothing next
    to skin is no longer swallowed by a fat halo.
    """
    from services.mask import protect_mask, _get_face_cascades

    out = mask.copy()
    if out.max() == 0:
        return out

    h, w = rgb.shape[:2]
    # --- Hard face protect ---
    face = np.zeros((h, w), np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    for c in _get_face_cascades():
        try:
            faces = c.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=5,
                minSize=(max(24, w // 25), max(24, h // 25)),
            )
        except Exception:  # noqa: BLE001
            continue
        for (x, y, fw, fh) in faces:
            ex, ey = int(fw * 0.5), int(fh * 0.65)
            cv2.rectangle(
                face,
                (max(0, x - ex), max(0, y - ey)),
                (min(w, x + fw + ex), min(h, y + fh + ey)),
                255, -1,
            )
    if face.max():
        out[face > 0] = 0

    # --- Soft body-skin: shrink halo ---
    keep = protect_mask(rgb)
    skin = keep.copy()
    skin[face > 0] = 0  # face already handled
    if skin.max() == 0 or out.max() == 0:
        return out

    on_skin = (out > 0) & (skin > 0)
    if not on_skin.any():
        return out

    # Erode mask 1–2 iterations so dilate fringe on skin/clothes is pulled back
    iters = 2 if mode in ("pro", "smart", "4.0") else 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(out, k, iterations=iters)
    # Outside skin: keep dilated mask; on skin: only eroded core
    out = np.where(skin > 0, eroded, out).astype(np.uint8)
    _, out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
    return out


def prepare_removal_mask(
    mask: Image.Image | np.ndarray,
    *,
    size: tuple[int, int] | None = None,
    dilate_px: int | None = None,
    mode: str = "smart",
    rgb: np.ndarray | Image.Image | None = None,
) -> Image.Image:
    """Binary {0,255} + conservative dilate + optional subject guard."""
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

    if rgb is not None:
        if isinstance(rgb, Image.Image):
            rgb = np.array(rgb.convert("RGB"))
        if rgb.shape[0] != binary.shape[0] or rgb.shape[1] != binary.shape[1]:
            rgb = cv2.resize(rgb, (binary.shape[1], binary.shape[0]), interpolation=cv2.INTER_AREA)
        binary = apply_subject_guard(binary, rgb, mode=mode)

    return Image.fromarray(binary, mode="L")
