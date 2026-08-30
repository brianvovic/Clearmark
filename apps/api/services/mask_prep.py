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

DEFAULT_DILATE_PX = 3


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
        px = max(1, px - 2)
    elif mode in ("pro", "aggressive", "4.0"):
        px = max(1, min(3, px - 1))
    elif mode == "smart":
        px = max(1, min(3, px - 1))
    return int(px)


def person_zone(rgb: np.ndarray, *, dilate_px: int = 8) -> np.ndarray:
    """
    Body + clothing sitting in skin holes (bikini, underwear).

    Skin-tone alone misses white fabric; those holes in the silhouette *are*
    the swimsuit. Closing them and growing a few pixels keeps LaMa off both
    skin and the garment next to it.
    """
    from services.mask import protect_mask

    skin = protect_mask(rgb)
    h, w = skin.shape[:2]
    k = max(5, int(min(h, w) * 0.03))
    if k % 2 == 0:
        k += 1
    body = cv2.morphologyEx(
        skin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    gaps = (body == 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(gaps, connectivity=8)
    if n > 1:
        edge = np.unique(
            np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        )
        holes = np.isin(labels, edge[edge != 0], invert=True) & (gaps > 0)
        body = cv2.bitwise_or(body, holes.astype(np.uint8) * 255)
    if dilate_px > 0:
        d = dilate_px * 2 + 1
        if d % 2 == 0:
            d += 1
        body = cv2.dilate(
            body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)), 1
        )
    return body


def apply_subject_guard(
    mask: np.ndarray,
    rgb: np.ndarray,
    *,
    mode: str = "smart",
) -> np.ndarray:
    """
    Prevent inpaint from erasing people:
      • Face / head boxes → always cleared from the mask
      • Body + swimsuit holes → keep only verified ink strokes (never a fat blob)
    """
    from services.mask import _get_face_cascades

    out = mask.copy()
    if out.max() == 0:
        return out

    h, w = rgb.shape[:2]
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
        if out.max() == 0:
            return out

    person = person_zone(rgb, dilate_px=8)
    person[face > 0] = 255
    on_person = (out > 0) & (person > 0)
    if not on_person.any():
        return out

    body = np.where(on_person, out, 0).astype(np.uint8)

    try:
        from services.ink import THIN_THICK_PX, ink_within

        kept = ink_within(rgb, body, min_delta=8.0, require_uniform=False)
        if kept.max():
            n, labels, stats, _ = cv2.connectedComponentsWithStats(
                (kept > 0).astype(np.uint8), connectivity=8
            )
            thin_only = np.zeros_like(kept)
            # A glyph in a bold font is thicker than a "stroke" but is still a
            # watermark, not anatomy. Keeping only thin components silently threw
            # away bold lettering on skin — the mask went empty and the mark
            # survived every later stage. Small ink components are therefore kept
            # on their size as well: a letter is tiny next to the frame, while the
            # limb-sized blobs the guard exists to stop are not.
            img_area = float(rgb.shape[0] * rgb.shape[1])
            small_ink_max = 0.015 * img_area
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < 6:
                    continue
                comp = (labels == i).astype(np.uint8)
                dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
                thick = float(dist.max() * 2.0) if dist.size else 0.0
                if thick <= THIN_THICK_PX or area <= small_ink_max:
                    thin_only[comp > 0] = 255
            kept = thin_only
    except Exception:  # noqa: BLE001
        kept = np.zeros_like(body)

    if kept.max() == 0:
        kept = np.zeros_like(body)  # no verified ink → do not fill the garment

    out = np.where(person > 0, kept, out).astype(np.uint8)
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

    rgb_arr: np.ndarray | None = None
    if rgb is not None:
        rgb_arr = np.array(rgb.convert("RGB")) if isinstance(rgb, Image.Image) else rgb
        if rgb_arr.shape[0] != binary.shape[0] or rgb_arr.shape[1] != binary.shape[1]:
            rgb_arr = cv2.resize(
                rgb_arr, (binary.shape[1], binary.shape[0]), interpolation=cv2.INTER_AREA
            )

    px = dilate_px if dilate_px is not None else smart_dilate_px(binary, mode=mode)
    if px > 0 and binary.max() > 0:
        k = max(3, int(px) * 2 + 1)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        if rgb_arr is not None:
            # Grow background blobs only — dilating on a body swallows bikini/skin
            person = person_zone(rgb_arr, dilate_px=8)
            bg = binary.copy()
            bg[person > 0] = 0
            bg = cv2.dilate(bg, kernel, iterations=1)
            body = binary.copy()
            body[person == 0] = 0
            binary = cv2.bitwise_or(bg, body)
        else:
            binary = cv2.dilate(binary, kernel, iterations=1)

    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

    if rgb_arr is not None:
        binary = apply_subject_guard(binary, rgb_arr, mode=mode)

    return Image.fromarray(binary, mode="L")
