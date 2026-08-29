"""
Person / body region for safe watermark routing.

mask_body  = face + skin + modest dilate (covers bikini adjacent to skin)
mask_bg    = everything else

Watermark ∩ body → peel / deblend only (never LaMa / SDXL).
Watermark ∩ bg   → LaMa / SDXL allowed.
"""

from __future__ import annotations

import cv2
import numpy as np

from services.mask import protect_mask


def body_mask(rgb: np.ndarray, *, dilate_px: int = 18) -> np.ndarray:
    """
    uint8 {0,255} body zone. ``dilate_px`` grows skin/face so swimwear next to
    skin is included without swallowing the whole frame.
    """
    keep = protect_mask(rgb)
    if keep.max() == 0:
        return keep
    k = max(3, int(dilate_px) * 2 + 1)
    if k % 2 == 0:
        k += 1
    # Cap dilate so we don't mark half the photo as "body"
    k = min(k, 37)
    return cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), 1)


def split_watermark_mask(
    wm: np.ndarray, body: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (on_body, on_bg) binary masks from a watermark mask."""
    m = wm > 127
    b = body > 0
    on_body = np.zeros_like(wm, dtype=np.uint8)
    on_bg = np.zeros_like(wm, dtype=np.uint8)
    on_body[m & b] = 255
    on_bg[m & ~b] = 255
    return on_body, on_bg


def thin_strokes_on_body(
    mask: np.ndarray,
    body: np.ndarray,
    *,
    max_thick: float = 16.0,
    max_body_cov: float = 0.06,
) -> np.ndarray:
    """
    On body pixels: keep only thin stroke components (text/neon).
    Fat blobs that paint over skin/clothes are dropped entirely.
    """
    m = (mask > 127).astype(np.uint8)
    if m.max() == 0:
        return m

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    body_area = max(1, int((body > 0).sum()))

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 10:
            continue
        comp = (labels == i).astype(np.uint8)
        on_b = comp & (body > 0).astype(np.uint8)
        body_frac = float(on_b.sum()) / max(1, area)

        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        thick = float(dist.max() * 2)

        if body_frac > 0.4:
            # Mostly on body → must be thin stroke
            if thick > max_thick:
                continue
            if on_b.sum() / body_area > max_body_cov:
                continue
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                peri = float(cv2.arcLength(cnts[0], True))
                # Filled blob ≈ circular (low peri²/area); text strokes are elongated
                if peri > 0 and (peri * peri) / (4 * np.pi * max(area, 1)) < 1.8 and thick > 6:
                    continue
        else:
            if thick > max_thick * 2.2 and area > 800:
                continue

        out[comp > 0] = 255
    return out
