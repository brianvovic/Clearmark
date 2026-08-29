"""
Person / body region for safe watermark routing.

mask_body = face + skin + modest dilate (bikini-adjacent)
mask_bg   = rest of the image

Body ∩ watermark → strong peel, optional thin Telea if residual remains.
Bg ∩ watermark   → full LaMa / SDXL (never drop these components).
"""

from __future__ import annotations

import cv2
import numpy as np

from services.mask import protect_mask


def body_mask(rgb: np.ndarray, *, dilate_px: int = 10) -> np.ndarray:
    """Tighter body zone — previous dilate swallowed half the frame as 'body'."""
    keep = protect_mask(rgb)
    if keep.max() == 0:
        return keep
    k = max(3, int(dilate_px) * 2 + 1)
    if k % 2 == 0:
        k += 1
    k = min(k, 25)  # hard cap ~12px radius
    return cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), 1)


def split_watermark_mask(
    wm: np.ndarray, body: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    m = wm > 127
    b = body > 0
    on_body = np.zeros_like(wm, dtype=np.uint8)
    on_bg = np.zeros_like(wm, dtype=np.uint8)
    on_body[m & b] = 255
    on_bg[m & ~b] = 255
    return on_body, on_bg


def refine_body_mask(mask: np.ndarray, body: np.ndarray) -> np.ndarray:
    """
    For the BODY portion only: prefer thinner strokes.
    Background pixels in ``mask`` are returned UNCHANGED (never dropped).
    """
    m = (mask > 127).astype(np.uint8) * 255
    if m.max() == 0:
        return m

    on_body, on_bg = split_watermark_mask(m, body)
    # Background: keep as-is (logos, stamps, banners must reach LaMa)
    out = on_bg.copy()

    if on_body.max() == 0:
        return out

    # Body: open lightly to kill speckles, keep components that look like strokes
    body_part = cv2.morphologyEx(on_body, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if body_part.max() == 0:
        body_part = on_body

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (body_part > 0).astype(np.uint8), connectivity=8
    )
    body_area = max(1, int((body > 0).sum()))
    kept = np.zeros_like(body_part)

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 8:
            continue
        # Drop only absurd body-covering blobs (>12% of body)
        if area / body_area > 0.12:
            continue
        kept[labels == i] = 255

    # If everything was dropped, fall back to original body mask (better weak peel than nothing)
    if kept.max() == 0:
        kept = on_body

    out = cv2.bitwise_or(out, kept)
    return out
