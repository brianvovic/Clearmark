"""
Full-resolution tiled inpainting.

Why this exists
---------------
The old path downscaled the *whole* image to <=2048px, ran LaMa, then upscaled
the result back. Every pixel — even far from any watermark — went through a
resize round-trip, so the entire photo came out soft. That is the "giảm chất
lượng / mất chi tiết / mờ" symptom.

This module keeps 100% of the original pixels at native resolution and only
touches the masked regions:

  1. Find each connected component of the (full-res) mask.
  2. Crop a padded window around it *at native resolution*.
  3. If the window is bigger than one LaMa tile, slide overlapping tiles across
     it, inpaint each tile, and blend the tiles with a raised-cosine (Hann)
     weight so seams disappear.
  4. Composite the filled window back — but only inside the mask, with a soft
     feathered edge. Pixels outside the mask are bit-exact original.

No global resize ever happens, so texture, skin, printed text and EXIF-safe
detail everywhere else are preserved.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from services.lama import fill_region

logger = logging.getLogger("clearmark.tiler")

TILE = 512          # LaMa is trained around this; quality is best near it
OVERLAP = 96        # px shared between neighbouring tiles for seamless blend
WINDOW_PAD = 48     # context padding around a component before tiling
MAX_COVERAGE = 0.12  # only small solid logos reach here; bigger = bad detection
MAX_WINDOWS = 24    # cap distinct inpaint windows so CPU latency stays bounded


def _hann_2d(h: int, w: int) -> np.ndarray:
    """Separable raised-cosine window, >0 everywhere so weights never divide by 0."""
    wy = np.hanning(max(h, 3))[:h] if h >= 3 else np.ones(h)
    wx = np.hanning(max(w, 3))[:w] if w >= 3 else np.ones(w)
    win = np.outer(wy, wx).astype(np.float32)
    return np.clip(win, 1e-3, 1.0)


def _fill_window(rgb: np.ndarray, mask_bin: np.ndarray) -> np.ndarray:
    """Inpaint one cropped window, tiling if it is larger than a single tile."""
    h, w = rgb.shape[:2]
    if h <= TILE and w <= TILE:
        return fill_region(rgb, mask_bin)

    step = TILE - OVERLAP
    acc = np.zeros((h, w, 3), dtype=np.float32)
    wsum = np.zeros((h, w, 1), dtype=np.float32)

    ys = list(range(0, max(1, h - OVERLAP), step))
    xs = list(range(0, max(1, w - OVERLAP), step))
    if ys[-1] + TILE < h:
        ys.append(h - TILE)
    if xs[-1] + TILE < w:
        xs.append(w - TILE)

    for y in ys:
        for x in xs:
            y0, x0 = max(0, y), max(0, x)
            y1, x1 = min(h, y0 + TILE), min(w, x0 + TILE)
            tile_mask = mask_bin[y0:y1, x0:x1]
            th, tw = y1 - y0, x1 - x0
            win = _hann_2d(th, tw)[..., None]
            if tile_mask.max() == 0:
                # Nothing to remove here — keep original, still weight it in.
                acc[y0:y1, x0:x1] += rgb[y0:y1, x0:x1].astype(np.float32) * win
                wsum[y0:y1, x0:x1] += win
                continue
            filled = fill_region(rgb[y0:y1, x0:x1].copy(), tile_mask)
            acc[y0:y1, x0:x1] += filled.astype(np.float32) * win
            wsum[y0:y1, x0:x1] += win

    blended = (acc / wsum).clip(0, 255).astype(np.uint8)
    # Keep original outside the mask exactly (blend only fills the holes).
    out = rgb.copy()
    sel = mask_bin > 0
    out[sel] = blended[sel]
    return out


def inpaint_fullres(original: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Remove everything under ``mask`` from ``original`` at native resolution.

    ``mask``: L-mode, white = remove. Any size (it is matched to the original).
    Returns a full-resolution RGB image; every non-masked pixel is untouched.
    """
    rgb = np.array(original.convert("RGB"))
    H, W = rgb.shape[:2]

    m = mask.convert("L")
    if m.size != (W, H):
        m = m.resize((W, H), Image.Resampling.NEAREST)
    _, mask_bin = cv2.threshold(np.array(m), 127, 255, cv2.THRESH_BINARY)

    coverage = float((mask_bin > 0).mean())
    logger.info("fullres inpaint coverage=%.4f%% size=%dx%d", 100.0 * coverage, W, H)
    if coverage < 1e-6:
        raise ValueError("Mask trống — không có vùng để xóa")
    if coverage > MAX_COVERAGE:
        raise ValueError(
            "Vùng xóa quá lớn (%.1f%% ảnh). Có thể phát hiện sai — hãy dùng Thủ công "
            "và tô đúng watermark/logo." % (100.0 * coverage)
        )

    # Merge components that sit within one tile-overlap of each other so a
    # cluster of nearby strokes becomes a single window instead of dozens of
    # separate LaMa calls (bounds worst-case latency and gives more context).
    merge = cv2.dilate(
        mask_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OVERLAP | 1, OVERLAP | 1)), 1
    )
    num, labels, stats, _ = cv2.connectedComponentsWithStats(merge, connectivity=8)
    # labels index the *merged* blobs; the real hole is still mask_bin within them.
    order = sorted(range(1, num), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)
    order = order[:MAX_WINDOWS]

    out = rgb.copy()
    for i in order:
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        x0 = max(0, x - WINDOW_PAD)
        y0 = max(0, y - WINDOW_PAD)
        x1 = min(W, x + cw + WINDOW_PAD)
        y1 = min(H, y + ch + WINDOW_PAD)

        sub_rgb = out[y0:y1, x0:x1].copy()
        # The real hole is the *original* mask restricted to this merged blob —
        # not the dilated blob itself (dilation was only for grouping).
        blob = labels[y0:y1, x0:x1] == i
        sub_mask = ((mask_bin[y0:y1, x0:x1] > 0) & blob).astype(np.uint8) * 255
        if sub_mask.max() == 0:
            continue
        filled = _fill_window(sub_rgb, sub_mask)
        # Write back only the masked pixels (soft edge handled inside fill).
        sel = sub_mask > 0
        region = out[y0:y1, x0:x1]
        region[sel] = filled[sel]
        out[y0:y1, x0:x1] = region

    return Image.fromarray(out)
