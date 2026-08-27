"""
Exact watermark removal by DE-BLENDING (not inpainting).

hoalau.xyz stamps a known logo with a plain alpha blend (see the site's
watermarkDraw.ts):  I = (1 - a)·B + a·W,  where a = opacity × logoAlpha(x,y),
B is the real photo underneath and W is the logo's colour.

Because the watermark is semi-transparent, B is still present in every pixel — so
instead of throwing those pixels away and letting LaMa hallucinate (which warps
skin/hands), we recover B algebraically:

        B = (I - a·W) / (1 - a)

Steps:
  1. Detect the neon watermark pixels (services.mask.neon_watermark_mask).
  2. Locate the logo: multi-scale template-match the logo's alpha shape against
     the detected mask to get position + scale (handles centre/corner; repeats
     are found one-by-one).
  3. Estimate the blend opacity that best flattens the region, then de-blend.
  4. Only the near-opaque logo core (a≈1, division unstable) is handed to LaMa;
     everything semi-transparent is recovered exactly, so no deformation.

Enable by placing the logo at apps/api/assets/hoalau_logo.png (RGBA) or setting
DEBLEND_LOGO. If the logo can't be located, the caller falls back to inpainting.
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("clearmark.deblend")

_LOGO_PATH = os.getenv(
    "DEBLEND_LOGO",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "hoalau_logo.png"),
)
_tmpl = None  # (W float32 HxWx3, A float32 HxW in 0..1)

MIN_MATCH = 0.55         # template-match score floor to accept a placement
A_CAP = 0.92             # above this effective alpha, de-blend is unstable → inpaint
MAX_INSTANCES = 12       # for tiled watermarks


def available() -> bool:
    return _load() is not None


def _load():
    global _tmpl
    if _tmpl is not None:
        return _tmpl if _tmpl != () else None
    try:
        im = Image.open(_LOGO_PATH).convert("RGBA")
        arr = np.asarray(im).astype(np.float32)
        W = arr[..., :3]
        A = arr[..., 3] / 255.0
        # Photoroom cutouts leave a grey anti-alias fringe; trust only solid alpha.
        A = np.clip((A - 0.15) / 0.85, 0.0, 1.0)
        _tmpl = (W, A)
        logger.info("de-blend logo loaded %s %s", _LOGO_PATH, im.size)
        return _tmpl
    except Exception as exc:  # noqa: BLE001
        logger.info("de-blend logo unavailable (%s): %s", _LOGO_PATH, exc)
        _tmpl = ()
        return None


def _logo_edges(W: np.ndarray, A: np.ndarray, tw: int, th: int) -> np.ndarray:
    lum = cv2.cvtColor(W.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lum = (lum * (A > 0.15)).astype(np.uint8)
    lum = cv2.resize(lum, (tw, th), interpolation=cv2.INTER_AREA)
    return cv2.Canny(lum, 40, 120)


def locate_all(image: Image.Image, min_score: float = 0.28):
    """
    Find every hoalau.xyz logo instance by matching its EDGE shape against the
    image across scales — robust when the colour detector is weak, and finds
    repeats (tiled watermarks). Returns a list of (score, x, y, w, h).
    """
    t = _load()
    if t is None:
        return []
    W, A = t
    lh, lw = A.shape
    aspect = lw / lh
    rgb = np.array(image.convert("RGB"))
    Hh, Ww = rgb.shape[:2]
    img_edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 40, 120).astype(np.float32)

    found: list[tuple[float, int, int, int, int]] = []
    for frac in np.linspace(0.16, 0.72, 22):
        tw = int(frac * Ww)
        th = int(tw / aspect)
        if tw < 60 or th < 16 or th >= Hh or tw >= Ww:
            continue
        tmpl = _logo_edges(W, A, tw, th).astype(np.float32)
        if tmpl.sum() < 1:
            continue
        res = cv2.matchTemplate(img_edges, tmpl, cv2.TM_CCORR_NORMED)
        # collect local peaks above threshold (supports multiple instances)
        work = res.copy()
        for _ in range(MAX_INSTANCES):
            _, maxv, _, maxloc = cv2.minMaxLoc(work)
            if maxv < min_score:
                break
            x, y = int(maxloc[0]), int(maxloc[1])
            found.append((float(maxv), x, y, tw, th))
            # suppress a neighbourhood so we don't re-pick the same spot
            cv2.rectangle(work, (x - tw // 2, y - th // 2), (x + tw // 2, y + th // 2), 0.0, -1)

    # Non-max suppress across scales: keep the highest-scoring, drop overlaps.
    found.sort(reverse=True)
    kept: list[tuple[float, int, int, int, int]] = []
    for f in found:
        s, x, y, tw, th = f
        box = (x, y, x + tw, y + th)
        if all(_iou(box, (kx, ky, kx + kw, ky + kh)) < 0.3 for _, kx, ky, kw, kh in kept):
            kept.append(f)
        if len(kept) >= MAX_INSTANCES:
            break
    return kept


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


def _locate_one(target_mask: np.ndarray, A: np.ndarray, img_w: int):
    """
    Best (score, x, y, w, h) placement of the logo over a binary target mask.

    The logo template IS the full "HL + hoalau.xyz" composition, i.e. the same
    thing the watermark draws — so its bounding box maps to the watermark's
    bounding box. We take the mask's robust bbox as the scale/position prior,
    then refine position with a local template match to lock onto the strokes.
    """
    lh, lw = A.shape
    aspect = lw / lh
    Hh, Ww = target_mask.shape

    # Use only the DOMINANT components (the watermark is the big central cluster);
    # drop scattered speckle so it doesn't inflate the bounding box.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(target_mask, connectivity=8)
    if num <= 1:
        return None
    # A watermark stroke is never nearly as tall/wide as the whole frame — drop
    # background/hair blobs that the colour rule caught along edges.
    cands = []
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        if ch > 0.55 * Hh or cw > 0.9 * Ww:
            continue
        cands.append((a, i))
    if not cands:
        return None
    biggest = max(a for a, _ in cands)
    keep = np.zeros_like(target_mask)
    for a, i in cands:
        if a >= max(80, 0.12 * biggest):
            keep[labels == i] = 255
    ys, xs = np.where(keep > 0)
    if len(xs) < 40:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0
    if bw < 30 or bh < 8:
        return None
    # Fit the logo by whichever dimension gives the larger (bounding) size so the
    # whole watermark is covered, clamped to the frame.
    tw = int(round(max(bw, bh * aspect)))
    th = int(round(tw / aspect))
    tw = min(tw, Ww)
    th = min(th, Hh)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    x = int(round(cx - tw / 2))
    y = int(round(cy - th / 2))

    # Local refine: slide the alpha template ±8% of its size to maximise overlap.
    a_bin = (cv2.resize(A, (tw, th), interpolation=cv2.INTER_AREA) > 0.2).astype(np.uint8) * 255
    best = (0.0, x, y)
    rx, ry = max(6, tw // 12), max(6, th // 12)
    x_lo, y_lo = max(0, x - rx), max(0, y - ry)
    x_hi, y_hi = min(Ww - tw, x + rx), min(Hh - th, y + ry)
    if x_hi > x_lo and y_hi > y_lo:
        win = target_mask[y_lo:y_hi + th, x_lo:x_hi + tw]
        if win.shape[0] >= th and win.shape[1] >= tw:
            res = cv2.matchTemplate(win, a_bin, cv2.TM_CCORR_NORMED)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)
            best = (float(maxv), x_lo + int(maxloc[0]), y_lo + int(maxloc[1]))
    else:
        x = max(0, min(x, Ww - tw))
        y = max(0, min(y, Hh - th))
        best = (0.6, x, y)
    return (best[0], best[1], best[2], tw, th)


def _deblend_region(region: np.ndarray, Wr: np.ndarray, Ar: np.ndarray,
                    stroke: np.ndarray) -> np.ndarray:
    """
    Recover the background B under a semi-transparent watermark in one region.

    Model per pixel: I = (1-a)·B + a·W  ⇒  B = (I - a·W)/(1 - a).
    We don't trust the cutout's alpha magnitude, so we ESTIMATE the effective
    alpha a from the image itself:
      • B0 = a smooth skin estimate (inpaint the strokes from surrounding skin);
      • given I, B0 and the logo colour W, the least-squares alpha is
            a = <I-B0, W-B0> / |W-B0|²   (clamped to [0, A_CAP]).
    Then B = (I - a·W)/(1-a) brings the real skin texture in I back, scaled by
    1/(1-a) — no hallucination, so no warped skin/limbs.
    """
    h, w = region.shape[:2]
    present = (Ar > 0.03) & (stroke > 0)
    if present.sum() < 20:
        return region
    # Smooth skin estimate underneath the strokes.
    strk = (stroke > 0).astype(np.uint8)
    strk = cv2.dilate(strk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
    bgr = cv2.cvtColor(region.astype(np.uint8), cv2.COLOR_RGB2BGR)
    B0 = cv2.cvtColor(cv2.inpaint(bgr, strk, 6, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB).astype(np.float32)

    d = Wr - B0
    num = ((region - B0) * d).sum(axis=2)
    den = (d * d).sum(axis=2) + 1.0
    a = np.clip(num / den, 0.0, A_CAP)
    a = cv2.GaussianBlur(a, (0, 0), 1.5)
    a = np.where(present, a, 0.0)[..., None]

    B = (region - a * Wr) / (1.0 - a)
    out = np.where(a > 0.01, B, region)
    return np.clip(out, 0, 255)


def aligned_mask(image: Image.Image, neon_mask: np.ndarray, min_score: float = 0.62):
    """
    Complete the removal mask using the known logo shape.

    The colour detector often misses the bright near-white neon cores. When the
    logo can be confidently aligned to the detected watermark, its alpha shape
    fills those gaps → a complete mask → clean removal. Returns the completed
    L-mode mask, or None when alignment isn't confident (caller keeps the colour
    mask). Gated by match score AND a coverage cap so a mis-alignment can't blow
    up into a giant mask.
    """
    t = _load()
    if t is None:
        return None
    _, A = t
    work = (neon_mask > 0).astype(np.uint8) * 255
    loc = _locate_one(work, A, image.width)
    if loc is None or loc[0] < min_score:
        return None
    _, x, y, tw, th = loc
    Ww, Hh = image.width, image.height
    Ar = cv2.resize(A, (tw, th), interpolation=cv2.INTER_AREA)
    aa = (Ar > 0.12).astype(np.uint8) * 255
    aa = cv2.dilate(aa, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 1)
    full = neon_mask.copy()
    full[y:y + th, x:x + tw] = np.maximum(full[y:y + th, x:x + tw], aa)
    if float((full > 0).mean()) > 0.20:  # implausible → likely mis-aligned
        return None
    logger.info("aligned mask: score=%.2f cov=%.2f%%", loc[0], 100.0 * (full > 0).mean())
    return Image.fromarray(full, mode="L")


def deblend(image: Image.Image, neon_mask: np.ndarray) -> tuple[Image.Image, np.ndarray] | None:
    """
    Try to remove the hoalau.xyz logo from ``image`` by de-blending.

    Returns (result_image, leftover_mask) on success, where leftover_mask marks
    the near-opaque core pixels the caller should still inpaint; or None if the
    logo could not be located (caller falls back to full inpainting).
    """
    t = _load()
    if t is None:
        return None
    W, A = t
    rgb = np.array(image.convert("RGB")).astype(np.float32)
    Hh, Ww = rgb.shape[:2]

    work_mask = (neon_mask > 0).astype(np.uint8) * 255
    if work_mask.max() == 0:
        return None

    out = rgb.copy()
    leftover = np.zeros((Hh, Ww), np.uint8)
    found = 0
    remaining = work_mask.copy()

    for _ in range(MAX_INSTANCES):
        loc = _locate_one(remaining, A, Ww)
        if loc is None or loc[0] < MIN_MATCH:
            break
        score, x, y, tw, th = loc
        Wr = cv2.resize(W, (tw, th), interpolation=cv2.INTER_AREA)
        Ar = cv2.resize(A, (tw, th), interpolation=cv2.INTER_AREA)
        region = out[y:y + th, x:x + tw].copy()
        stroke = work_mask[y:y + th, x:x + tw]  # actually-detected watermark pixels
        region = _deblend_region(region, Wr, Ar, stroke)
        out[y:y + th, x:x + tw] = region
        # remove this instance from the search mask so we can find repeats
        cv2.rectangle(remaining, (x, y), (x + tw, y + th), 0, -1)
        found += 1
        logger.info("de-blend instance %d: score=%.2f at (%d,%d) %dx%d",
                    found, score, x, y, tw, th)
        if remaining.max() == 0:
            break

    if found == 0:
        return None
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), leftover
