"""
Ink extraction + component routing.

The detector (and OCR/Florence boxes) return *blobs*. Inpainting a blob repaints
whatever is under it — that is what turned bikinis into smeared skin. So before
any removal we shrink every blob to the pixels that actually differ from the
local background ("ink"), then route each component to the cheapest tool that
can remove it without inventing content:

    thin   → alpha peel + tiny Telea      (text strokes, thin logos)
    solid  → LaMa, only when small        (opaque stamp / sticker)
    wide   → tint correction, never fill  (translucent bands / tiled overlays)

Nothing wide ever reaches a generative model, so a wrong mask can dim an area
but can never delete a body part.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger("clearmark.ink")

THIN_THICK_PX = 11.0      # thickness (2*distance transform) treated as a stroke
SOLID_MAX_FRAC = 0.015    # a "solid logo" may cover at most 1.5% of the image
INK_MIN_FRAC = 0.03       # below this the blob is mostly background → keep ink only
FILL_MAX_FRAC = 0.06      # no guessed blob larger than this may ever be filled
TRUSTED_FILL_MAX_FRAC = 0.04  # thick brushed / colour-detected blobs, no ink proof


def _odd(v: int, lo: int = 3, hi: int = 61) -> int:
    v = int(max(lo, min(hi, v)))
    return v if v % 2 == 1 else v + 1


def _background_estimate(crop: np.ndarray) -> np.ndarray:
    """Median filter big enough to swallow strokes but keep photo structure."""
    h, w = crop.shape[:2]
    k = _odd(min(h, w) // 3, 5, 31)
    return cv2.medianBlur(crop, k)


FLAT_SPREAD = 30.0        # colour spread that is uniform enough to be ink outright
MAX_SPREAD_RATIO = 0.85   # ink must be markedly flatter than its surroundings


def ink_of_component(
    rgb: np.ndarray,
    comp: np.ndarray,
    *,
    min_delta: float = 7.0,
    max_spread_ratio: float = MAX_SPREAD_RATIO,
    require_uniform: bool = True,
) -> np.ndarray:
    """
    Pixels inside ``comp`` that look *painted on top of* the photo.

    Background is predicted by filling the component from its own boundary, so
    the test works for both hairline glyphs and thick lettering. The candidate
    pixels then have to share one colour (a watermark is drawn in a single ink);
    real photo content inside a wrong mask is colourful and gets rejected, which
    is what makes a bad detection harmless instead of destructive.
    """
    ys, xs = np.where(comp > 0)
    if ys.size == 0:
        return np.zeros(comp.shape, np.uint8)
    pad = 18
    y0, y1 = max(0, ys.min() - pad), min(comp.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(comp.shape[1], xs.max() + 1 + pad)

    crop = rgb[y0:y1, x0:x1]
    sub = (comp[y0:y1, x0:x1] > 0).astype(np.uint8)
    grown = cv2.dilate(sub * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)

    bg = cv2.inpaint(crop, grown, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    diff = np.abs(crop.astype(np.int16) - bg.astype(np.int16)).max(axis=2).astype(np.float32)

    inside = sub > 0
    vals = diff[inside]
    if vals.size == 0:
        return np.zeros(comp.shape, np.uint8)

    thr = float(cv2.threshold(vals.astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    thr = max(min_delta, thr)
    cand = (diff >= thr) & inside
    if cand.sum() < 8:
        return np.zeros(comp.shape, np.uint8)

    if require_uniform:
        spread = float(crop[cand].astype(np.float32).std(axis=0).mean())
        ring = (
            cv2.dilate(sub * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)), 1) > 0
        ) & (~inside)
        ring_spread = (
            float(crop[ring].astype(np.float32).std(axis=0).mean()) if ring.sum() > 50 else 0.0
        )
        ratio = spread / ring_spread if ring_spread > 5.0 else None
        flat = spread <= FLAT_SPREAD or (ratio is not None and ratio <= max_spread_ratio)
        if not flat:
            logger.info(
                "reject component: spread %.1f vs ring %.1f (photo content, not ink)",
                spread, ring_spread,
            )
            return np.zeros(comp.shape, np.uint8)

    # Hysteresis: grow from the stroke core into its anti-aliased edge / glow,
    # otherwise a faint outline of the watermark survives every fill.
    weak = (diff >= max(min_delta * 0.55, thr * 0.45)) & inside
    ink_crop = _grow_into_weak(cand, weak)
    ink_crop = cv2.morphologyEx(ink_crop, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    out = np.zeros(comp.shape, np.uint8)
    out[y0:y1, x0:x1] = ink_crop
    return out


def _grow_into_weak(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Keep weak-threshold blobs that contain at least one strong pixel."""
    w = weak.astype(np.uint8)
    if w.max() == 0:
        return strong.astype(np.uint8) * 255
    n, labels = cv2.connectedComponents(w, connectivity=8)
    keep = np.unique(labels[strong])
    keep = keep[keep != 0]
    if keep.size == 0:
        return strong.astype(np.uint8) * 255
    return (np.isin(labels, keep)).astype(np.uint8) * 255


def ink_within(rgb: np.ndarray, mask: np.ndarray, **kw) -> np.ndarray:
    """Ink pixels inside an arbitrary mask, component by component."""
    m = (mask > 127).astype(np.uint8)
    out = np.zeros(mask.shape, np.uint8)
    if m.max() == 0:
        return out
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < 6:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        out = cv2.bitwise_or(out, ink_of_component(rgb, comp, **kw))
    return out


def classify(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    solid_max_frac: float = SOLID_MAX_FRAC,
    trusted: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a binary mask into (thin, solid, wide) masks.

    ``thin``  – stroke-like ink, filled with a tight Telea
    ``solid`` – compact blob, small enough for LaMa
    ``wide``  – too large to fill; only tint correction is allowed

    ``trusted`` marks masks that came from a stroke-accurate source — a user's
    brush, the colour detector, OCR glyphs. Those are taken at face value. Blobby
    guesses (segmentation nets, box detectors) must first prove there is really
    ink inside them, otherwise a wrong guess would repaint the photo.
    """
    m = (mask > 127).astype(np.uint8)
    thin = np.zeros(mask.shape, np.uint8)
    solid = np.zeros(mask.shape, np.uint8)
    wide = np.zeros(mask.shape, np.uint8)
    if m.max() == 0:
        return thin, solid, wide

    H, W = m.shape
    img_area = float(H * W)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 6:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        dist = cv2.distanceTransform(comp // 255, cv2.DIST_L2, 3)
        thick = float(dist.max() * 2.0) if dist.size else 0.0

        if thick <= THIN_THICK_PX:
            # Strokes are safe to fill at any size — a tiled watermark grid
            # spans the frame but never asks the fill to invent content.
            thin[comp > 0] = 255
            continue

        if trusted and area <= TRUSTED_FILL_MAX_FRAC * img_area:
            solid[comp > 0] = 255
            continue

        ink = ink_of_component(rgb, comp)
        ink_frac = float((ink > 0).sum()) / max(1, area)

        if ink_frac < INK_MIN_FRAC:
            # Nothing painted on top here — the mask is wrong. Touch nothing.
            logger.info("drop fat component: no ink (area=%d, frac=%.3f)", area, ink_frac)
            continue

        if ink_frac > 0.55 and area <= solid_max_frac * img_area:
            solid[comp > 0] = 255
            continue

        # Fat box whose content is really strokes (text on a plate, tiled logo).
        # A guessed blob may only be filled when it is stroke-shaped, or painted
        # in a colour the photo does not otherwise use: a dark bird on a plain
        # sky passes every uniformity test, and filling it would erase the bird.
        allow = FILL_MAX_FRAC if (trusted or _is_vivid(rgb, ink)) else 0.0
        _route_by_geometry(ink, thin, solid, wide, img_area, solid_max_frac, fill_max_frac=allow)
        wide[comp > 0] = 255

    logger.info(
        "ink classify: thin=%d solid=%d wide=%d",
        int((thin > 0).sum()),
        int((solid > 0).sum()),
        int((wide > 0).sum()),
    )
    return thin, solid, wide


A_CAP = 0.90  # above this opacity nothing of the photo survives → must be filled


def unmix(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Undo a translucent overlay instead of painting over it.

    A watermark composites as ``I = a*W + (1-a)*B``, so the photo underneath is
    still in every pixel — only damped and tinted. Estimating the ink colour
    ``W`` and the per-pixel opacity ``a`` recovers ``B = (I - a*W)/(1 - a)`` with
    its texture intact, which is the difference between a clean removal and the
    blurred patch a fill leaves behind.

    Returns ``(result, leftover)``; ``leftover`` marks near-opaque pixels where
    the division is unstable and the caller should fall back to filling.
    """
    m = (mask > 127).astype(np.uint8) * 255
    leftover = np.zeros(m.shape, np.uint8)
    if m.max() == 0:
        return rgb, leftover

    src = rgb.astype(np.float32)
    out = src.copy()
    grown = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
    base = cv2.inpaint(rgb, grown, inpaintRadius=6, flags=cv2.INPAINT_TELEA).astype(np.float32)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < 12:
            continue
        sel = labels == i
        dev = np.linalg.norm(src[sel] - base[sel], axis=1)
        if dev.size < 12:
            continue

        # Ink colour: the median of the most strongly tinted pixels of this
        # component, so a mixed anti-aliased rim does not drag the estimate.
        strong = dev >= np.percentile(dev, 75)
        if strong.sum() < 6:
            continue
        W = np.median(src[sel][strong], axis=0)

        d = W[None, :] - base[sel]
        den = (d * d).sum(axis=1) + 1.0
        a = np.clip(((src[sel] - base[sel]) * d).sum(axis=1) / den, 0.0, 1.0)

        opaque = a >= A_CAP
        a = np.clip(a, 0.0, A_CAP)[:, None]
        recovered = (src[sel] - a * W[None, :]) / (1.0 - a)
        out[sel] = np.clip(recovered, 0, 255)

        if opaque.any():
            idx = np.where(sel)
            leftover[idx[0][opaque], idx[1][opaque]] = 255

    # Feather the seam so the recovered patch does not show its outline.
    alpha = cv2.GaussianBlur((m > 0).astype(np.float32), (0, 0), 0.7)[..., None]
    out = src * (1.0 - alpha) + out * alpha
    logger.info("unmix %d px (%d left opaque)", int((m > 0).sum()), int((leftover > 0).sum()))
    return np.clip(out, 0, 255).astype(np.uint8), leftover


def _is_vivid(rgb: np.ndarray, ink: np.ndarray, *, margin: float = 35.0) -> bool:
    """True when the ink is far more saturated than the photo it sits on."""
    sel = ink > 0
    if sel.sum() < 12:
        return False
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32)
    ink_sat = float(np.median(sat[sel]))
    return ink_sat > float(np.median(sat)) + margin and ink_sat > 60.0


def _route_by_geometry(
    mask: np.ndarray,
    thin: np.ndarray,
    solid: np.ndarray,
    wide: np.ndarray,
    img_area: float,
    solid_max_frac: float,
    fill_max_frac: float = FILL_MAX_FRAC,
) -> None:
    """Send each blob of ``mask`` to the fill it can survive, in place."""
    m = (mask > 0).astype(np.uint8)
    if m.max() == 0:
        return
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 6:
            continue
        comp = (labels == i).astype(np.uint8)
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        thick = float(dist.max() * 2.0) if dist.size else 0.0
        if thick <= THIN_THICK_PX:
            # A tiled watermark grid is thin everywhere, however much of the
            # frame it spans — filling strokes never invents content.
            thin[comp > 0] = 255
        elif area > fill_max_frac * img_area:
            wide[comp > 0] = 255
        else:
            solid[comp > 0] = 255


def promote_matching_ink(
    rgb: np.ndarray, trusted: np.ndarray, guess: np.ndarray, *, tol: float = 62.0
) -> np.ndarray:
    """
    Return the guessed pixels drawn in the same ink as a confirmed watermark.

    A watermark repeats in one colour: if the colour detector locked onto
    "aigu" and a coarse box also covers a "g" of the same pink, that "g" is the
    same mark and can be removed with the same confidence — while a dark bird
    inside the very same box stays untouched.
    """
    t = (trusted > 127)
    g = (guess > 127).astype(np.uint8)
    out = np.zeros(guess.shape, np.uint8)
    if not t.any() or g.max() == 0:
        return out

    ref = np.median(rgb[t].astype(np.float32), axis=0)
    ink = ink_within(rgb, g * 255, require_uniform=False)
    if ink.max() == 0:
        return out

    # Judge colour per ink blob, not per box: one box can hold both a glyph and
    # a bird, and their average colour is neither.
    n, labels, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < 12:
            continue
        blob = labels == i
        colour = np.median(rgb[blob].astype(np.float32), axis=0)
        dist = float(np.linalg.norm(colour - ref))
        if dist <= tol:
            out[blob] = 255
            logger.info("promote guess ink: colour distance %.1f", dist)

    if out.max() == 0:
        return out
    if float((out > 0).mean()) > 0.08:
        # That much "matching ink" means the reference colour is the photo
        # itself (a pink dress, a sunset), not a watermark.
        logger.info("promotion refused: %.1f%% of the frame matched", 100.0 * (out > 0).mean())
        return np.zeros(guess.shape, np.uint8)
    return cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)


def split_by_surround_texture(
    rgb: np.ndarray, mask: np.ndarray, *, thresh: float = 9.0, ring_px: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a mask by how textured the photo around each blob is.

    A generative fill shines on texture but leaves a visible patch on a smooth
    gradient (sky, wall, skin), where plain interpolation is invisible instead.
    Returns ``(smooth, textured)``.
    """
    m = (mask > 127).astype(np.uint8)
    smooth = np.zeros(mask.shape, np.uint8)
    textured = np.zeros(mask.shape, np.uint8)
    if m.max() == 0:
        return smooth, textured

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.GaussianBlur(gray, (0, 0), 3.0)
    sd = np.sqrt(np.maximum(cv2.GaussianBlur(gray * gray, (0, 0), 3.0) - mu * mu, 0.0))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < 6:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        ring = (cv2.dilate(comp, k, 1) > 0) & (comp == 0)
        level = float(np.median(sd[ring])) if ring.sum() > 40 else 99.0
        (smooth if level < thresh else textured)[comp > 0] = 255
    return smooth, textured


def correct_tint(rgb: np.ndarray, mask: np.ndarray, *, ring_px: int = 16) -> np.ndarray:
    """
    Remove a translucent overlay by matching each region's colour statistics to
    the ring of pixels around it. Detail is untouched — no blur, no inpaint.
    """
    m = (mask > 127).astype(np.uint8)
    if m.max() == 0:
        return rgb

    out = rgb.astype(np.float32)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1))

    img_area = float(rgb.shape[0] * rgb.shape[1])
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 40 or area > 0.25 * img_area:
            continue  # huge region = bad detection; dimming it would wreck the photo
        comp = (labels == i).astype(np.uint8) * 255
        ring = (cv2.dilate(comp, k, 1) > 0) & (comp == 0)
        if ring.sum() < 60:
            continue
        sel = comp > 0

        shift = max(
            abs(float(out[..., c][sel].mean()) - float(out[..., c][ring].mean()))
            for c in range(3)
        )
        if shift < 2.5:
            continue  # no visible overlay here — leave the pixels alone

        adj = out.copy()
        for c in range(3):
            vin = out[..., c][sel]
            vr = out[..., c][ring]
            mu_i, sd_i = float(vin.mean()), float(vin.std()) + 1e-3
            mu_r, sd_r = float(vr.mean()), float(vr.std()) + 1e-3
            gain = float(np.clip(sd_r / sd_i, 0.9, 1.2))
            adj[..., c] = (out[..., c] - mu_i) * gain + mu_r

        alpha = cv2.GaussianBlur((comp > 0).astype(np.float32), (0, 0), 2.0)[..., None]
        cand = out * (1.0 - alpha) + adj * alpha
        # Refuse absurd corrections (that would repaint, not de-tint)
        if float(np.abs(cand[sel] - out[sel]).mean()) > 18.0:
            continue
        out = cand

    return np.clip(out, 0, 255).astype(np.uint8)
