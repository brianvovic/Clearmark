"""
Strong peel / attenuate for watermarks on skin & clothing + optional thin fill.

Peel keeps high-frequency texture. If ink still remains, ``thin_fill`` runs a
small-radius Telea on the stroke core only (not a fat blob / not SDXL).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _stroke_core(mask: np.ndarray, grow: int = 2) -> np.ndarray:
    m = (mask > 127).astype(np.uint8) * 255
    if m.max() == 0:
        return m
    core = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if core.max() == 0:
        core = m
    if grow > 0:
        k = grow * 2 + 1
        core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), 1)
    return core


def peel_overlay(rgb: np.ndarray, mask: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
    """
    Aggressive alpha peel. ``strength`` 0.7–1.35 (smart/pro use ≥1.0).
    Outside mask is bit-exact original.
    """
    core = _stroke_core(mask, grow=2)
    if core.max() == 0:
        return rgb

    orig = rgb.astype(np.float32)
    B = cv2.inpaint(rgb, core, inpaintRadius=3, flags=cv2.INPAINT_TELEA).astype(np.float32)

    delta = np.abs(orig - B).mean(axis=2, keepdims=True)
    # Stronger alpha — previous /32 left gaigu almost untouched
    a = np.clip(delta / 18.0, 0.15, 0.95) * float(strength)
    a = np.clip(a, 0.0, 0.98)
    core3 = (core > 0).astype(np.float32)[..., None]
    a = a * core3
    a = cv2.GaussianBlur(a, (0, 0), 0.5)
    if a.ndim == 2:
        a = a[..., None]

    blur_o = cv2.GaussianBlur(orig, (0, 0), 1.2)
    blur_b = cv2.GaussianBlur(B, (0, 0), 1.2)
    detail = orig - blur_o
    # Mix more toward clean low-freq; keep detail for pores/fabric
    low = blur_o * (1.0 - a) + blur_b * a
    peeled = np.clip(low + detail * (1.0 - 0.35 * a), 0, 255)

    w = (core > 0).astype(np.float32)
    w = cv2.GaussianBlur(w, (0, 0), 0.45)
    if w.ndim == 2:
        w = w[..., None]
    # Force almost full replace under core when strength high
    w = np.clip(w * (0.55 + 0.45 * float(strength)), 0, 1)
    out = orig * (1.0 - w) + peeled * w
    return np.clip(out, 0, 255).astype(np.uint8)


def erase_ink(rgb: np.ndarray, mask: np.ndarray, *, radius: int = 3) -> np.ndarray:
    """
    Structure-aware fill of the ink pixels themselves.

    Only the (1px grown) strokes are replaced, with a sub-pixel feather at the
    edge. Nothing outside is blurred, so a stroke disappears without softening
    the photo around it.
    """
    core = (mask > 127).astype(np.uint8) * 255
    if core.max() == 0:
        return rgb
    core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
    filled = cv2.inpaint(rgb, core, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    a = cv2.GaussianBlur((core > 0).astype(np.float32), (0, 0), 0.6)[..., None]
    out = rgb.astype(np.float32) * (1.0 - a) + filled.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def smooth_fill(rgb: np.ndarray, mask: np.ndarray, *, grow: int = 3) -> np.ndarray:
    """
    Fill a blob sitting on a smooth backdrop (sky, wall, out-of-focus bokeh).

    Telea traces visible fan-shaped streaks across areas this large. Inpainting
    a downscaled copy and scaling the patch back up interpolates the gradient
    instead, which is all a smooth backdrop actually contains.
    """
    m = (mask > 127).astype(np.uint8) * 255
    if m.max() == 0:
        return rgb
    k = max(3, grow * 2 + 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), 1)

    h, w = rgb.shape[:2]
    scale = max(2, int(round(min(h, w) / 128.0)))
    sh, sw = max(16, h // scale), max(16, w // scale)
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    small_m = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST)
    small_m = cv2.dilate(small_m, np.ones((3, 3), np.uint8), 1)
    if small_m.max() == 0:
        return erase_ink(rgb, mask, radius=5)

    filled = cv2.inpaint(small, small_m, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    up = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)

    a = cv2.GaussianBlur((m > 0).astype(np.float32), (0, 0), 2.0)[..., None]
    out = rgb.astype(np.float32) * (1.0 - a) + up * a
    return np.clip(out, 0, 255).astype(np.uint8)


def thin_fill(rgb: np.ndarray, mask: np.ndarray, *, radius: int = 3) -> np.ndarray:
    """Backwards-compatible alias of :func:`erase_ink`."""
    return erase_ink(rgb, mask, radius=radius)


def residual_strokes(
    rgb_after: np.ndarray,
    seed_mask: np.ndarray | None = None,
    *,
    min_delta: float = 8.0,
) -> np.ndarray:
    """Leftover ink near the original watermark region after peel."""
    try:
        from services.mask import neon_watermark_mask

        neon = neon_watermark_mask(rgb_after)
    except Exception:  # noqa: BLE001
        neon = np.zeros(rgb_after.shape[:2], np.uint8)

    gray = cv2.cvtColor(rgb_after, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local = cv2.GaussianBlur(gray, (0, 0), 2.5)
    hi = (np.abs(gray - local) > min_delta).astype(np.uint8) * 255
    hi = cv2.morphologyEx(hi, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    residual = cv2.bitwise_or(neon, hi)
    if seed_mask is not None and seed_mask.max():
        near = cv2.dilate(
            (seed_mask > 127).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            1,
        )
        residual = cv2.bitwise_and(residual, near)
    return residual


def peel_image(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    m = np.asarray(mask.convert("L"))
    if m.shape[:2] != rgb.shape[:2]:
        m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(peel_overlay(rgb, m))
