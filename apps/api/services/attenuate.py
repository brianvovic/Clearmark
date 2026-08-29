"""
Peel / attenuate translucent watermarks on skin & clothing.

Never uses LaMa or SDXL. Pipeline:
  1) Restrict to thin stroke core
  2) Estimate local background B and alpha a
  3) Unblend: B ≈ (I - a·W) / (1-a) with W≈I (ink colour ≈ observed)
  4) Re-inject high-frequency detail from the original (pores / fabric)
  5) residual_strokes() — thin leftover ink after peel for an optional 2nd peel
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _stroke_core(mask: np.ndarray) -> np.ndarray:
    m = (mask > 127).astype(np.uint8) * 255
    if m.max() == 0:
        return m
    core = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if core.max() == 0:
        core = m
    return cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)


def peel_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Alpha-aware peel; outside mask is bit-exact original."""
    core = _stroke_core(mask)
    if core.max() == 0:
        return rgb

    orig = rgb.astype(np.float32)
    # Local background via small Telea — used only as B estimate, never as final
    B = cv2.inpaint(rgb, core, inpaintRadius=2, flags=cv2.INPAINT_TELEA).astype(np.float32)

    # Alpha from colour distance to background (translucent ink lifts chroma/luma)
    delta = np.abs(orig - B).mean(axis=2, keepdims=True)
    a = np.clip(delta / 32.0, 0.0, 0.85)
    core3 = (core > 0).astype(np.float32)[..., None]
    a = a * core3
    a = cv2.GaussianBlur(a, (0, 0), 0.6)
    if a.ndim == 2:
        a = a[..., None]

    blur_o = cv2.GaussianBlur(orig, (0, 0), 1.4)
    blur_b = cv2.GaussianBlur(B, (0, 0), 1.4)
    detail = orig - blur_o
    low = blur_o * (1.0 - a) + blur_b * a
    peeled = np.clip(low + detail, 0, 255)

    w = (core > 0).astype(np.float32)
    w = cv2.GaussianBlur(w, (0, 0), 0.5)
    if w.ndim == 2:
        w = w[..., None]
    out = orig * (1.0 - w) + peeled * w
    return np.clip(out, 0, 255).astype(np.uint8)


def residual_strokes(
    rgb_after: np.ndarray,
    body: np.ndarray | None = None,
    *,
    min_delta: float = 12.0,
) -> np.ndarray:
    """
    Thin leftover watermark ink after peel — for a second peel pass only.
    Uses neon/chroma residual; never returns fat blobs.
    """
    try:
        from services.mask import neon_watermark_mask

        neon = neon_watermark_mask(rgb_after)
    except Exception:  # noqa: BLE001
        neon = np.zeros(rgb_after.shape[:2], np.uint8)

    # Also catch pale white/pink text via local contrast
    gray = cv2.cvtColor(rgb_after, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local = cv2.GaussianBlur(gray, (0, 0), 3.0)
    hi = (np.abs(gray - local) > min_delta).astype(np.uint8) * 255
    hi = cv2.morphologyEx(hi, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    residual = cv2.bitwise_or(neon, hi)
    if body is not None and body.max():
        residual = cv2.bitwise_and(residual, (body > 0).astype(np.uint8) * 255)

    # Keep only thin components
    from services.body_region import thin_strokes_on_body

    body_m = body if body is not None else np.full(residual.shape, 255, np.uint8)
    return thin_strokes_on_body(residual, body_m, max_thick=8.0, max_body_cov=0.02)


def peel_image(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    m = np.asarray(mask.convert("L"))
    if m.shape[:2] != rgb.shape[:2]:
        m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(peel_overlay(rgb, m))
