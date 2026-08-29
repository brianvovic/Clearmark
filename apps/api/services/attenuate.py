"""
Peel translucent watermarks WITHOUT destroying skin / fabric texture.

LaMa/SDXL replace pixels → blur or invent missing limbs when the mask sits on a
person. Frequency separation keeps high-frequency detail (pores, weave) from the
original and only swaps the low-frequency tint that watermarks usually add.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def peel_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    ``rgb`` HxWx3 uint8, ``mask`` HxW {0,255}. Outside mask is bit-exact original.
    """
    m = (mask > 127).astype(np.uint8)
    if m.max() == 0:
        return rgb

    # Thin the mask — only peel the stroke core, not a fat blob of skin
    core = cv2.morphologyEx(m * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if core.max() == 0:
        core = m * 255
    # Tiny dilate so soft glyph edges are covered
    core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)

    orig = rgb.astype(np.float32)
    # Low-freq guide: Telea on a slim mask estimates clean colour under the ink
    guide = cv2.inpaint(rgb, core, inpaintRadius=2, flags=cv2.INPAINT_TELEA).astype(np.float32)

    blur_o = cv2.GaussianBlur(orig, (0, 0), 1.6)
    blur_g = cv2.GaussianBlur(guide, (0, 0), 1.6)
    detail = orig - blur_o  # pores / cloth weave stay
    peeled = np.clip(blur_g + detail, 0, 255)

    a = (core > 0).astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), 0.7)[..., None]
    a = np.clip(a, 0, 1)
    # Prefer original more when residual is small (don't over-peel clean skin)
    delta = np.abs(orig - guide).mean(axis=2, keepdims=True)
    conf = np.clip(delta / 18.0, 0.25, 1.0)  # stronger peel where colour differs
    a = a * conf

    out = orig * (1.0 - a) + peeled * a
    return np.clip(out, 0, 255).astype(np.uint8)


def peel_image(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    m = np.asarray(mask.convert("L"))
    if m.shape[:2] != rgb.shape[:2]:
        m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(peel_overlay(rgb, m))
