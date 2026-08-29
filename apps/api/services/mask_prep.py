"""
Prepare detector masks for Removal / LaMa.

Critical pipeline fix (Dewatermark-style):
  1) Hard binarize → only 0 or 255 (no soft alpha / gradient)
  2) Dilate 5–10px so the fill bites into clean background and does not
     interpolate leftover watermark fringe colours into a blurry blotch.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

# Grow mask into clean background so inpainting has uncontaminated border context.
DEFAULT_DILATE_PX = 8


def prepare_removal_mask(
    mask: Image.Image | np.ndarray,
    *,
    size: tuple[int, int] | None = None,
    dilate_px: int = DEFAULT_DILATE_PX,
) -> Image.Image:
    """Return an L-mode mask that is strictly binary {0,255} and dilated."""
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

    # Absolute binary — never feed soft/gradient masks into LaMa / Removal
    _, binary = cv2.threshold(arr.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

    if dilate_px > 0 and binary.max() > 0:
        # Ellipse radius ≈ dilate_px (odd kernel)
        k = max(3, int(dilate_px) * 2 + 1)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.dilate(binary, kernel, iterations=1)

    # Re-binarize after morphology (safety)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary, mode="L")
