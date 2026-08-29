"""
Optional texture polish inside the mask only (never global).

If Real-ESRGAN weights are present and ``SHARPEN_ENABLE=1``, upscale the crop
and paste back. Otherwise callers use a cheap unsharp fallback in engine.py.
"""

from __future__ import annotations

import logging
import os
import threading

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("clearmark.sharpen")

_up = None
_state = None
_lock = threading.Lock()


def _enabled() -> bool:
    return os.getenv("SHARPEN_ENABLE", "0").strip().lower() in ("1", "true", "on", "yes")


def available() -> bool:
    if not _enabled():
        return False
    if _state is not None:
        return _state == "ok"
    return _load() is not None


def _load():
    global _up, _state
    if _state == "ok":
        return _up
    with _lock:
        if _state == "ok":
            return _up
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=2)
            model_path = os.getenv(
                "REALESRGAN_MODEL",
                os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "assets", "RealESRGAN_x2plus.pth"),
            )
            if not os.path.exists(model_path):
                _state = "off"
                return None
            _up = RealESRGANer(scale=2, model_path=model_path, model=model,
                               tile=256, tile_pad=8, pre_pad=0, half=False)
            _state = "ok"
            logger.info("Real-ESRGAN polish ready (%s)", model_path)
            return _up
        except Exception as exc:  # noqa: BLE001
            logger.warning("Real-ESRGAN unavailable: %s", exc)
            _state = "off"
            return None


def polish_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Sharpen / re-texture only pixels under ``mask``."""
    up = _load()
    if up is None:
        return image
    rgb = np.array(image.convert("RGB"))
    m = np.array(mask.convert("L"))
    if m.shape[:2] != rgb.shape[:2]:
        m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(m > 127)
    if len(xs) == 0:
        return image
    pad = 8
    x0, x1 = max(0, int(xs.min()) - pad), min(rgb.shape[1], int(xs.max()) + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(rgb.shape[0], int(ys.max()) + pad)
    crop = rgb[y0:y1, x0:x1]
    try:
        out, _ = up.enhance(crop, outscale=1)  # enhance in-place quality
    except Exception as exc:  # noqa: BLE001
        logger.warning("sharpen crop failed: %s", exc)
        return image
    if out.shape[:2] != crop.shape[:2]:
        out = cv2.resize(out, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)
    sel = m[y0:y1, x0:x1] > 127
    crop2 = crop.copy()
    crop2[sel] = out[sel]
    rgb[y0:y1, x0:x1] = crop2
    return Image.fromarray(rgb)
