"""
Trained watermark detector — inference side.

Loads the U-Net trained by training/pipeline.py (weights at
apps/api/assets/wm_detector.pt) and predicts a watermark mask for a full-res
image. The engine prefers this over the colour heuristic when a model exists, so
detection generalises to positions/scales/tiling the heuristic can't handle.

Fully optional and lazy: no torch import and no model load until first use, and
if the weights aren't there `available()` is False.
"""

from __future__ import annotations

import logging
import os
import threading

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("clearmark.wm_detector")

MODEL_PATH = os.getenv(
    "WM_DETECTOR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "wm_detector.pt"),
)

_net = None
_size = 384
_dev = "cpu"
_state = None  # None untried, "ok", "off"
_lock = threading.Lock()


def available() -> bool:
    if _state is not None:
        return _state == "ok"
    if not os.path.exists(MODEL_PATH):
        return False
    return _load() is not None


def _load():
    global _net, _size, _dev, _state
    if _state == "ok":
        return _net
    with _lock:
        if _state == "ok":
            return _net
        try:
            import torch

            from training.pipeline import _build_unet

            ckpt = torch.load(MODEL_PATH, map_location="cpu")
            net = _build_unet()
            net.load_state_dict(ckpt["state"])
            _size = int(ckpt.get("img_size", 384))
            _dev = "cuda" if torch.cuda.is_available() else "cpu"
            net.to(_dev).eval()
            _net = net
            _state = "ok"
            logger.info("watermark detector loaded (%s, %dpx) on %s", MODEL_PATH, _size, _dev)
            return _net
        except Exception as exc:  # noqa: BLE001
            logger.warning("watermark detector unavailable: %s", exc)
            _state = "off"
            return None


def detect(image: Image.Image, thresh: float = 0.5) -> Image.Image:
    """Predict an L-mode watermark mask at the ORIGINAL resolution."""
    import torch

    net = _load()
    if net is None:
        return Image.new("L", image.size, 0)
    rgb = np.array(image.convert("RGB"))
    H, W = rgb.shape[:2]
    small = cv2.resize(rgb, (_size, _size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(small).permute(2, 0, 1).float().div(255).unsqueeze(0).to(_dev)
    with torch.inference_mode():
        prob = torch.sigmoid(net(x))[0, 0].cpu().numpy()
    m = (prob >= thresh).astype(np.uint8) * 255
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    # Hard binary + light clean only — dilate happens once in engine.erase
    # (prepare_removal_mask) so we don't double-grow and smear.
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return Image.fromarray(m, "L")


def reset():
    """Force a reload after (re)training."""
    global _state, _net
    with _lock:
        _state, _net = None, None
