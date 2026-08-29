"""
End-to-end removal model — inference.

Loads the network trained by training/removal.py (assets/wm_remover.pt) and turns a
watermarked image into a clean one. To guarantee non-watermark areas stay pixel-
faithful, the network's output is composited back ONLY inside the watermark mask
(feathered); everything else is the untouched original.

engine.erase uses this instead of LaMa when the model exists. Optional and lazy.
"""

from __future__ import annotations

import logging
import os
import threading

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("clearmark.wm_remover")

MODEL_PATH = os.getenv(
    "WM_REMOVER",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "wm_remover.pt"),
)

_net = None
_size = 384
_dev = "cpu"
_state = None
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

            from training.removal import build_removal_net

            ck = torch.load(MODEL_PATH, map_location="cpu")
            net = build_removal_net()
            net.load_state_dict(ck["state"])
            _size = int(ck.get("img_size", 384))
            _dev = "cuda" if torch.cuda.is_available() else "cpu"
            net.to(_dev).eval()
            _net = net
            _state = "ok"
            logger.info("removal model loaded (%s, %dpx) on %s", MODEL_PATH, _size, _dev)
            return _net
        except Exception as exc:  # noqa: BLE001
            logger.warning("removal model unavailable: %s", exc)
            _state = "off"
            return None


def remove(image: Image.Image, mask: Image.Image | None) -> Image.Image:
    """Clean the watermark. If ``mask`` is given, only its region is replaced."""
    import torch

    net = _load()
    if net is None:
        return image.convert("RGB")
    rgb = np.array(image.convert("RGB"))
    H, W = rgb.shape[:2]
    small = cv2.resize(rgb, (_size, _size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(small).permute(2, 0, 1).float().div(255).unsqueeze(0).to(_dev)
    with torch.inference_mode():
        pred = net(x)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    clean = cv2.resize((pred * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)

    if mask is None:
        return Image.fromarray(clean)

    # Hard binary mask only — soft alpha reintroduces watermark fringe colours
    from services.mask_prep import prepare_removal_mask

    m = prepare_removal_mask(mask, size=(W, H), dilate_px=8)
    mbin = (np.array(m) > 127).astype(np.float32)[..., None]
    # Tiny feather (1px) only for seam; core stays binary
    a = cv2.GaussianBlur(mbin[..., 0], (0, 0), 0.8)[..., None]
    a = np.clip(a, 0, 1)
    out = clean.astype(np.float32) * a + rgb.astype(np.float32) * (1 - a)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def reset():
    global _state, _net
    with _lock:
        _state, _net = None, None
