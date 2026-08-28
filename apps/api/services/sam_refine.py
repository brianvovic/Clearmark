"""
Optional mask refinement with SAM 2 / SAM.

The detector gives a good-but-rough mask; SAM ("Segment Anything") snaps it tight
to the watermark's real edges (thin glyphs, ornate logos, stickers) using each
mask component's box as a prompt. Better edges → cleaner inpaint, less halo.

Opt-in and defensive (Gemini suggested SAM 2; on Windows SAM 2 is hard to build,
so this also accepts SAM v1 `segment_anything`, which is pip-installable):

    pip install "git+https://github.com/facebookresearch/sam2.git"     # SAM 2, or
    pip install segment-anything                                        # SAM v1
    # + a checkpoint; point SAM_CHECKPOINT / SAM2_MODEL at it
    SAM_ENABLE=1

If nothing is available it is a transparent no-op (returns the mask unchanged), so
the pipeline never breaks. Wired into engine.erase_auto for smart/pro modes.
"""

from __future__ import annotations

import logging
import os
import threading

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("clearmark.sam")

_predictor = None
_kind = None       # "sam2" | "sam1"
_state = None      # None untried, "ok", "off"
_lock = threading.Lock()


def _enabled() -> bool:
    return os.getenv("SAM_ENABLE", "0").strip().lower() in ("1", "true", "on", "yes")


def available() -> bool:
    if not _enabled():
        return False
    if _state is not None:
        return _state == "ok"
    return _load() is not None


def _load():
    global _predictor, _kind, _state
    if _state == "ok":
        return _predictor
    with _lock:
        if _state == "ok":
            return _predictor
        # Try SAM 2 first, then SAM v1.
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            _predictor = SAM2ImagePredictor.from_pretrained(
                os.getenv("SAM2_MODEL", "facebook/sam2-hiera-large"))
            _kind, _state = "sam2", "ok"
            logger.info("SAM 2 ready")
            return _predictor
        except Exception as exc:  # noqa: BLE001
            logger.info("SAM 2 not available (%s); trying SAM v1", str(exc)[:80])
        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry

            ckpt = os.getenv("SAM_CHECKPOINT")
            if not ckpt or not os.path.exists(ckpt):
                raise RuntimeError("set SAM_CHECKPOINT to a downloaded SAM .pth")
            model_type = os.getenv("SAM_MODEL_TYPE", "vit_b")
            sam = sam_model_registry[model_type](checkpoint=ckpt)
            sam.to("cuda" if torch.cuda.is_available() else "cpu")
            _predictor = SamPredictor(sam)
            _kind, _state = "sam1", "ok"
            logger.info("SAM v1 ready (%s)", model_type)
            return _predictor
        except Exception as exc:  # noqa: BLE001
            logger.info("SAM unavailable, mask refinement off: %s", str(exc)[:100])
            _state = "off"
            return None


def refine(image: Image.Image, mask: Image.Image, max_boxes: int = 12) -> Image.Image:
    """Tighten ``mask`` to real object edges with SAM. No-op if SAM isn't loaded."""
    pred = _load()
    if pred is None:
        return mask
    try:
        rgb = np.array(image.convert("RGB"))
        H, W = rgb.shape[:2]
        m = np.array(mask.convert("L"))
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        mbin = (m > 100).astype(np.uint8)
        num, _, stats, _ = cv2.connectedComponentsWithStats(mbin, connectivity=8)
        if num <= 1:
            return mask
        order = sorted(range(1, num), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)[:max_boxes]

        pred.set_image(rgb)
        out = np.zeros((H, W), np.uint8)
        for i in order:
            x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
            box = np.array([x, y, x + w, y + h])
            if _kind == "sam2":
                masks, scores, _ = pred.predict(box=box, multimask_output=False)
                sm = masks[0]
            else:
                masks, scores, _ = pred.predict(box=box[None, :], multimask_output=False)
                sm = masks[0]
            out[sm.astype(bool)] = 255
        # Keep only where SAM AND the original roughly agree (avoid over-growth).
        grown = cv2.dilate(mbin * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), 1)
        refined = cv2.bitwise_and(out, grown)
        refined = cv2.bitwise_or(refined, mbin * 255)  # never lose original coverage
        return Image.fromarray(refined, "L")
    except Exception as exc:  # noqa: BLE001
        logger.warning("SAM refine failed, using original mask: %s", exc)
        return mask


def reset():
    global _predictor, _state, _kind
    with _lock:
        _predictor, _state, _kind = None, None, None
