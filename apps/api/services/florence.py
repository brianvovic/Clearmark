"""
Local watermark detector — Florence-2 on the machine's own GPU.

This replaces the "text/colour" heuristics (which grabbed skin and missed
translucent overlays) with an open-vocabulary object detector. Florence-2 is
prompted to localise watermark-like things; each returned box becomes part of the
removal mask. Faces/skin are then subtracted (services.mask.protect_mask), so a
detection is never allowed to eat a person's face.

Runs on CUDA when available (RTX-class GPU) — fast and accurate — and is fully
optional: if transformers/the model aren't present, ``available()`` is False and
the engine falls back to the legacy heuristic mask. Enable/disable explicitly
with LOCAL_FLORENCE=1/0; default is auto (on when CUDA + transformers exist).
"""

from __future__ import annotations

import logging
import os
import threading

import cv2
import numpy as np
from PIL import Image

from services.mask import protect_mask

logger = logging.getLogger("clearmark.florence")

_model = None
_proc = None
_lock = threading.Lock()
_state = None  # None=untried, "ok", "off"

MODEL_ID = os.getenv("FLORENCE_MODEL", "microsoft/Florence-2-large")


def _enabled() -> bool:
    v = os.getenv("LOCAL_FLORENCE", "auto").lower()
    if v in ("0", "false", "off", "no"):
        return False
    if v in ("1", "true", "on", "yes"):
        return True
    # auto: only worth it on GPU
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def available() -> bool:
    """True if Florence-2 is loaded (or can be). Cheap after first call."""
    global _state
    if _state is not None:
        return _state == "ok"
    if not _enabled():
        _state = "off"
        return False
    return _load() is not None


def _load():
    global _model, _proc, _state
    if _state == "ok":
        return _model
    with _lock:
        if _state == "ok":
            return _model
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading Florence-2 %s on %s...", MODEL_ID, dev)
            _model = (
                AutoModelForCausalLM.from_pretrained(
                    MODEL_ID, trust_remote_code=True, torch_dtype=dtype
                )
                .to(dev)
                .eval()
            )
            _proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
            _model._cm_device = dev
            _model._cm_dtype = dtype
            _state = "ok"
            logger.info("Florence-2 ready.")
            return _model
        except Exception as exc:  # noqa: BLE001
            logger.warning("Florence-2 unavailable, using heuristic mask: %s", exc)
            _state = "off"
            return None


def _run_task(img: Image.Image, task: str, text: str) -> dict:
    import torch

    model = _load()
    dev, dtype = model._cm_device, model._cm_dtype
    inputs = _proc(text=task + text, images=img, return_tensors="pt")
    inputs = {k: (v.to(dev, dtype) if v.dtype.is_floating_point else v.to(dev))
              for k, v in inputs.items()}
    with torch.inference_mode():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
    out = _proc.batch_decode(ids, skip_special_tokens=False)[0]
    return _proc.post_process_generation(out, task=task, image_size=img.size)


# Phrases we ask Florence-2 to localise. Kept generic so it catches logos,
# translucent overlays, tiled marks and corner signatures alike.
_PHRASES = ["watermark", "logo", "translucent text overlay", "stamp", "signature"]
_TEXT_PHRASES = ["text", "caption"]


def detect(image: Image.Image, remove_text: bool) -> Image.Image:
    """Return an L-mode mask (white = remove) at the original resolution."""
    img = image.convert("RGB")
    W, H = img.size
    rgb = np.array(img)
    mask = np.zeros((H, W), np.uint8)

    phrases = list(_PHRASES) + (_TEXT_PHRASES if remove_text else [])
    task = "<CAPTION_TO_PHRASE_GROUNDING>"
    boxes: list[tuple[int, int, int, int]] = []
    for phrase in phrases:
        try:
            res = _run_task(img, task, phrase)
            for b in res.get(task, {}).get("bboxes", []):
                boxes.append(tuple(int(v) for v in b))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Florence grounding '%s' failed: %s", phrase, exc)

    img_area = float(W * H)
    for (x0, y0, x1, y1) in boxes:
        bw, bh = x1 - x0, y1 - y0
        if bw <= 1 or bh <= 1:
            continue
        # Drop full-frame boxes (grounding sometimes returns the whole image).
        if bw * bh > 0.6 * img_area:
            continue
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)

    if mask.max() == 0:
        return Image.fromarray(mask, "L")

    # Tighten each box a little and NEVER cover faces/skin.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    keep = protect_mask(rgb)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(keep))
    logger.info("florence mask coverage=%.3f%% boxes=%d", 100.0 * (mask > 0).mean(), len(boxes))
    return Image.fromarray(mask, "L")
