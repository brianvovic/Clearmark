"""
PRO tier — local SDXL inpainting (diffusion) for HARD cases.

LaMa/the removal net "borrow surrounding pixels"; when a big logo sits over a face
or busy texture, they blur. SDXL Inpainting instead *understands* the context and
repaints the hidden content. It is heavy, so:

  • it only runs on the masked region (crop → 1024 → paste back), not the whole image;
  • it uses CPU offload + VAE slicing so it fits an 8 GB card (RTX 3060 Ti);
  • it is OPT-IN — set SDXL_ENABLE=1 (first use downloads ~7 GB). Off by default so
    the normal fast/smart tiers stay light.

engine.erase(mode="pro") routes here; if unavailable it falls back to the trained
removal model / LaMa automatically.
"""

from __future__ import annotations

import logging
import os
import threading

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("clearmark.sdxl")

MODEL_ID = os.getenv("SDXL_MODEL", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
_pipe = None
_state = None  # None untried, "ok", "off"
_lock = threading.Lock()


def _enabled() -> bool:
    return os.getenv("SDXL_ENABLE", "0").strip().lower() in ("1", "true", "on", "yes")


def available() -> bool:
    if not _enabled():
        return False
    if _state is not None:
        return _state == "ok"
    return _load() is not None


def _load():
    global _pipe, _state
    if _state == "ok":
        return _pipe
    with _lock:
        if _state == "ok":
            return _pipe
        try:
            import torch
            from diffusers import StableDiffusionXLInpaintPipeline

            cuda = torch.cuda.is_available()
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16 if cuda else torch.float32,
                variant="fp16" if cuda else None,
                use_safetensors=True,
            )
            pipe.set_progress_bar_config(disable=True)
            if cuda:
                pipe.enable_model_cpu_offload()   # fits 8 GB VRAM
                try:
                    pipe.enable_vae_slicing()
                    pipe.enable_vae_tiling()
                except Exception:  # noqa: BLE001
                    pass
            else:
                pipe = pipe.to("cpu")
            _pipe = pipe
            _state = "ok"
            logger.info("SDXL inpaint ready (%s)", MODEL_ID)
            return _pipe
        except Exception as exc:  # noqa: BLE001
            logger.warning("SDXL unavailable: %s", exc)
            _state = "off"
            return None


def inpaint(image: Image.Image, mask: Image.Image,
            prompt: str | None = None, steps: int = 25) -> Image.Image:
    """Diffusion-inpaint the masked region; the rest of the image is untouched."""
    pipe = _load()
    if pipe is None:
        return image.convert("RGB")

    rgb = np.array(image.convert("RGB"))
    H, W = rgb.shape[:2]
    m = mask.convert("L")
    if m.size != (W, H):
        m = m.resize((W, H), Image.Resampling.NEAREST)
    mbin = (np.array(m) > 100).astype(np.uint8) * 255
    bbox = Image.fromarray(mbin).getbbox()
    if bbox is None:
        return image.convert("RGB")

    # Pad the crop for context, keep it square-ish, work at 1024.
    x0, y0, x1, y1 = bbox
    pad = int(0.35 * max(x1 - x0, y1 - y0)) + 16
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
    crop = Image.fromarray(rgb[y0:y1, x0:x1])
    cmask = Image.fromarray(cv2.dilate(mbin[y0:y1, x0:x1],
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 1))
    cw, ch = crop.size

    gen = pipe(
        prompt=prompt or "clean background, seamless, photorealistic, high detail, no text, no watermark",
        negative_prompt="text, watermark, logo, letters, signature, blurry, artifacts, distorted",
        image=crop.resize((1024, 1024)),
        mask_image=cmask.resize((1024, 1024)),
        num_inference_steps=steps,
        strength=0.99,
        guidance_scale=7.0,
    ).images[0].resize((cw, ch))

    out = rgb.copy()
    region = np.array(gen.convert("RGB"))
    sel = mbin[y0:y1, x0:x1] > 0
    sub = out[y0:y1, x0:x1]
    sub[sel] = region[sel]
    out[y0:y1, x0:x1] = sub
    return Image.fromarray(out)


def reset():
    global _pipe, _state
    with _lock:
        _pipe, _state = None, None
