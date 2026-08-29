"""
SDXL Inpainting — PRO / hard-case diffusion fill.

Opt-in: ``SDXL_ENABLE=1`` (first run downloads ~7 GB). Tuned for RTX 3060 Ti 8GB:
fp16 + model CPU offload + VAE slicing/tiling + optional xformers.

Works on a padded crop around the mask (1024²) then pastes back so untouched
pixels stay bit-exact — safer VRAM than full-frame diffusion.
"""

from __future__ import annotations

import logging
import os
import threading

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("clearmark.sdxl")

MODEL_ID = os.getenv(
    "SDXL_INPAINT_PATH",
    os.getenv("SDXL_MODEL", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"),
)
_pipe = None
_state = None  # None untried, "ok", "off"
_lock = threading.Lock()


def _enabled() -> bool:
    return os.getenv("SDXL_ENABLE", "0").strip().lower() in ("1", "true", "on", "yes")


def available() -> bool:
    if not _enabled():
        return False
    try:
        if not torch.cuda.is_available() and os.getenv("SDXL_ALLOW_CPU", "0") != "1":
            return False
    except Exception:  # noqa: BLE001
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
            from diffusers import StableDiffusionXLInpaintPipeline

            cuda = torch.cuda.is_available()
            logger.info("Loading SDXL Inpaint from %s ...", MODEL_ID)
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16 if cuda else torch.float32,
                variant="fp16" if cuda else None,
                use_safetensors=True,
            )
            pipe.set_progress_bar_config(disable=True)
            if cuda:
                pipe.enable_model_cpu_offload()
                try:
                    pipe.enable_vae_slicing()
                    pipe.enable_vae_tiling()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    logger.info("xformers attention enabled")
                except Exception:  # noqa: BLE001
                    pass
            else:
                pipe = pipe.to("cpu")
            _pipe = pipe
            _state = "ok"
            logger.info("SDXL Inpaint ready (fp16 + offload)")
            return _pipe
        except Exception as exc:  # noqa: BLE001
            logger.warning("SDXL unavailable: %s", exc)
            _state = "off"
            return None


def inpaint(
    image: Image.Image,
    mask: Image.Image,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    strength: float = 0.68,
    steps: int = 20,
    guidance_scale: float = 6.5,
) -> Image.Image:
    """Diffusion-inpaint the masked region; outside the mask stays original."""
    pipe = _load()
    if pipe is None:
        return image.convert("RGB")

    rgb = np.array(image.convert("RGB"))
    H, W = rgb.shape[:2]
    m = mask.convert("L")
    if m.size != (W, H):
        m = m.resize((W, H), Image.Resampling.NEAREST)
    mbin = (np.array(m) > 127).astype(np.uint8) * 255
    bbox = Image.fromarray(mbin).getbbox()
    if bbox is None:
        return image.convert("RGB")

    x0, y0, x1, y1 = bbox
    pad = int(0.25 * max(x1 - x0, y1 - y0)) + 12
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
    crop = Image.fromarray(rgb[y0:y1, x0:x1])
    # Tiny fringe only — large dilate here was eating limbs/clothes
    cmask = Image.fromarray(
        cv2.dilate(
            mbin[y0:y1, x0:x1],
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            1,
        )
    )
    cw, ch = crop.size
    work = 1024

    gen = pipe(
        prompt=prompt
        or (
            "high quality photo, keep the person clothing skin and body completely intact, "
            "only remove watermark text and logo, natural seamless texture"
        ),
        negative_prompt=negative_prompt
        or (
            "blurry, deformed body, missing clothes, transparent skin, erased limbs, "
            "watermark, text, logo, signature, artifacts"
        ),
        image=crop.resize((work, work), Image.Resampling.LANCZOS),
        mask_image=cmask.resize((work, work), Image.Resampling.NEAREST),
        num_inference_steps=steps,
        strength=strength,
        guidance_scale=guidance_scale,
    ).images[0].resize((cw, ch), Image.Resampling.LANCZOS)

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
