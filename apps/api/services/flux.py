"""
Flux.1 Inpainting — PRO tier (opt-in).

Heavy. For RTX 3060 Ti 8GB set ``FLUX_ENABLE=1`` and keep ``FLUX_MAX_SIDE≤1024``.
Uses aggressive CPU offload; falls back gracefully if the pipeline isn't installed.
"""

from __future__ import annotations

import logging
import os
import threading

import torch
from PIL import Image

logger = logging.getLogger("clearmark.flux")

_pipe = None
_state = None
_lock = threading.Lock()

MODEL_ID = os.getenv("FLUX_MODEL", "black-forest-labs/FLUX.1-Fill-dev")


def _enabled() -> bool:
    return os.getenv("FLUX_ENABLE", "0").strip().lower() in ("1", "true", "on", "yes")


def available() -> bool:
    if not _enabled():
        return False
    try:
        if not torch.cuda.is_available():
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
            # Prefer dedicated fill/inpaint pipeline when present in diffusers
            try:
                from diffusers import FluxFillPipeline as _Pipe
            except Exception:  # noqa: BLE001
                from diffusers import FluxInpaintPipeline as _Pipe  # type: ignore

            logger.info("Loading Flux from %s …", MODEL_ID)
            pipe = _Pipe.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
            pipe.enable_model_cpu_offload()
            try:
                pipe.enable_sequential_cpu_offload()
            except Exception:  # noqa: BLE001
                pass
            try:
                pipe.vae.enable_tiling()
                pipe.vae.enable_slicing()
            except Exception:  # noqa: BLE001
                pass
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:  # noqa: BLE001
                pass
            pipe.set_progress_bar_config(disable=True)
            _pipe = pipe
            _state = "ok"
            logger.info("Flux ready (offloaded for 8GB)")
            return _pipe
        except Exception as exc:  # noqa: BLE001
            logger.warning("Flux unavailable: %s", exc)
            _state = "off"
            return None


def inpaint(
    original: Image.Image,
    mask: Image.Image,
    prompt: str = (
        "remove the watermark completely, clean natural background, "
        "high detail, photorealistic, sharp"
    ),
    strength: float = 0.80,
    steps: int = 18,
) -> Image.Image:
    pipe = _load()
    if pipe is None:
        return original.convert("RGB")

    original = original.convert("RGB")
    mask = mask.convert("L")
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.Resampling.NEAREST)

    max_side = int(os.getenv("FLUX_MAX_SIDE", "1024"))
    w, h = original.size
    ow, oh = w, h
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        w, h = int(w * scale) // 16 * 16, int(h * scale) // 16 * 16
        original = original.resize((w, h), Image.Resampling.LANCZOS)
        mask = mask.resize((w, h), Image.Resampling.NEAREST)

    kwargs = dict(
        prompt=prompt,
        image=original,
        mask_image=mask,
        height=h,
        width=w,
        num_inference_steps=steps,
        guidance_scale=3.5,
    )
    # Some Flux fill APIs use different arg names
    try:
        result = pipe(**kwargs, strength=strength).images[0]
    except TypeError:
        result = pipe(**kwargs).images[0]

    if result.size != (ow, oh):
        result = result.resize((ow, oh), Image.Resampling.LANCZOS)
    return result


def reset():
    global _pipe, _state
    with _lock:
        _pipe, _state = None, None
