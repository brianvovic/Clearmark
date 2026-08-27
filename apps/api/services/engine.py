"""
Removal engine — one seam between the local CPU pipeline and a GPU worker.

Every route calls ``engine.detect_mask`` / ``engine.erase`` / ``engine.erase_auto``.
Where the work runs is decided here, by environment, not by the routes:

    GPU_WORKER_URL   = https://<you>--clearmark-worker.modal.run   (unset → local)
    GPU_WORKER_TOKEN = shared secret sent as Bearer token
    GPU_PREDICT_MODE = "3.0" (default, fast/cheap) | "4.0" (SDXL, hard cases)

When ``GPU_WORKER_URL`` is unset (self-host CPU, the default), everything runs
through the local LaMa full-res tiler — identical behaviour to v1.1. When it is
set, detection uses Florence-2 + SAM2 and removal is routed between LaMa and
SDXL-inpaint on the worker, with a Real-ESRGAN sharpening pass. This mirrors
dewatermark.ai's predict_mode routing (cheap default, expensive on demand).

The remote contract (see ``worker/modal_app.py``):

    POST {GPU_WORKER_URL}/detect
        multipart: image, remove_text=0/1  ->  image/png  (L mask, white=remove)
    POST {GPU_WORKER_URL}/erase
        multipart: image, mask, predict_mode  ->  image/png  (RGB, full-res)
"""

from __future__ import annotations

import io
import logging
import os

import numpy as np
from PIL import Image

from services.mask import build_auto_mask
from services.tiler import inpaint_fullres

logger = logging.getLogger("clearmark.engine")


def worker_url() -> str | None:
    url = os.getenv("GPU_WORKER_URL", "").strip()
    return url or None


def _headers() -> dict[str, str]:
    token = os.getenv("GPU_WORKER_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _timeout() -> float:
    return float(os.getenv("GPU_WORKER_TIMEOUT", "180"))


def _post(path: str, files: dict, data: dict) -> bytes:
    """POST to the GPU worker and return raw PNG bytes. Raises on failure."""
    import httpx  # local import: only needed when a worker is configured

    url = worker_url()
    assert url, "worker_url() is None"
    with httpx.Client(timeout=_timeout()) as client:
        resp = client.post(
            url.rstrip("/") + path, files=files, data=data, headers=_headers()
        )
        resp.raise_for_status()
        return resp.content


def backend_label() -> str:
    if worker_url():
        return "gpu-worker"
    try:
        import torch

        from services import florence

        if torch.cuda.is_available():
            det = "florence" if florence.available() else "heuristic"
            return f"local-gpu ({det} detect + lama)"
    except Exception:  # noqa: BLE001
        pass
    return "local-cpu"


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def detect_mask(original: Image.Image, remove_text: bool) -> Image.Image:
    """Return an L-mode mask (white = remove) at the ORIGINAL resolution."""
    if worker_url():
        try:
            png = _post(
                "/detect",
                files={"image": ("image.png", _png(original), "image/png")},
                data={"remove_text": "1" if remove_text else "0"},
            )
            m = Image.open(io.BytesIO(png)).convert("L")
            if m.size != original.size:
                m = m.resize(original.size, Image.Resampling.NEAREST)
            return m
        except Exception as exc:  # noqa: BLE001
            logger.warning("GPU detect failed, falling back to local: %s", exc)

    # Best when trained: the learned watermark detector generalises to any
    # position / scale / tiling. Used as the primary detector once a model exists.
    try:
        from services import wm_detector

        if wm_detector.available():
            mask = wm_detector.detect(original)
            arr = np.array(mask)
            if arr.max() >= 128 and float((arr > 0).mean()) <= 0.3:
                return mask
    except Exception as exc:  # noqa: BLE001
        logger.warning("trained detector failed: %s", exc)

    # Fast heuristic pass: neon / saturated-colour watermark strokes (magenta/cyan
    # semi-transparent logos & text over skin). Near-instant and very precise for
    # this common case — no GPU needed, and it does NOT touch the skin underneath.
    try:
        from services.mask import neon_watermark_mask

        neon = neon_watermark_mask(np.array(original.convert("RGB")))
        if neon.max() >= 128 and float((neon > 0).mean()) <= 0.25:
            # If the known hoalau.xyz logo can be aligned, use it to complete the
            # mask (fills the white neon cores the colour rule misses).
            try:
                from services import deblend

                completed = deblend.aligned_mask(original, neon)
                if completed is not None:
                    return completed
            except Exception as exc:  # noqa: BLE001
                logger.warning("aligned_mask failed: %s", exc)
            return Image.fromarray(neon, mode="L")
    except Exception as exc:  # noqa: BLE001
        logger.warning("neon detect failed: %s", exc)

    # Local GPU: Florence-2 open-vocabulary watermark detector (accurate; skips
    # faces). Falls through to the legacy heuristic if the model isn't present.
    try:
        from services import florence

        if florence.available():
            mask = florence.detect(original, remove_text)
            if np.array(mask).max() >= 128:
                return mask
            # Florence found nothing → try the heuristic as a second opinion.
    except Exception as exc:  # noqa: BLE001
        logger.warning("Florence detect failed, using heuristic: %s", exc)

    # Legacy heuristic: detect on a fast downscaled copy, upscale to native size.
    small = _maybe_downscale(original, 2048)
    mask = build_auto_mask(small, remove_text=remove_text)
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.Resampling.NEAREST)
    return mask


# --------------------------------------------------------------------------- #
# Removal
# --------------------------------------------------------------------------- #
def erase(original: Image.Image, mask: Image.Image) -> Image.Image:
    """Remove everything under ``mask`` at full resolution."""
    if worker_url():
        try:
            png = _post(
                "/erase",
                files={
                    "image": ("image.png", _png(original), "image/png"),
                    "mask": ("mask.png", _png(mask.convert("L")), "image/png"),
                },
                data={"predict_mode": os.getenv("GPU_PREDICT_MODE", "3.0")},
            )
            return Image.open(io.BytesIO(png)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logger.warning("GPU erase failed, falling back to local: %s", exc)

    # Trained end-to-end removal model (learns to rebuild the background). Applied
    # only inside the mask, so untouched areas stay pixel-faithful.
    try:
        from services import wm_remover

        if wm_remover.available():
            return wm_remover.remove(original, mask)
    except Exception as exc:  # noqa: BLE001
        logger.warning("removal model failed, using LaMa: %s", exc)

    return inpaint_fullres(original, mask)


def erase_auto(original: Image.Image, remove_text: bool) -> Image.Image:
    mask = detect_mask(original, remove_text)
    if np.array(mask).max() < 128:
        hint = "chữ/logo mờ" if remove_text else "logo/watermark màu"
        raise ValueError(
            f"Không phát hiện {hint} rõ. Chuyển tab Thủ công, tô đúng vùng cần xóa rồi bấm Xử lý."
        )
    return erase(original, mask)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _maybe_downscale(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
