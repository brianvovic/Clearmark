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
_MAX_MASK_COVERAGE = 0.35  # refuse absurd whole-frame unions


def _mask_ok(mask: Image.Image | np.ndarray | None, *, max_cov: float = 0.3) -> bool:
    if mask is None:
        return False
    arr = np.array(mask) if not isinstance(mask, np.ndarray) else mask
    if arr.ndim == 3:
        arr = arr[..., 0]
    return bool(arr.max() >= 128 and float((arr > 127).mean()) <= max_cov)


def _union_masks(size: tuple[int, int], *masks: Image.Image | None) -> Image.Image | None:
    """OR-combine binary masks; return None if empty or absurdly large."""
    acc = np.zeros((size[1], size[0]), dtype=np.uint8)
    any_hit = False
    for m in masks:
        if m is None:
            continue
        a = m.convert("L")
        if a.size != size:
            a = a.resize(size, Image.Resampling.NEAREST)
        arr = np.asarray(a)
        hit = arr > 127
        if hit.any():
            any_hit = True
            acc[hit] = 255
    if not any_hit:
        return None
    if float((acc > 0).mean()) > _MAX_MASK_COVERAGE:
        return None
    return Image.fromarray(acc, mode="L")


def detect_mask(original: Image.Image, remove_text: bool, mode: str = "smart") -> Image.Image:
    """Return an L-mode mask (white = remove) at the ORIGINAL resolution.

    ``mode``: "smart" (default) **unions** trained detector + Florence + neon +
    heuristic so multi-watermark images don't miss whole logos; "fast" is
    heuristic/neon only.
    """
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

    size = original.size
    parts: list[Image.Image] = []
    fallback: Image.Image | None = None

    # 1) Trained detector — general logos / tiles / opacity
    try:
        from services import wm_detector

        if mode != "fast" and wm_detector.available():
            mask = wm_detector.detect(original)
            if _mask_ok(mask):
                parts.append(mask)
                fallback = fallback or mask
    except Exception as exc:  # noqa: BLE001
        logger.warning("trained detector failed: %s", exc)

    # 2) Neon / saturated colour strokes
    try:
        from services.mask import neon_watermark_mask

        neon = neon_watermark_mask(np.array(original.convert("RGB")))
        if _mask_ok(neon, max_cov=0.25):
            try:
                from services import deblend

                completed = deblend.aligned_mask(original, neon)
                if completed is not None and _mask_ok(completed):
                    parts.append(completed)
                    fallback = fallback or completed
                else:
                    nimg = Image.fromarray(neon, mode="L")
                    parts.append(nimg)
                    fallback = fallback or nimg
            except Exception as exc:  # noqa: BLE001
                logger.warning("aligned_mask failed: %s", exc)
                nimg = Image.fromarray(neon, mode="L")
                parts.append(nimg)
                fallback = fallback or nimg
    except Exception as exc:  # noqa: BLE001
        logger.warning("neon detect failed: %s", exc)

    # 3) Florence-2 open-vocab (when installed)
    try:
        from services import florence

        if mode != "fast" and florence.available():
            mask = florence.detect(original, remove_text)
            if _mask_ok(mask):
                parts.append(mask)
                fallback = fallback or mask
    except Exception as exc:  # noqa: BLE001
        logger.warning("Florence detect failed: %s", exc)

    # 4) Legacy colour/OCR heuristic
    try:
        small = _maybe_downscale(original, 2048)
        mask = build_auto_mask(small, remove_text=remove_text)
        if mask.size != size:
            mask = mask.resize(size, Image.Resampling.NEAREST)
        if _mask_ok(mask):
            parts.append(mask)
            fallback = fallback or mask
    except Exception as exc:  # noqa: BLE001
        logger.warning("heuristic detect failed: %s", exc)

    united = _union_masks(size, *parts)
    if united is not None:
        return united
    if fallback is not None:
        return fallback
    return Image.new("L", size, 0)


# --------------------------------------------------------------------------- #
# Removal
# --------------------------------------------------------------------------- #
def erase(original: Image.Image, mask: Image.Image, mode: str = "smart") -> Image.Image:
    """Remove everything under ``mask`` at full resolution.

    Mask is always hard-binarized + dilated (5–10px) before any fill — soft /
    gradient masks cause fringe colour bleed and blotchy blur.

    ``mode``: "smart" uses LaMa inpainting (texture rebuild) and optionally the
    trained remover when ``USE_TRAINED_REMOVER=1``; "fast" = LaMa only; "pro" =
    SDXL when available.
    """
    from services.mask_prep import prepare_removal_mask

    # CRITICAL: binary 0/255 + dilate so fill bites into clean background
    mask = prepare_removal_mask(mask, size=original.size, dilate_px=8)

    if worker_url():
        try:
            png = _post(
                "/erase",
                files={
                    "image": ("image.png", _png(original), "image/png"),
                    "mask": ("mask.png", _png(mask.convert("L")), "image/png"),
                },
                data={"predict_mode": "4.0" if mode == "pro" else os.getenv("GPU_PREDICT_MODE", "3.0")},
            )
            return Image.open(io.BytesIO(png)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logger.warning("GPU erase failed, falling back to local: %s", exc)

    # PRO: local SDXL diffusion inpainting for hard cases (heavy — only if enabled).
    if mode == "pro":
        try:
            from services import sdxl

            if sdxl.available():
                return sdxl.inpaint(original, mask)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SDXL PRO failed, falling back: %s", exc)

    # Default: LaMa hole-fill (real inpainting / texture). Pure L1-trained remover
    # tends to blur — only use it when explicitly opted in after LPIPS retrain.
    use_trained = os.getenv("USE_TRAINED_REMOVER", "0").strip() in ("1", "true", "yes")
    if mode != "fast" and use_trained:
        try:
            from services import wm_remover

            if wm_remover.available():
                rem = wm_remover.remove(original, mask)
                # LaMa polish on same dilated mask — restores high-frequency texture
                # where the residual net smeared
                lama = inpaint_fullres(original, mask)
                return _blend_prefer_texture(original, rem, lama, mask)
        except Exception as exc:  # noqa: BLE001
            logger.warning("removal model failed, using LaMa: %s", exc)

    return inpaint_fullres(original, mask)


def _blend_prefer_texture(
    original: Image.Image, rem: Image.Image, lama: Image.Image, mask: Image.Image
) -> Image.Image:
    """Inside mask, keep the fill with higher local variance (sharper texture)."""
    o = np.asarray(original.convert("RGB"), dtype=np.float32)
    r = np.asarray(rem.convert("RGB"), dtype=np.float32)
    l = np.asarray(lama.convert("RGB"), dtype=np.float32)
    m = (np.asarray(mask.convert("L")) > 127).astype(np.float32)[..., None]
    # Local std as a cheap sharpness proxy (blurry patches have low variance)
    def local_std(x: np.ndarray) -> np.ndarray:
        gray = x.mean(axis=2)
        mu = cv2_blur(gray, 9)
        mu2 = cv2_blur(gray * gray, 9)
        return np.sqrt(np.maximum(mu2 - mu * mu, 0.0))

    def cv2_blur(g: np.ndarray, k: int) -> np.ndarray:
        import cv2

        return cv2.GaussianBlur(g, (k, k), 0)

    sr, sl = local_std(r), local_std(l)
    prefer_lama = (sl >= sr * 0.95).astype(np.float32)[..., None]
    filled = r * (1 - prefer_lama) + l * prefer_lama
    out = o * (1 - m) + filled * m
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def erase_auto(original: Image.Image, remove_text: bool, mode: str = "smart") -> Image.Image:
    """Detect → erase, then re-detect leftovers and erase again (multi-pass).

    Handles images with many watermarks where pass-1 only clears some of them.
    Caps passes via ``ERASE_PASSES`` (default 3). Fast mode = single pass.
    """
    mask = detect_mask(original, remove_text, mode=mode)
    if np.array(mask).max() < 128:
        hint = "chữ/logo mờ" if remove_text else "logo/watermark màu"
        raise ValueError(
            f"Không phát hiện {hint} rõ. Chuyển tab Thủ công, tô đúng vùng cần xóa rồi bấm Xử lý."
        )
    # Optional: snap the mask to real edges with SAM (opt-in, smart/pro only).
    if mode != "fast":
        try:
            from services import sam_refine

            if sam_refine.available():
                mask = sam_refine.refine(original, mask)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SAM refine skipped: %s", exc)

    out = erase(original, mask, mode=mode)
    if mode == "fast":
        return out

    max_passes = max(1, int(os.getenv("ERASE_PASSES", "3")))
    for p in range(1, max_passes):
        leftover = detect_mask(out, remove_text, mode=mode)
        arr = np.asarray(leftover.convert("L"))
        if arr.max() < 128:
            break
        cov = float((arr > 127).mean())
        # Ignore tiny noise speckles and absurd full-frame false positives
        if cov < 5e-5 or cov > _MAX_MASK_COVERAGE:
            break
        if mode != "fast":
            try:
                from services import sam_refine

                if sam_refine.available():
                    leftover = sam_refine.refine(out, leftover)
            except Exception:  # noqa: BLE001
                pass
        logger.info("erase multi-pass %d/%d — leftover coverage %.3f%%",
                    p + 1, max_passes, cov * 100)
        out = erase(out, leftover, mode=mode)
    return out


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
