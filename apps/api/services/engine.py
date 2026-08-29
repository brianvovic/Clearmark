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

import cv2
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


def _thin_components(mask: np.ndarray, *, max_thick: float = 14.0) -> np.ndarray:
    """Keep only thin connected components (text/strokes); drop fat body blobs."""
    m = (mask > 127).astype(np.uint8)
    if m.max() == 0:
        return m
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 12:
            continue
        comp = (labels == i).astype(np.uint8)
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        thick = float(dist.max() * 2)
        if thick <= max_thick or area < 400:
            out[comp > 0] = 255
    return out


def detect_mask(original: Image.Image, remove_text: bool, mode: str = "smart") -> Image.Image:
    """Detect watermark mask. On body: thin strokes only — never fat blobs."""
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

    from services.body_region import body_mask, thin_strokes_on_body

    size = original.size
    rgb = np.array(original.convert("RGB"))
    body = body_mask(rgb, dilate_px=16)
    person = float((body > 0).mean()) > 0.05
    parts: list[Image.Image] = []
    fallback: Image.Image | None = None

    def _accept(mask_img: Image.Image | np.ndarray, *, max_cov: float = 0.3) -> Image.Image | None:
        if isinstance(mask_img, np.ndarray):
            arr = mask_img if mask_img.ndim == 2 else mask_img[..., 0]
        else:
            arr = np.asarray(mask_img.convert("L"))
        if not _mask_ok(arr, max_cov=max_cov):
            return None
        if person:
            # Body: only thin strokes; bg parts of same mask also filtered
            thin = thin_strokes_on_body(
                arr, body,
                max_thick=12.0 if mode == "fast" else 16.0,
                max_body_cov=0.06,
            )
            if thin.max() == 0:
                return None
            return Image.fromarray(thin, mode="L")
        thin = _thin_components(arr, max_thick=18.0)
        return Image.fromarray(thin if thin.max() else arr, mode="L")

    # 1) Trained detector
    try:
        from services import wm_detector

        if mode != "fast" and wm_detector.available():
            ok = _accept(wm_detector.detect(original), max_cov=0.18 if person else 0.3)
            if ok is not None:
                parts.append(ok)
                fallback = fallback or ok
    except Exception as exc:  # noqa: BLE001
        logger.warning("trained detector failed: %s", exc)

    # 2) Neon
    try:
        from services.mask import neon_watermark_mask

        neon = neon_watermark_mask(rgb)
        ok = _accept(neon, max_cov=0.18)
        if ok is not None:
            parts.append(ok)
            fallback = fallback or ok
            try:
                from services import deblend

                completed = deblend.aligned_mask(original, neon)
                cok = _accept(completed, max_cov=0.18) if completed is not None else None
                if cok is not None:
                    parts.append(cok)
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("neon detect failed: %s", exc)

    # 3) Florence — skip on person (fat boxes)
    try:
        from services import florence

        if mode != "fast" and not person and florence.available():
            ok = _accept(florence.detect(original, remove_text), max_cov=0.25)
            if ok is not None:
                parts.append(ok)
                fallback = fallback or ok
    except Exception as exc:  # noqa: BLE001
        logger.warning("Florence detect failed: %s", exc)

    # 4) Heuristic — OCR only when user opted in (never auto on person)
    try:
        small = _maybe_downscale(original, 2048)
        mask = build_auto_mask(small, remove_text=remove_text)
        if mask.size != size:
            mask = mask.resize(size, Image.Resampling.NEAREST)
        ok = _accept(mask, max_cov=0.18 if person else 0.3)
        if ok is not None:
            parts.append(ok)
            fallback = fallback or ok
    except Exception as exc:  # noqa: BLE001
        logger.warning("heuristic detect failed: %s", exc)

    united = _union_masks(size, *parts)
    if united is not None:
        if person:
            thin = thin_strokes_on_body(
                np.asarray(united), body, max_thick=16.0, max_body_cov=0.06
            )
            if thin.max():
                return Image.fromarray(thin, mode="L")
        return united
    if fallback is not None:
        return fallback
    return Image.new("L", size, 0)


def erase(original: Image.Image, mask: Image.Image, mode: str = "smart") -> Image.Image:
    """
    A) mask_body / mask_bg split
    B) body → peel/deblend only (+ residual 2nd peel on smart/pro)
    C) bg → LaMa; smart may SDXL; pro SDXL/Flux
    Never LaMa/SDXL on body.
    """
    from services.attenuate import peel_overlay, residual_strokes
    from services.body_region import body_mask, split_watermark_mask, thin_strokes_on_body
    from services.mask_prep import prepare_removal_mask

    rgb = np.asarray(original.convert("RGB"))
    body = body_mask(rgb, dilate_px=16)

    # Prep mask: binary + tiny dilate, then re-thin on body
    mask = prepare_removal_mask(mask, size=original.size, mode=mode, rgb=original)
    m = np.asarray(mask.convert("L"))
    m = thin_strokes_on_body(m, body, max_thick=16.0, max_body_cov=0.07)
    if m.max() < 128:
        return original.convert("RGB")

    on_body, on_bg = split_watermark_mask(m, body)
    out = rgb.copy()

    # ----- BODY: peel only -----
    if on_body.max():
        out = peel_overlay(out, on_body)
        logger.info("body peel %d px", int((on_body > 0).sum()))

        # Residual check → second peel (smart/pro), still no LaMa
        if mode in ("smart", "pro"):
            resid = residual_strokes(out, body)
            resid = cv2.bitwise_and(resid, on_body)  # stay inside first mask neighbourhood
            # Also allow tiny new residual near original strokes
            near = cv2.dilate(on_body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 1)
            resid = cv2.bitwise_and(resid, near)
            resid = thin_strokes_on_body(resid, body, max_thick=8.0, max_body_cov=0.015)
            if resid.max():
                logger.info("body residual peel %d px", int((resid > 0).sum()))
                out = peel_overlay(out, resid)

    # ----- BG: LaMa / SDXL -----
    if on_bg.max():
        bg_img = Image.fromarray(out)
        bg_m = Image.fromarray(on_bg, mode="L")

        if worker_url():
            try:
                png = _post(
                    "/erase",
                    files={
                        "image": ("image.png", _png(bg_img), "image/png"),
                        "mask": ("mask.png", _png(bg_m), "image/png"),
                    },
                    data={"predict_mode": "4.0" if mode == "pro" else os.getenv("GPU_PREDICT_MODE", "3.0")},
                )
                filled = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
                out[on_bg > 0] = filled[on_bg > 0]
                return Image.fromarray(out)
            except Exception as exc:  # noqa: BLE001
                logger.warning("GPU erase failed: %s", exc)

        if mode == "pro":
            for modname in ("flux", "sdxl"):
                try:
                    mod = __import__(f"services.{modname}", fromlist=["available", "inpaint"])
                    if mod.available():
                        filled = np.asarray(mod.inpaint(bg_img, bg_m).convert("RGB"))
                        out[on_bg > 0] = filled[on_bg > 0]
                        return Image.fromarray(out)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s bg failed: %s", modname, exc)

        filled = np.asarray(inpaint_fullres(bg_img, bg_m).convert("RGB"))
        out[on_bg > 0] = filled[on_bg > 0]

        if mode == "smart" and _should_escalate_sdxl(bg_img, Image.fromarray(out), bg_m):
            try:
                from services import sdxl

                if sdxl.available():
                    filled = np.asarray(sdxl.inpaint(bg_img, bg_m).convert("RGB"))
                    out[on_bg > 0] = filled[on_bg > 0]
            except Exception as exc:  # noqa: BLE001
                logger.warning("SDXL bg escalate skipped: %s", exc)

    return Image.fromarray(out)


def _mask_skin_overlap(mask: Image.Image, original: Image.Image) -> float:
    """Fraction of masked pixels that fall on face/skin protect zones."""
    from services.mask import protect_mask

    m = np.asarray(mask.convert("L")) > 127
    if not m.any():
        return 0.0
    keep = protect_mask(np.asarray(original.convert("RGB")))
    return float((keep[m] > 0).mean())


def _should_escalate_sdxl(
    original: Image.Image, filled: Image.Image, mask: Image.Image
) -> bool:
    """Escalate only if LaMa failed to change the hole AND hole is not on a person."""
    skin_ov = _mask_skin_overlap(mask, original)
    if skin_ov > 0.22:
        # Watermark sits on body — SDXL tends to invent/erase clothing; stay LaMa
        return False
    m = np.asarray(mask.convert("L")) > 127
    if not m.any():
        return False
    o = np.asarray(original.convert("RGB"), dtype=np.float32)
    f = np.asarray(filled.convert("RGB"), dtype=np.float32)
    residual = float(np.abs(f[m] - o[m]).mean())
    # LaMa barely changed pixels → still looks watermarked
    return residual < 10.0


def _maybe_sharpen(img: Image.Image, mask: Image.Image) -> Image.Image:
    """Optional Real-ESRGAN / unsharp only inside the mask (texture polish)."""
    try:
        from services import sharpen

        if sharpen.available():
            return sharpen.polish_mask(img, mask)
    except Exception:  # noqa: BLE001
        pass
    # Lightweight unsharp fallback — only in mask, never global
    import cv2

    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    m = (np.asarray(mask.convert("L")) > 127).astype(np.float32)[..., None]
    blur = cv2.GaussianBlur(rgb, (0, 0), 1.2)
    sharp = np.clip(rgb + 0.35 * (rgb - blur), 0, 255)
    out = rgb * (1 - m) + sharp * m
    return Image.fromarray(out.astype(np.uint8))



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

    # Fewer passes — extra rounds compound bad masks into body smear
    default_passes = {"smart": 2, "pro": 2}.get(mode, 2)
    max_passes = max(1, int(os.getenv("ERASE_PASSES", str(default_passes))))
    for p in range(1, max_passes):
        leftover = detect_mask(out, remove_text, mode=mode)
        arr = np.asarray(leftover.convert("L"))
        if arr.max() < 128:
            break
        cov = float((arr > 127).mean())
        if cov < 1e-4 or cov > _MAX_MASK_COVERAGE:
            break
        # Subject-guard leftover; skip pass if guard clears it or it's mostly skin noise
        from services.mask_prep import prepare_removal_mask

        leftover = prepare_removal_mask(leftover, size=out.size, mode=mode, rgb=out)
        larr = np.asarray(leftover.convert("L"))
        if larr.max() < 128:
            break
        if _mask_skin_overlap(leftover, out) > 0.35 and cov < 0.02:
            # Tiny leftover on skin — don't risk another body-smear pass
            break
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
