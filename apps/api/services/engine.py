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
    """Detect watermark mask. BG components kept fully; body refined to strokes."""
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

    from services.body_region import body_mask, refine_body_mask

    size = original.size
    rgb = np.array(original.convert("RGB"))
    body = body_mask(rgb, dilate_px=10)
    person = float((body > 0).mean()) > 0.05
    parts: list[Image.Image] = []
    fallback: Image.Image | None = None

    def _accept(mask_img: Image.Image | np.ndarray, *, max_cov: float = 0.35) -> Image.Image | None:
        if isinstance(mask_img, np.ndarray):
            arr = mask_img if mask_img.ndim == 2 else mask_img[..., 0]
        else:
            arr = np.asarray(mask_img.convert("L"))
        if not _mask_ok(arr, max_cov=max_cov):
            return None
        # Always refine: keep BG intact, trim absurd body blobs only
        refined = refine_body_mask(arr, body)
        if refined.max() == 0:
            return None
        return Image.fromarray(refined, mode="L")

    try:
        from services import wm_detector

        if mode != "fast" and wm_detector.available():
            ok = _accept(wm_detector.detect(original), max_cov=0.35)
            if ok is not None:
                parts.append(ok)
                fallback = fallback or ok
    except Exception as exc:  # noqa: BLE001
        logger.warning("trained detector failed: %s", exc)

    try:
        from services.mask import neon_watermark_mask

        neon = neon_watermark_mask(rgb)
        ok = _accept(neon, max_cov=0.3)
        if ok is not None:
            parts.append(ok)
            fallback = fallback or ok
    except Exception as exc:  # noqa: BLE001
        logger.warning("neon detect failed: %s", exc)

    try:
        from services import florence

        if mode != "fast" and florence.available():
            # Florence OK on non-person; on person still useful for big logos off-body
            ok = _accept(florence.detect(original, remove_text), max_cov=0.35)
            if ok is not None:
                parts.append(ok)
                fallback = fallback or ok
    except Exception as exc:  # noqa: BLE001
        logger.warning("Florence detect failed: %s", exc)

    try:
        small = _maybe_downscale(original, 2048)
        mask = build_auto_mask(small, remove_text=remove_text)
        if mask.size != size:
            mask = mask.resize(size, Image.Resampling.NEAREST)
        ok = _accept(mask, max_cov=0.35)
        if ok is not None:
            parts.append(ok)
            fallback = fallback or ok
    except Exception as exc:  # noqa: BLE001
        logger.warning("heuristic detect failed: %s", exc)

    united = _union_masks(size, *parts)
    if united is not None:
        return Image.fromarray(refine_body_mask(np.asarray(united), body), mode="L")
    if fallback is not None:
        return fallback
    # Union too large — keep best single part rather than empty (empty = zero removal)
    if parts:
        best = max(parts, key=lambda m: int((np.asarray(m.convert("L")) > 127).sum()))
        return Image.fromarray(refine_body_mask(np.asarray(best), body), mode="L")
    return Image.new("L", size, 0)


def erase(
    original: Image.Image,
    mask: Image.Image,
    mode: str = "smart",
    *,
    remove_text: bool = False,
) -> Image.Image:
    """
    Balanced removal:
      body → strong peel (+ thin Telea if residual / text mode)
      bg   → full LaMa (SDXL/Flux on smart escalate / pro)
    """
    from services.attenuate import peel_overlay, residual_strokes, thin_fill
    from services.body_region import body_mask, refine_body_mask, split_watermark_mask
    from services.mask_prep import prepare_removal_mask

    rgb = np.asarray(original.convert("RGB"))
    body = body_mask(rgb, dilate_px=10)

    # Binary + small dilate — do NOT subject-guard away text on skin
    mask = prepare_removal_mask(mask, size=original.size, mode=mode, rgb=None)
    m = refine_body_mask(np.asarray(mask.convert("L")), body)
    if m.max() < 128:
        m = np.asarray(mask.convert("L"))
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    if m.max() < 128:
        return original.convert("RGB")

    on_body, on_bg = split_watermark_mask(m, body)
    # If body zone swallowed the whole frame (skin-tone bg), treat thick blobs as bg
    body_frac = float((body > 0).mean())
    if body_frac > 0.55 and on_bg.max() == 0 and on_body.max():
        # Reclassify fat components as background so LaMa can run
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (on_body > 0).astype(np.uint8), connectivity=8
        )
        body_area = max(1, int((body > 0).sum()))
        new_body = np.zeros_like(on_body)
        new_bg = np.zeros_like(on_body)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            comp = labels == i
            dist = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
            thick = float(dist.max() * 2) if dist.size else 0.0
            # Fat logo / stamp → bg inpaint; thin text stays body peel
            if thick > 22 or area / body_area > 0.04:
                new_bg[comp] = 255
            else:
                new_body[comp] = 255
        if new_bg.max():
            on_body, on_bg = new_body, new_bg
            logger.info(
                "body-zone reclass: peel=%d inpaint=%d (body_frac=%.2f)",
                int((on_body > 0).sum()),
                int((on_bg > 0).sum()),
                body_frac,
            )

    out = rgb.copy()
    peel_strength = {"fast": 1.05, "smart": 1.25, "pro": 1.35}.get(mode, 1.2)
    if remove_text:
        peel_strength = min(1.45, peel_strength + 0.15)

    # ----- BODY -----
    if on_body.max():
        out = peel_overlay(out, on_body, strength=peel_strength)
        rounds = 1 if mode == "fast" else (2 if mode == "smart" else 3)
        if remove_text:
            rounds += 1
        seed = on_body
        for _ in range(rounds):
            resid = residual_strokes(out, seed_mask=seed, min_delta=7.0)
            if resid.max() == 0:
                break
            out = peel_overlay(out, resid, strength=peel_strength)
            seed = resid
        # Thin Telea when residual remains (always if remove_text)
        if mode in ("smart", "pro") or remove_text:
            resid = residual_strokes(out, seed_mask=on_body, min_delta=6.0)
            if resid.max():
                logger.info("body thin_fill %d px", int((resid > 0).sum()))
                out = thin_fill(
                    out, resid, radius=2 if mode == "smart" and not remove_text else 3
                )
        logger.info("body peel done (%d px)", int((on_body > 0).sum()))

    # ----- BACKGROUND -----
    if on_bg.max():
        bg_img = Image.fromarray(out)
        bg_m = Image.fromarray(on_bg, mode="L")
        logger.info("bg inpaint %d px mode=%s", int((on_bg > 0).sum()), mode)

        def _telea_fallback(img: Image.Image, msk: Image.Image) -> np.ndarray:
            arr = np.asarray(img.convert("RGB"))
            mb = np.asarray(msk.convert("L"))
            _, mb = cv2.threshold(mb, 127, 255, cv2.THRESH_BINARY)
            return thin_fill(arr, mb, radius=5)

        filled_arr: np.ndarray | None = None

        if worker_url():
            try:
                png = _post(
                    "/erase",
                    files={
                        "image": ("image.png", _png(bg_img), "image/png"),
                        "mask": ("mask.png", _png(bg_m), "image/png"),
                    },
                    data={
                        "predict_mode": "4.0"
                        if mode == "pro"
                        else os.getenv("GPU_PREDICT_MODE", "3.0")
                    },
                )
                filled_arr = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("GPU erase failed: %s", exc)

        if filled_arr is None and mode == "pro":
            for modname in ("flux", "sdxl"):
                try:
                    mod = __import__(
                        f"services.{modname}", fromlist=["available", "inpaint"]
                    )
                    if mod.available():
                        filled_arr = np.asarray(
                            mod.inpaint(bg_img, bg_m).convert("RGB")
                        )
                        break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s bg failed: %s", modname, exc)

        if filled_arr is None:
            try:
                filled_arr = np.asarray(
                    inpaint_fullres(bg_img, bg_m).convert("RGB")
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("LaMa/tiler failed, Telea fallback: %s", exc)
                filled_arr = _telea_fallback(bg_img, bg_m)

        out[on_bg > 0] = filled_arr[on_bg > 0]

        if mode == "smart":
            try:
                from services import sdxl

                before = bg_img
                after = Image.fromarray(out)
                if sdxl.available() and _should_escalate_sdxl(before, after, bg_m):
                    filled_arr = np.asarray(sdxl.inpaint(bg_img, bg_m).convert("RGB"))
                    out[on_bg > 0] = filled_arr[on_bg > 0]
            except Exception as exc:  # noqa: BLE001
                logger.warning("SDXL bg escalate skipped: %s", exc)

    if not on_body.max() and not on_bg.max() and m.max():
        logger.warning("split empty — fallback Telea")
        out = thin_fill(out, m, radius=4)

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

    out = erase(original, mask, mode=mode, remove_text=remove_text)
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
        # Prep without subject-guard — leftover text on skin must stay
        from services.mask_prep import prepare_removal_mask

        leftover = prepare_removal_mask(leftover, size=out.size, mode=mode, rgb=None)
        larr = np.asarray(leftover.convert("L"))
        if larr.max() < 128:
            break
        # Skip only tiny non-text skin noise; keep going when remove_text
        if (
            not remove_text
            and _mask_skin_overlap(leftover, out) > 0.35
            and cov < 0.02
        ):
            break
        logger.info(
            "erase multi-pass %d/%d — leftover coverage %.3f%%",
            p + 1,
            max_passes,
            cov * 100,
        )
        out = erase(out, leftover, mode=mode, remove_text=remove_text)
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
