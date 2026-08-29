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
    """Union of every detector — for previews and for the session mask."""
    trusted, guess = detect_split(original, remove_text, mode=mode)
    united = _union_masks(original.size, trusted, guess)
    return united if united is not None else Image.new("L", original.size, 0)


def detect_split(
    original: Image.Image, remove_text: bool, mode: str = "smart"
) -> tuple[Image.Image, Image.Image]:
    """
    Detect watermarks, keeping stroke-accurate hits apart from blobby guesses.

    Merging them was why nothing got removed: the colour detector's thin "gaigu"
    strokes were absorbed into a Florence box covering a fifth of the frame, and
    the whole component was then rejected as too coarse to touch.
    """
    empty = Image.new("L", original.size, 0)
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
            return empty, m
        except Exception as exc:  # noqa: BLE001
            logger.warning("GPU detect failed, falling back to local: %s", exc)

    size = original.size
    rgb = np.array(original.convert("RGB"))
    exact: list[Image.Image] = []
    rough: list[Image.Image] = []

    def _accept(mask_img: Image.Image | np.ndarray, *, max_cov: float = 0.35) -> Image.Image | None:
        if isinstance(mask_img, np.ndarray):
            arr = mask_img if mask_img.ndim == 2 else mask_img[..., 0]
        else:
            arr = np.asarray(mask_img.convert("L"))
        if not _mask_ok(arr, max_cov=max_cov):
            return None
        return Image.fromarray(arr, mode="L")

    def _file(mask: Image.Image, trust_cov: float) -> None:
        """Trust a stroke-accurate source only while it stays plausible: a colour
        rule that lights up a fifth of the frame is describing the photo."""
        cov = float((np.asarray(mask) > 127).mean())
        if cov <= trust_cov:
            exact.append(mask)
        else:
            logger.info("source demoted to guess: coverage %.1f%%", cov * 100)
            rough.append(mask)

    # --- stroke-accurate sources ---
    try:
        from services.mask import neon_watermark_mask

        ok = _accept(neon_watermark_mask(rgb), max_cov=0.3)
        if ok is not None:
            _file(ok, 0.06)
    except Exception as exc:  # noqa: BLE001
        logger.warning("neon detect failed: %s", exc)

    try:
        small = _maybe_downscale(original, 2048)
        mask = build_auto_mask(small, remove_text=remove_text)
        if mask.size != size:
            mask = mask.resize(size, Image.Resampling.NEAREST)
        ok = _accept(mask, max_cov=0.35)
        if ok is not None:
            _file(ok, 0.08)
    except Exception as exc:  # noqa: BLE001
        logger.warning("heuristic detect failed: %s", exc)

    # --- blobby sources (must prove ink before anything is filled) ---
    try:
        from services import wm_detector

        if mode != "fast" and wm_detector.available():
            ok = _accept(wm_detector.detect(original), max_cov=0.35)
            if ok is not None:
                rough.append(ok)
    except Exception as exc:  # noqa: BLE001
        logger.warning("trained detector failed: %s", exc)

    try:
        from services import florence

        if mode != "fast" and florence.available():
            ok = _accept(florence.detect(original, remove_text), max_cov=0.35)
            if ok is not None:
                rough.append(ok)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Florence detect failed: %s", exc)

    trusted = _union_masks(size, *exact) or empty
    guess = _union_masks(size, *rough) or empty
    logger.info(
        "detect: trusted=%.2f%% guess=%.2f%%",
        100.0 * float(np.asarray(trusted).mean() / 255.0),
        100.0 * float(np.asarray(guess).mean() / 255.0),
    )
    return trusted, guess


def erase(
    original: Image.Image,
    mask: Image.Image,
    mode: str = "smart",
    *,
    remove_text: bool = False,
    trusted: Image.Image | None = None,
    manual: Image.Image | None = None,
) -> Image.Image:
    """
    Removal routed by what the masked pixels actually are:

      thin ink   → alpha peel + tiny Telea   (works on skin, clothes, anything)
      small solid→ LaMa / SDXL(pro)          (opaque stamp, capped by size)
      wide area  → tint correction only      (never generative → never smears)

    A wrong mask can therefore dim a region, but can never repaint a person.
    """
    from services.attenuate import erase_ink, peel_overlay, smooth_fill
    from services.ink import (
        classify,
        correct_tint,
        ink_within,
        promote_matching_ink,
        revert_worn_garments,
        split_by_surround_texture,
    )
    from services.mask import _apply_face_protection
    from services.mask_prep import person_zone, prepare_removal_mask

    rgb = np.asarray(original.convert("RGB"))

    mask = prepare_removal_mask(mask, size=original.size, mode=mode, rgb=original)
    m = np.asarray(mask.convert("L"))
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    m = _apply_face_protection(m, rgb)
    if m.max() < 128:
        return original.convert("RGB")

    def _binarize(img: Image.Image | None) -> np.ndarray:
        if img is None:
            return np.zeros_like(m)
        a = np.asarray(img.convert("L"))
        if a.shape != m.shape:
            a = cv2.resize(a, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
        return cv2.threshold(a, 127, 255, cv2.THRESH_BINARY)[1]

    hand = cv2.bitwise_and(_binarize(manual), m)

    # Stroke-accurate masks (brush, colour detector, OCR) are removed as drawn;
    # everything else has to show ink first.
    t = cv2.bitwise_and(_apply_face_protection(_binarize(trusted), rgb), m) \
        if trusted is not None else np.zeros_like(m)

    guess = cv2.bitwise_and(m, cv2.bitwise_not(t))
    if t.max() and guess.max():
        # Let a confirmed mark claim the halo a coarse box adds around it, but
        # no further: a box drawn around the watermark often also covers a bird
        # or a face, and those must not inherit the confirmation.
        halo = cv2.dilate(t, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), 1)
        t = cv2.bitwise_or(t, cv2.bitwise_and(guess, halo))
        guess = cv2.bitwise_and(m, cv2.bitwise_not(t))
        # Same ink elsewhere in the box = same watermark
        t = cv2.bitwise_or(t, promote_matching_ink(rgb, t, guess))
        guess = cv2.bitwise_and(m, cv2.bitwise_not(t))

    thin, solid, wide = classify(rgb, guess)
    if t.max():
        t_thin, t_solid, t_wide = classify(rgb, t, trusted=True)
        thin = cv2.bitwise_or(thin, t_thin)
        solid = cv2.bitwise_or(solid, t_solid)
        wide = cv2.bitwise_or(wide, t_wide)

    # Never hole-fill a person. Manual brush is the only exception.
    person = person_zone(rgb, dilate_px=8)
    if hand.max():
        person = cv2.bitwise_and(person, cv2.bitwise_not(hand))
    on_body = cv2.bitwise_or(solid, wide)
    on_body[person == 0] = 0
    if on_body.max():
        body_ink = ink_within(rgb, on_body, min_delta=8.0)
        thin = cv2.bitwise_or(thin, body_ink)
        solid[person > 0] = 0
        wide[person > 0] = 0
        logger.info("body diverted from LaMa: %d px → peel/ink", int((on_body > 0).sum()))

    out = rgb.copy()

    # 1) Translucent panels / bands — flatten the tint, keep every detail
    if wide.max():
        out = correct_tint(out, wide)

    # 2) Body first: peel overlay (generic deblend, not hoalau-only)
    body_thin = np.zeros_like(thin)
    if thin.max():
        body_thin = thin.copy()
        body_thin[person == 0] = 0
        if body_thin.max():
            strength = {"fast": 1.05, "smart": 1.2, "pro": 1.3}.get(mode, 1.15)
            if remove_text:
                strength = min(1.4, strength + 0.1)
            out = peel_overlay(out, body_thin, strength=strength)
        radius = 2 if mode == "fast" else 3
        bg_thin = thin.copy()
        bg_thin[person > 0] = 0
        area0 = max(1, int((thin > 0).sum()))
        if bg_thin.max():
            out = erase_ink(out, bg_thin, radius=radius)
        resid_body = ink_within(out, body_thin if body_thin.max() else thin, min_delta=9.0)
        resid_body[person == 0] = 0
        if resid_body.max():
            out = erase_ink(out, resid_body, radius=2)

        rounds = 1 if mode == "fast" else (2 if mode == "smart" else 3)
        seed = bg_thin if bg_thin.max() else thin
        for _ in range(rounds):
            resid = ink_within(out, seed, min_delta=9.0)
            left = int((resid > 0).sum())
            if left == 0 or left < 0.02 * area0:
                break
            out = erase_ink(out, resid, radius=radius)
            seed = resid
        logger.info("ink handled (%d px)", area0)

    # 3) Compact blobs — the only path allowed to invent pixels
    if solid.max():
        # On a smooth backdrop a generative fill leaves a visible patch, while
        # interpolation is invisible; keep the heavy model for real texture.
        flat, textured = split_by_surround_texture(out, solid)
        if flat.max():
            logger.info("smooth fill %d px", int((flat > 0).sum()))
            out = smooth_fill(out, flat)
        solid = textured
    if solid.max():
        src = Image.fromarray(out)
        sm = Image.fromarray(solid, mode="L")
        logger.info("solid inpaint %d px mode=%s", int((solid > 0).sum()), mode)
        filled: np.ndarray | None = None

        if worker_url():
            try:
                png = _post(
                    "/erase",
                    files={
                        "image": ("image.png", _png(src), "image/png"),
                        "mask": ("mask.png", _png(sm), "image/png"),
                    },
                    data={
                        "predict_mode": "4.0"
                        if mode == "pro"
                        else os.getenv("GPU_PREDICT_MODE", "3.0")
                    },
                )
                filled = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("GPU erase failed: %s", exc)

        if filled is None and mode == "pro":
            for modname in ("flux", "sdxl"):
                try:
                    mod = __import__(
                        f"services.{modname}", fromlist=["available", "inpaint"]
                    )
                    if mod.available():
                        filled = np.asarray(mod.inpaint(src, sm).convert("RGB"))
                        break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s inpaint failed: %s", modname, exc)

        if filled is None:
            try:
                filled = np.asarray(inpaint_fullres(src, sm).convert("RGB"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("LaMa/tiler failed, Telea fallback: %s", exc)
                filled = erase_ink(out, solid, radius=5)

        sel = solid > 0
        out[sel] = filled[sel]

    return Image.fromarray(revert_worn_garments(rgb, out, keep=hand))


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
    trusted, guess = detect_split(original, remove_text, mode=mode)
    mask = _union_masks(original.size, trusted, guess)
    if mask is None or np.array(mask).max() < 128:
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

    out = erase(original, mask, mode=mode, remove_text=remove_text, trusted=trusted)
    if mode == "fast":
        return out

    # Fewer passes — extra rounds compound bad masks into body smear
    default_passes = {"smart": 2, "pro": 2}.get(mode, 2)
    max_passes = max(1, int(os.getenv("ERASE_PASSES", str(default_passes))))
    for p in range(1, max_passes):
        left_trusted, left_guess = detect_split(out, remove_text, mode=mode)
        leftover = _union_masks(out.size, left_trusted, left_guess)
        if leftover is None:
            break
        arr = np.asarray(leftover.convert("L"))
        cov = float((arr > 127).mean())
        if arr.max() < 128 or cov < 1e-4 or cov > _MAX_MASK_COVERAGE:
            break
        logger.info(
            "erase multi-pass %d/%d — leftover coverage %.3f%%",
            p + 1,
            max_passes,
            cov * 100,
        )
        out = erase(
            out, leftover, mode=mode, remove_text=remove_text, trusted=left_trusted
        )
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
