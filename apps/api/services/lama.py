"""
Removal: thin-stroke Telea for text, LaMa for small opaque logos.
Outside the mask is always bit-exact original (no global fade).
"""

from __future__ import annotations

import logging
import os
import threading

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("clearmark.lama")

_lock = threading.Lock()
_lama = None
_backend: str | None = None

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
)


class _LamaRunner:
    def __init__(self, device: str = "cpu"):
        import torch
        from simple_lama_inpainting.utils import download_model, prepare_img_and_mask

        model_path = os.environ.get("LAMA_MODEL") or download_model(LAMA_MODEL_URL)
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)
        self._prepare = prepare_img_and_mask

    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        import torch

        image_t, mask_t = self._prepare(image, mask, self.device)
        with torch.inference_mode():
            inpainted = self.model(image_t, mask_t)
            cur = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            return Image.fromarray(np.clip(cur * 255, 0, 255).astype(np.uint8))


def _device() -> str:
    # Explicit override wins; otherwise auto-use CUDA when torch sees a GPU.
    forced = os.getenv("LAMA_DEVICE", "auto").lower()
    if forced == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_lama():
    global _lama, _backend
    if _backend is not None:
        return _lama
    with _lock:
        if _backend is not None:
            return _lama
        try:
            _lama = _LamaRunner(_device())
            _backend = "lama"
            logger.info("LaMa ready")
            return _lama
        except Exception as exc:  # noqa: BLE001
            logger.warning("LaMa unavailable: %s", exc)
            _lama = None
            _backend = "opencv"
            return None


def backend_name() -> str:
    get_lama()
    return _backend or "opencv"


def _bin_mask(mask: Image.Image, size: tuple[int, int]) -> np.ndarray:
    m = mask.convert("L")
    if m.size != size:
        m = m.resize(size, Image.Resampling.NEAREST)
    _, b = cv2.threshold(np.array(m), 100, 255, cv2.THRESH_BINARY)
    return b


def _feather(src: np.ndarray, dst: np.ndarray, mask_bin: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    a = (mask_bin > 0).astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), sigmaX=sigma)
    a = np.clip(a, 0, 1)[..., None]
    out = dst.astype(np.float32) * a + src.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def _telea_thin(rgb: np.ndarray, mask_bin: np.ndarray, radius: int = 3) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # 1px grow so glyph edges are covered — no fat dilate
    mask_use = cv2.dilate(mask_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
    filled = cv2.inpaint(bgr, mask_use, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    out_bgr = bgr.copy()
    out_bgr[mask_use > 0] = filled[mask_use > 0]
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def _lama_small(rgb: np.ndarray, mask_bin: np.ndarray) -> np.ndarray:
    lama = get_lama()
    if lama is None:
        return _telea_thin(rgb, mask_bin, radius=3)
    mask_use = cv2.dilate(mask_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
    result = lama(Image.fromarray(rgb), Image.fromarray(mask_use, mode="L"))
    if not isinstance(result, Image.Image):
        result = Image.fromarray(result)
    result = result.convert("RGB")
    if result.size != (rgb.shape[1], rgb.shape[0]):
        result = result.resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.LANCZOS)
    return _feather(rgb, np.array(result), mask_use, sigma=1.2)


def compose_fullres(
    original: Image.Image, small_result: Image.Image, small_mask: Image.Image
) -> Image.Image:
    if small_result.size == original.size:
        # Still protect outside mask
        src = np.array(original.convert("RGB"))
        dst = np.array(small_result.convert("RGB"))
        mask_bin = _bin_mask(small_mask, original.size)
        return Image.fromarray(_feather(src, dst, mask_bin, sigma=0.8))
    up_result = small_result.resize(original.size, Image.Resampling.LANCZOS)
    up_mask = small_mask.convert("L").resize(original.size, Image.Resampling.BILINEAR)
    mask_bin = _bin_mask(up_mask, original.size)
    src = np.array(original.convert("RGB"))
    dst = np.array(up_result.convert("RGB"))
    return Image.fromarray(_feather(src, dst, mask_bin, sigma=1.0))


def fill_region(rgb: np.ndarray, mask_bin: np.ndarray) -> np.ndarray:
    """
    Core removal on one already-cropped RGB window at its native resolution.

    Splits the mask by saturation (opaque coloured logos vs. translucent
    text/overlay) and fills each with the best available backend, feathering
    the result back inside the mask only. Returns an RGB array the same size
    as ``rgb``; pixels outside ``mask_bin`` are the untouched original.

    This is the tiling-friendly primitive — no coverage cap, no resize. The
    caller (tiler) is responsible for windowing and stitching.
    """
    if mask_bin.max() == 0:
        return rgb

    # ONE inpaint pass over the whole mask. (An earlier version split the mask by
    # saturation into two sequential passes; that left semi-transparent / neon
    # watermarks behind — a single LaMa pass on the union removes them cleanly and
    # is ~2x faster.) LaMa when available, Telea as the OpenCV-only fallback.
    if backend_name() == "lama":
        filled = _lama_small(rgb, mask_bin)
    else:
        filled = _telea_thin(rgb, mask_bin, radius=3)

    sel = mask_bin > 0
    final = rgb.copy()
    final[sel] = filled[sel]
    return final


def inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    mask_bin = _bin_mask(mask, image.size)
    coverage = float((mask_bin > 0).mean())
    logger.info("inpaint coverage=%.4f%%", 100.0 * coverage)
    if coverage < 1e-6:
        raise ValueError("Mask trống — không có vùng để xóa")

    # Safety: refuse huge masks (source of smear/fade)
    if coverage > 0.10:
        raise ValueError(
            "Vùng xóa quá lớn (mask %.1f%%). Hãy dùng Thủ công và tô mỏng đúng chữ/logo."
            % (100.0 * coverage)
        )

    return Image.fromarray(fill_region(rgb, mask_bin))
