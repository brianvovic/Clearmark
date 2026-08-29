"""
Proof that training worked.

`evaluate()` synthesises held-out watermarks, runs the REAL erase_auto pipeline,
reports detection IoU / residual / LPIPS, plus:
  • leftover_rate — detector still fires after erase (missed / incomplete wipe)
  • ocr_survive  — EasyOCR still reads text inside the GT mask after erase

Hard failures are saved into the hard-neg bank so the next train oversamples them.
"""

from __future__ import annotations

import logging
import random

import cv2
import numpy as np
from PIL import Image

from training import hard_neg
from training.pipeline import IMG_SIZE, _list_clean, load_wm_assets, synthesize

logger = logging.getLogger("clearmark.eval")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


_lpips_net = None
_lpips_dev = "cpu"


def _lpips_score(a_rgb: np.ndarray, b_rgb: np.ndarray) -> float | None:
    global _lpips_net, _lpips_dev
    try:
        import lpips
        import torch

        if _lpips_net is None:
            _lpips_dev = "cuda" if torch.cuda.is_available() else "cpu"
            _lpips_net = lpips.LPIPS(net="alex", verbose=False).to(_lpips_dev).eval()

        def to_t(x):
            t = torch.from_numpy(x).permute(2, 0, 1).float().div(127.5).sub(1.0)
            return t.unsqueeze(0).to(_lpips_dev)

        with torch.inference_mode():
            return float(_lpips_net(to_t(a_rgb), to_t(b_rgb)).item())
    except Exception:  # noqa: BLE001
        return None


def _ocr_still_readable(removed_rgb: np.ndarray, gt_mask: np.ndarray) -> bool | None:
    """True if EasyOCR still finds text overlapping the watermark region after erase."""
    try:
        from services.mask import _ocr_word_boxes

        boxes = _ocr_word_boxes(removed_rgb)
        if not boxes:
            return False
        gt = gt_mask > 0
        for poly, _txt in boxes:
            # poly is Nx2 points — rasterize rough bbox overlap
            xs = poly[:, 0].astype(int)
            ys = poly[:, 1].astype(int)
            x0, x1 = max(0, xs.min()), min(removed_rgb.shape[1], xs.max() + 1)
            y0, y1 = max(0, ys.min()), min(removed_rgb.shape[0], ys.max() + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            if gt[y0:y1, x0:x1].mean() > 0.15:
                return True
        return False
    except Exception:  # noqa: BLE001
        return None


def evaluate(clean_dir: str, n: int = 6) -> tuple[np.ndarray, dict]:
    from services import engine, wm_detector

    assets = load_wm_assets()
    files = _list_clean(clean_dir)
    rng = random.Random()
    rows = []
    ious, residuals, lpips_scores = [], [], []
    detected = leftover = ocr_alive = ocr_checked = hard_saved = 0

    for i in range(n):
        if files:
            clean = np.array(Image.open(files[rng.randrange(len(files))])
                             .convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
        else:
            clean = np.full((IMG_SIZE, IMG_SIZE, 3), 150, np.uint8)
        img, gt = synthesize(clean, assets, random.Random(rng.randint(0, 1 << 30)))
        pil = Image.fromarray(img)

        pred = np.array(wm_detector.detect(pil)) if wm_detector.available() else np.zeros_like(gt)
        iou = _iou(pred > 0, gt > 0)
        ious.append(iou)
        detected += int(iou > 0.4)

        try:
            removed_pil = engine.erase_auto(pil, True, mode="smart")
            removed = np.array(removed_pil.convert("RGB"))
        except Exception:  # noqa: BLE001
            removed_pil = pil
            removed = img.copy()

        m = gt > 0
        residual = float(np.abs(removed[m].astype(int) - clean[m].astype(int)).mean()) if m.any() else 0.0
        residuals.append(residual)
        lp = _lpips_score(removed, clean)
        if lp is not None:
            lpips_scores.append(lp)

        # Leftover: detector still sees watermark after erase
        try:
            after = np.array(wm_detector.detect(removed_pil)) if wm_detector.available() else np.zeros_like(gt)
            left_iou = _iou(after > 0, gt > 0)
            if left_iou > 0.25 or float((after > 0).mean()) > 0.01:
                leftover += 1
        except Exception:  # noqa: BLE001
            left_iou = 0.0

        ocr_flag = _ocr_still_readable(removed, gt)
        if ocr_flag is not None:
            ocr_checked += 1
            if ocr_flag:
                ocr_alive += 1

        # Bank hard failures for the next train
        is_hard = (
            iou < 0.4
            or residual > 25.0
            or left_iou > 0.25
            or ocr_flag is True
        )
        if is_hard:
            reason = "miss" if iou < 0.4 else ("residual" if residual > 25 else "leftover")
            hard_neg.save_case(img, gt, clean, reason=reason)
            hard_saved += 1

        row = np.concatenate([img, np.full((IMG_SIZE, 6, 3), 255, np.uint8), removed], axis=1)
        rows.append(row)

    hard_neg.prune(400)
    gap = np.full((8, rows[0].shape[1], 3), 240, np.uint8)
    montage = rows[0]
    for r in rows[1:]:
        montage = np.concatenate([montage, gap, r], axis=0)

    metrics = {
        "samples": n,
        "detected": detected,
        "detect_rate": round(detected / n, 2),
        "mean_iou": round(float(np.mean(ious)), 3),
        "mean_residual": round(float(np.mean(residuals)), 1),
        "lpips": round(float(np.mean(lpips_scores)), 3) if lpips_scores else None,
        "leftover": leftover,
        "leftover_rate": round(leftover / n, 2),
        "ocr_survive": ocr_alive if ocr_checked else None,
        "ocr_checked": ocr_checked,
        "hard_neg_saved": hard_saved,
        "hard_neg_bank": hard_neg.count(),
        "wm_assets": len(assets),
    }
    logger.info("eval %s", metrics)
    return montage, metrics
