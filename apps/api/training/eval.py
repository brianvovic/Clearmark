"""
Proof that training worked.

`evaluate()` generates held-out watermarked images (diverse: logos, text, stickers
— the SAME variety the detector trained on), runs the REAL removal pipeline on
each, and returns:
  • a montage: each row is [watermarked | removed] so you can SEE it working;
  • metrics: how many watermarks the detector localised (IoU>0.4) and the average
    IoU, plus the average "residual" (how much watermark colour is left after
    removal — lower is better).

This answers "tạo bao nhiêu watermark và xóa được chưa?" with evidence rather than
just a loss number.
"""

from __future__ import annotations

import logging
import random

import cv2
import numpy as np
from PIL import Image

from training.pipeline import IMG_SIZE, _list_clean, load_wm_assets, synthesize

logger = logging.getLogger("clearmark.eval")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def evaluate(clean_dir: str, n: int = 6) -> tuple[np.ndarray, dict]:
    from services import engine, wm_detector

    assets = load_wm_assets()
    files = _list_clean(clean_dir)
    rng = random.Random()
    rows = []
    ious, residuals, detected = [], [], 0

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
            removed = np.array(engine.erase_auto(pil, False).convert("RGB"))
        except Exception:  # noqa: BLE001
            removed = img.copy()
        # residual = mean abs diff between removed and the (unknown) clean, over the
        # watermark area — we DO know clean here, so this is a fair measure.
        m = gt > 0
        residuals.append(float(np.abs(removed[m].astype(int) - clean[m].astype(int)).mean())
                         if m.any() else 0.0)

        row = np.concatenate([img, np.full((IMG_SIZE, 6, 3), 255, np.uint8), removed], axis=1)
        rows.append(row)

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
        "wm_assets": len(assets),
    }
    logger.info("eval %s", metrics)
    return montage, metrics
