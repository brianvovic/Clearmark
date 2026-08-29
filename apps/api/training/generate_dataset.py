#!/usr/bin/env python3
"""
ClearMark — offline synthetic dataset generator for the watermark detector.

Creates tens of thousands of (image, mask) pairs using the upgraded
``training.pipeline.synthesize`` (soft text, blend modes, diagonal tiles…).

Usage (from apps/api):
    python -m training.generate_dataset \\
        --clean-dir training/_data/clean \\
        --out-dir training/_data/synth \\
        --num 20000 --size 384 --workers 6 --negative-prob 0.12
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("generate_dataset")


def _worker(payload: tuple) -> bool:
    (
        idx, clean_path, wm_dir, size, out_img, out_mask, neg_prob, seed,
    ) = payload
    # Imports inside worker — Windows spawn-safe
    from training.pipeline import load_wm_assets, synthesize

    rng = random.Random(seed + idx)
    try:
        assets = _worker.assets  # type: ignore[attr-defined]
    except AttributeError:
        # Each process loads once via initializer
        assets = load_wm_assets() if not wm_dir else _load_assets(wm_dir)

    try:
        clean = np.array(Image.open(clean_path).convert("RGB").resize((size, size), Image.BILINEAR))
        if rng.random() < neg_prob:
            wm_img, mask = clean, np.zeros((size, size), np.uint8)
        else:
            wm_img, mask = synthesize(clean, assets, rng, augment=True)
        name = f"{idx:06d}"
        cv2.imwrite(
            os.path.join(out_img, f"{name}.jpg"),
            cv2.cvtColor(wm_img, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        cv2.imwrite(os.path.join(out_mask, f"{name}.png"), mask)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("sample %d failed: %s", idx, exc)
        return False


def _load_assets(wm_dir: str) -> list:
    from training.pipeline import WATERMARKS_DIR, load_wm_assets
    import glob

    # Temporarily point at custom dir if different
    if os.path.abspath(wm_dir) == os.path.abspath(WATERMARKS_DIR):
        return load_wm_assets()
    out = []
    for f in glob.glob(os.path.join(wm_dir, "*")):
        try:
            out.append(np.array(Image.open(f).convert("RGBA")))
        except Exception:  # noqa: BLE001
            pass
    return out


_ASSETS_CACHE: list | None = None


def _init_worker(wm_dir: str) -> None:
    global _ASSETS_CACHE
    _ASSETS_CACHE = _load_assets(wm_dir)
    _worker.assets = _ASSETS_CACHE  # type: ignore[attr-defined]


def main() -> None:
    from training.pipeline import WATERMARKS_DIR, _list_clean

    p = argparse.ArgumentParser(description="ClearMark synthetic dataset generator")
    p.add_argument("--clean-dir", required=True)
    p.add_argument("--out-dir", default="training/_data/synth")
    p.add_argument("--wm-dir", default=WATERMARKS_DIR)
    p.add_argument("--num", type=int, default=20000)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--negative-prob", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cleans = _list_clean(args.clean_dir)
    if len(cleans) < 5:
        raise SystemExit(f"Need ≥5 clean images in {args.clean_dir}")

    out_img = Path(args.out_dir) / "images"
    out_mask = Path(args.out_dir) / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    assets_n = len(_load_assets(args.wm_dir))
    logger.info("clean=%d logos=%d → %d samples @%dpx workers=%d",
                len(cleans), assets_n, args.num, args.size, args.workers)

    tasks = [
        (i, cleans[i % len(cleans)], args.wm_dir, args.size,
         str(out_img), str(out_mask), args.negative_prob, args.seed)
        for i in range(args.num)
    ]

    ok = 0
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x  # noqa: E731

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.wm_dir,),
    ) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs), desc="Generating"):
            if f.result():
                ok += 1

    logger.info("Done %d/%d → %s", ok, args.num, args.out_dir)


if __name__ == "__main__":
    main()
