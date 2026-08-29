#!/usr/bin/env python3
"""
ClearMark — High-speed synthetic dataset generator.

Speed opts:
  - Cap OpenMP/MKL/OpenBLAS threads (set before cv2 import) to avoid oversubscribe
  - Pre-load + pre-resize logo assets once, pickle into workers
  - ProcessPoolExecutor initializer + large chunksize
  - Uses training.pipeline.synthesize (blend modes, soft text, diagonal tiles)

Usage (from apps/api):
    python -m training.generate_dataset \\
        --clean-dir training/_data/clean \\
        --out-dir training/_data/synth \\
        --wm-dir assets/watermarks \\
        --num 20000 --size 384 --workers 8
"""

from __future__ import annotations

# === MUST be set before importing cv2 / numpy ===
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import glob
import logging
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("generate_dataset")

# Per-worker globals (set by initializer)
WORKER_ASSETS: list = []


def _init_worker(assets_pickled: bytes) -> None:
    global WORKER_ASSETS
    WORKER_ASSETS = pickle.loads(assets_pickled)


def _worker(payload: tuple) -> bool:
    idx, clean_path, size, out_img, out_mask, neg_prob, seed = payload
    from training.pipeline import synthesize

    rng = random.Random(seed + idx)
    try:
        clean = np.array(
            Image.open(clean_path).convert("RGB").resize((size, size), Image.BILINEAR)
        )
        if rng.random() < neg_prob:
            wm_img = clean
            mask = np.zeros((size, size), dtype=np.uint8)
        else:
            wm_img, mask = synthesize(clean, WORKER_ASSETS, rng, augment=True)

        name = f"{idx:06d}"
        cv2.imwrite(
            os.path.join(out_img, f"{name}.jpg"),
            cv2.cvtColor(wm_img, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        cv2.imwrite(os.path.join(out_mask, f"{name}.png"), mask)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("sample %d failed: %s", idx, exc)
        return False


def load_and_prepare_assets(wm_dir: str, max_side: int = 420) -> list:
    assets = []
    if not os.path.isdir(wm_dir):
        return assets
    for f in glob.glob(os.path.join(wm_dir, "*")):
        try:
            arr = np.array(Image.open(f).convert("RGBA"))
            h, w = arr.shape[:2]
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                arr = cv2.resize(
                    arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
                )
            if arr.size > 0:
                assets.append(arr)
        except Exception:  # noqa: BLE001
            pass
    return assets


def generate(
    clean_dir: str,
    out_dir: str,
    wm_dir: str,
    *,
    num: int = 20000,
    size: int = 384,
    workers: int | None = None,
    negative_prob: float = 0.12,
    seed: int = 42,
    progress_cb=None,
) -> dict:
    """Programmatic entry used by train_jobs after logo upload."""
    from training.pipeline import _list_clean

    cleans = _list_clean(clean_dir)
    if len(cleans) < 4:
        raise ValueError(f"Cần ≥4 ảnh sạch trong {clean_dir} trước khi generate.")

    assets = load_and_prepare_assets(wm_dir)
    workers = max(1, workers or ((os.cpu_count() or 4) - 1))
    out_img = Path(out_dir) / "images"
    out_mask = Path(out_dir) / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    assets_pickled = pickle.dumps(assets, protocol=pickle.HIGHEST_PROTOCOL)
    tasks = [
        (i, cleans[i % len(cleans)], size, str(out_img), str(out_mask), negative_prob, seed)
        for i in range(num)
    ]
    logger.info(
        "Generating %d @%dpx | clean=%d logos=%d workers=%d → %s",
        num, size, len(cleans), len(assets), workers, out_dir,
    )

    ok = 0
    chunk = max(8, num // (workers * 32) if workers else 8)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(assets_pickled,),
    ) as ex:
        for i, result in enumerate(ex.map(_worker, tasks, chunksize=chunk), start=1):
            if result:
                ok += 1
            if progress_cb and (i % 50 == 0 or i == num):
                progress_cb(i / num, i, num)

    logger.info("Done %d/%d → %s", ok, num, out_dir)
    return {"created": ok, "requested": num, "out_dir": out_dir, "logos": len(assets)}


def main() -> None:
    from training.pipeline import WATERMARKS_DIR

    p = argparse.ArgumentParser(description="ClearMark high-speed synth generator")
    p.add_argument("--clean-dir", required=True)
    p.add_argument("--out-dir", default="training/_data/synth")
    p.add_argument("--wm-dir", default=WATERMARKS_DIR)
    p.add_argument("--num", type=int, default=20000)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--negative-prob", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    bar = None

    def cb(frac, done, total):
        nonlocal bar
        if tqdm is None:
            return
        if bar is None:
            bar = tqdm(total=total, desc="Generating", unit="img")
        bar.n = done
        bar.refresh()

    info = generate(
        args.clean_dir, args.out_dir, args.wm_dir,
        num=args.num, size=args.size, workers=args.workers,
        negative_prob=args.negative_prob, seed=args.seed,
        progress_cb=cb,
    )
    if bar is not None:
        bar.close()
    print(info)


if __name__ == "__main__":
    main()
