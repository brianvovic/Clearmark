"""Measure where detector training spends its time: data synthesis vs GPU step.

Run before/after tuning to see the real gain:
    .venv311\\Scripts\\python.exe tools_bench_train.py
"""

from __future__ import annotations

import random
import sys
import time

import numpy as np


def bench_samples(n: int = 24) -> tuple[float, float]:
    import torch

    from training.fastload import ThreadedBatches, worker_count
    from training.pipeline import IMG_SIZE, _list_clean, load_wm_assets, make_sample

    assets = load_wm_assets()
    files = _list_clean("training/_data/clean")
    if not files:
        raise SystemExit("no clean images in training/_data/clean")

    t0 = time.perf_counter()
    for i in range(n):
        make_sample(files[i % len(files)], assets, random.Random(i))
    serial = n / (time.perf_counter() - t0)

    def make(i: int):
        img, mask = make_sample(files[i % len(files)], assets, random.Random(i))
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        return x, y

    batch = 8
    dl = ThreadedBatches(make, n * 2, batch)
    t0 = time.perf_counter()
    got = 0
    for x, _y in dl:
        got += x.shape[0]
    threaded = got / (time.perf_counter() - t0)
    print(f"synthesis: serial={serial:5.1f} img/s  threaded({worker_count()})={threaded:5.1f} img/s"
          f"  ->  {threaded / max(serial, 1e-3):.1f}x   ({IMG_SIZE}px)")
    return serial, threaded


def bench_step(batch: int = 8, iters: int = 12) -> tuple[float, float]:
    import torch

    from training.fastload import tune_backend
    from training.pipeline import IMG_SIZE, _build_unet

    dev = tune_backend()
    if dev != "cuda":
        print("no CUDA — skipping step benchmark")
        return 0.0, 0.0

    net = _build_unet().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    x = torch.rand(batch, 3, IMG_SIZE, IMG_SIZE, device=dev)
    y = (torch.rand(batch, 1, IMG_SIZE, IMG_SIZE, device=dev) > 0.9).float()

    def run(amp: bool) -> float:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        for _ in range(3):  # warm up cudnn autotuning
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = bce(net(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = bce(net(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        return iters * batch / (time.perf_counter() - t0)

    fp32, fp16 = run(False), run(True)
    print(f"gpu step:  fp32={fp32:5.1f} img/s  amp={fp16:5.1f} img/s  ->  {fp16 / max(fp32, 1e-3):.1f}x")
    return fp32, fp16


def main() -> int:
    serial, threaded = bench_samples()
    fp32, fp16 = bench_step()
    if fp32 and serial:
        old = 1 / (1 / serial + 1 / fp32)      # data and compute were serialised
        new = min(threaded, fp16)              # overlapped: the slower half wins
        print(f"epoch throughput: before~{old:5.1f} img/s  after~{new:5.1f} img/s"
              f"  ->  {new / max(old, 1e-3):.1f}x faster")
    return 0


if __name__ == "__main__":
    sys.exit(main())
