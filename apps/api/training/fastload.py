"""
Threaded sample prefetching for the training loops.

Both trainers synthesise their data on the fly (decode a JPEG, render text,
rotate, blend) and used ``num_workers=0``, so the GPU sat idle for the whole of
every sample. Process workers are not an option here: training runs inside the
API process and Windows spawns workers by re-importing ``__main__``, which would
boot a second uvicorn. Threads are safe and nearly as good, because the decode,
resize and blend calls all drop the GIL while they run.
"""

from __future__ import annotations

import logging
import os
import queue
import threading

logger = logging.getLogger("clearmark.train")


def worker_count(default: int = 8) -> int:
    env = os.getenv("TRAIN_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, min(default, (os.cpu_count() or 4)))


class ThreadedBatches:
    """Iterate ``(batch_x, batch_y, ...)`` tensors built by background threads.

    ``make(i)`` returns a tuple of tensors for one sample; whatever arity it has
    is preserved, so the detector (x, y) and the removal net (x, y, w) can share
    this loader.
    """

    def __init__(self, make, n: int, batch: int, *, workers: int | None = None, depth: int = 4):
        self.make, self.n, self.batch = make, n, batch
        self.workers = workers if workers is not None else worker_count()
        self.q: queue.Queue = queue.Queue(maxsize=max(2, depth) * batch)

    def __len__(self) -> int:
        return max(1, self.n // self.batch)

    def __iter__(self):
        import torch

        counter = iter(range(self.n))
        lock = threading.Lock()
        stop = threading.Event()

        def produce():
            while not stop.is_set():
                with lock:
                    i = next(counter, None)
                if i is None:
                    return
                try:
                    item = self.make(i)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("sample %d failed: %s", i, exc)
                    continue
                while not stop.is_set():
                    try:
                        self.q.put(item, timeout=0.25)
                        break
                    except queue.Full:
                        continue

        threads = [threading.Thread(target=produce, daemon=True) for _ in range(self.workers)]
        for t in threads:
            t.start()
        try:
            for _ in range(len(self)):
                items = []
                while len(items) < self.batch:
                    try:
                        items.append(self.q.get(timeout=30))
                    except queue.Empty:
                        if not any(t.is_alive() for t in threads):
                            break
                if not items:
                    break
                yield tuple(torch.stack(col) for col in zip(*items))
        finally:
            stop.set()
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            for t in threads:
                t.join(timeout=1.0)


def tune_backend() -> str:
    """Pick the device and switch on the fast kernels. Returns the device."""
    import torch

    if not torch.cuda.is_available():
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        return "cpu"
    torch.backends.cudnn.benchmark = True          # fixed input size → tuned kernels
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return "cuda"
