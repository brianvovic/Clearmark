"""
Training-job manager for the /train page.

Holds uploaded clean images on disk and runs one training job at a time in a
background thread, exposing live progress. Kept simple and single-box (one active
job) — matches a self-host tool. On success it writes the detector weights to
assets/wm_detector.pt and hot-reloads services.wm_detector so removal immediately
uses the new model.
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger("clearmark.train_jobs")

_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "training", "_data")
CLEAN_DIR = os.path.join(_BASE, "clean")
_ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
MODEL_OUT = os.path.join(_ASSETS, "wm_detector.pt")          # detector (find mask)
MODEL_REMOVAL = os.path.join(_ASSETS, "wm_remover.pt")       # end-to-end removal
JOB_STATE = os.path.join(_ASSETS, "train_job.json")          # survives restarts


def _model_path(kind: str) -> str:
    return MODEL_REMOVAL if kind == "removal" else MODEL_OUT


def _write_job(kind: str, target: int, status: str) -> None:
    import json

    try:
        with open(JOB_STATE, "w") as f:
            json.dump({"kind": kind, "target_epochs": target, "status": status}, f)
    except OSError:
        pass


def _read_job() -> dict | None:
    import json

    try:
        with open(JOB_STATE) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None

from training.pipeline import WATERMARKS_DIR  # library of logos/stickers to learn

os.makedirs(CLEAN_DIR, exist_ok=True)
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _safe_name(filename: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in filename)[-80:] or "img.png"


def save_watermark(filename: str, data: bytes) -> None:
    with open(os.path.join(WATERMARKS_DIR, _safe_name(filename)), "wb") as f:
        f.write(data)


def count_watermarks() -> int:
    return sum(1 for f in os.listdir(WATERMARKS_DIR) if f.lower().endswith(_IMG_EXTS))


def clear_watermarks() -> None:
    for f in os.listdir(WATERMARKS_DIR):
        if f == "hoalau_logo.png":
            continue  # keep the built-in default
        try:
            os.remove(os.path.join(WATERMARKS_DIR, f))
        except OSError:
            pass

_state = {
    "status": "idle",   # idle | running | done | error
    "progress": 0.0,
    "loss": None,
    "message": "",
    "updated": 0.0,
}
_lock = threading.Lock()


def _set(**kw):
    with _lock:
        _state.update(kw)
        _state["updated"] = time.time()


def _model_epochs(kind: str = "detector") -> int:
    """Cumulative epochs the given model has been trained for (0 if none)."""
    path = _model_path(kind)
    if not os.path.exists(path):
        return 0
    try:
        import torch

        ck = torch.load(path, map_location="cpu")
        return int(ck.get("trained_epochs", 0))
    except Exception:  # noqa: BLE001
        return 0


def status() -> dict:
    with _lock:
        s = dict(_state)
    s["clean_count"] = count_clean()
    s["watermark_count"] = count_watermarks()
    s["has_model"] = os.path.exists(MODEL_OUT)
    s["model_epochs"] = _model_epochs("detector")
    s["has_removal_model"] = os.path.exists(MODEL_REMOVAL)
    s["removal_epochs"] = _model_epochs("removal")
    return s


def count_clean() -> int:
    if not os.path.isdir(CLEAN_DIR):
        return 0
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    return sum(1 for f in os.listdir(CLEAN_DIR) if f.lower().endswith(exts))


def save_clean(filename: str, data: bytes) -> None:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in filename)[-80:] or "img.png"
    with open(os.path.join(CLEAN_DIR, safe), "wb") as f:
        f.write(data)


def clear_clean() -> None:
    for f in os.listdir(CLEAN_DIR):
        try:
            os.remove(os.path.join(CLEAN_DIR, f))
        except OSError:
            pass


def start(epochs: int = 8, fresh: bool = False, kind: str = "detector") -> None:
    with _lock:
        if _state["status"] in ("running", "scraping"):
            raise RuntimeError("Đang bận, chờ xong đã.")
    if count_clean() < 4:
        raise ValueError("Cần ít nhất 4 ảnh sạch để train (nên vài chục ảnh càng tốt).")
    kind = "removal" if kind == "removal" else "detector"
    path = _model_path(kind)
    base = 0 if fresh else _model_epochs(kind)
    target = base + epochs                       # FINAL cumulative epoch count
    resume = None if fresh else (path if os.path.exists(path) else None)
    _write_job(kind, target, "running")
    _set(status="running", progress=0.0, loss=None, kind=kind,
         message=f"Chuẩn bị dữ liệu ({'Removal' if kind == 'removal' else 'Detector'})…")
    threading.Thread(target=_run, args=(epochs, resume, kind, target), daemon=True).start()


def resume_if_interrupted() -> None:
    """
    On server startup: if a training job was cut off (crash/close), continue the
    remaining epochs from the last per-epoch checkpoint instead of losing progress.
    """
    job = _read_job()
    if not job or job.get("status") != "running":
        return
    kind = job.get("kind", "detector")
    target = int(job.get("target_epochs", 0))
    base = _model_epochs(kind)
    remaining = target - base
    if remaining <= 0 or count_clean() < 4:
        _write_job(kind, target, "done")
        return
    logger.info("Auto-resuming interrupted %s training: %d/%d epochs done, %d to go",
                kind, base, target, remaining)
    try:
        start(remaining, fresh=False, kind=kind)
    except Exception as exc:  # noqa: BLE001
        logger.warning("auto-resume failed: %s", exc)


def scrape(count: int, source: str = "picsum", api_key: str | None = None,
           query: str | None = None) -> None:
    with _lock:
        if _state["status"] in ("running", "scraping"):
            raise RuntimeError("Đang bận, chờ xong đã.")
    _set(status="scraping", progress=0.0, message="Đang tải ảnh sạch từ internet…")
    threading.Thread(target=_run_scrape, args=(count, source, api_key, query), daemon=True).start()


def _run_scrape(count: int, source: str, api_key: str | None, query: str | None) -> None:
    try:
        from training.scrape_clean import download

        def cb(p: float, n: int):
            _set(progress=min(0.99, p), message=f"Đã tải {n}/{count} ảnh sạch…")

        info = download(count, CLEAN_DIR, source=source, api_key=api_key, query=query, progress_cb=cb)
        _set(status="done", progress=1.0,
             message=f"Xong! Đã tải {info['downloaded']} ảnh sạch. Giờ bấm train.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("scrape failed")
        _set(status="error", message=str(exc)[:300])


def set_model(data: bytes, kind: str = "detector") -> None:
    """Accept an uploaded .pt to continue training from (validates it loads)."""
    kind = "removal" if kind == "removal" else "detector"
    target = _model_path(kind)
    tmp = target + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    try:
        import torch

        ck = torch.load(tmp, map_location="cpu")
        if kind == "removal":
            from training.removal import build_removal_net

            build_removal_net().load_state_dict(ck["state"])
        else:
            from training.pipeline import _build_unet

            _build_unet().load_state_dict(ck["state"])
    except Exception as exc:  # noqa: BLE001
        os.remove(tmp)
        raise ValueError(f"File model không hợp lệ (sai loại?): {exc}") from exc
    os.replace(tmp, target)
    _reload(kind)


def _reload(kind: str) -> None:
    try:
        if kind == "removal":
            from services import wm_remover

            wm_remover.reset()
        else:
            from services import wm_detector

            wm_detector.reset()
    except Exception:  # noqa: BLE001
        pass


def _run(epochs: int, resume: str | None, kind: str, target: int) -> None:
    label = "Removal" if kind == "removal" else "Detector"
    try:
        if kind == "removal":
            from training.removal import train_removal as _train
        else:
            from training.pipeline import train as _train

        def cb(p: float, loss: float):
            _set(progress=min(0.99, p), loss=round(loss, 4))

        def on_epoch(done_total: int, run_target: int, loss: float):
            # Each finished epoch is already saved to disk by the trainer — reload
            # it so the live system uses it immediately (auto-integrated), and show
            # the cumulative epoch count.
            _reload(kind)
            _set(status="running", loss=loss,
                 message=f"{label}: đã xong & lưu vòng {done_total}/{target} "
                         f"(loss {loss}). An toàn nếu tắt máy.")

        info = _train(CLEAN_DIR, _model_path(kind), resume_from=resume,
                      epochs=epochs, progress_cb=cb, epoch_cb=on_epoch)
        _reload(kind)
        _write_job(kind, target, "done")
        _set(status="done", progress=1.0, loss=info.get("final_loss"),
             message=f"Xong {label}! Loss {info.get('final_loss')}, tổng "
                     f"{info.get('trained_epochs')} vòng. Model đã tự tích hợp, sẵn sàng xóa.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("training failed")
        _write_job(kind, target, "done")  # a real error — don't auto-resume-loop
        _set(status="error", message=str(exc)[:300])
