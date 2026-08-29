"""
Training-job manager for the /train page.

Holds uploaded clean images on disk and runs one training job at a time in a
background thread, exposing live progress. On success it writes weights to
assets/wm_*.pt + checkpoints/{kind}_epN.pt and status.json so a power cut
loses at most one epoch — "Train tiếp" resumes from the highest named ckpt.
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
    # Mirror live UI state into durable status.json (survives crash mid-epoch)
    try:
        from training.checkpoints import read_status, write_status

        st = read_status()
        for k in ("status", "progress", "loss", "message", "kind"):
            if k in kw:
                st[k] = kw[k]
        write_status(st)
    except Exception:  # noqa: BLE001
        pass


def _model_epochs(kind: str = "detector") -> int:
    """Highest cumulative epoch from named ckpt or rolling latest."""
    from training.checkpoints import latest_named, resolve_resume

    path = _model_path(kind)
    _, ep_named = latest_named(kind)
    resume, ep = resolve_resume(kind, path, fresh=False)
    return max(ep_named, ep, 0)


def status() -> dict:
    with _lock:
        s = dict(_state)
    # Prefer durable status.json progress when a job was interrupted / just resumed
    try:
        from training.checkpoints import read_status

        disk = read_status()
        if disk.get("status") == "running" and s.get("status") != "running":
            s["status"] = "running"
            s["progress"] = float(disk.get("progress") or 0)
            s["message"] = disk.get("message") or s.get("message") or ""
            s["loss"] = disk.get("loss", s.get("loss"))
            s["kind"] = disk.get("kind", s.get("kind"))
        elif s.get("status") == "running" and disk.get("progress") is not None:
            # Keep UI bar in sync with last saved epoch even if mid-batch
            s["progress"] = max(float(s.get("progress") or 0), float(disk.get("progress") or 0))
            if disk.get("message"):
                s["message"] = disk["message"]
    except Exception:  # noqa: BLE001
        pass
    s["clean_count"] = count_clean()
    s["watermark_count"] = count_watermarks()
    s["has_model"] = os.path.exists(MODEL_OUT)
    s["model_epochs"] = _model_epochs("detector")
    s["model_size_mb"] = _size_mb(MODEL_OUT)
    s["has_removal_model"] = os.path.exists(MODEL_REMOVAL)
    s["removal_epochs"] = _model_epochs("removal")
    s["removal_size_mb"] = _size_mb(MODEL_REMOVAL)
    return s


def _size_mb(path: str) -> float:
    try:
        return round(os.path.getsize(path) / 1e6, 1)
    except OSError:
        return 0.0


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

    from training.checkpoints import resolve_resume, write_status

    resume, base = (None, 0) if fresh else resolve_resume(kind, path, fresh=False)
    if fresh:
        base = 0
    else:
        base = _model_epochs(kind)
        resume, _ = resolve_resume(kind, path, fresh=False)

    target = base + epochs
    _write_job(kind, target, "running")
    write_status({
        "status": "running", "kind": kind, "epoch": base,
        "target_epochs": target, "base_epochs": base, "progress": 0.0,
        "message": (
            f"Chuẩn bị {'Removal' if kind == 'removal' else 'Detector'} — "
            f"tiếp tục từ vòng {base} → {target}"
            if base else f"Chuẩn bị {'Removal' if kind == 'removal' else 'Detector'}…"
        ),
        "latest_checkpoint": resume,
    })
    _set(status="running", progress=0.0, loss=None, kind=kind,
         message=f"Chuẩn bị dữ liệu ({'Removal' if kind == 'removal' else 'Detector'}) "
                 f"— resume epoch {base}…")
    threading.Thread(
        target=_run, args=(epochs, resume, kind, target, base), daemon=True
    ).start()


def resume_if_interrupted() -> None:
    """
    On server startup: if a training job was cut off (crash/close), continue the
    remaining epochs from the last per-epoch checkpoint instead of losing progress.
    """
    job = _read_job()
    if not job or job.get("status") != "running":
        # Also honour status.json if train_job.json was lost but ckpt exists
        try:
            from training.checkpoints import read_status

            st = read_status()
            if st.get("status") == "running" and st.get("target_epochs"):
                job = {
                    "kind": st.get("kind", "detector"),
                    "target_epochs": int(st["target_epochs"]),
                    "status": "running",
                }
            else:
                return
        except Exception:  # noqa: BLE001
            return
    kind = job.get("kind", "detector")
    target = int(job.get("target_epochs", 0))
    base = _model_epochs(kind)
    remaining = target - base
    if remaining <= 0 or count_clean() < 4:
        _write_job(kind, target, "done")
        try:
            from training.checkpoints import write_status

            write_status({"status": "done", "kind": kind, "epoch": base,
                          "target_epochs": target, "progress": 1.0})
        except Exception:  # noqa: BLE001
            pass
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
        # The Removal net is deeper (has a 4th down block "d4"); the Detector net
        # does not. Catch a wrong-kind upload with a clear message BEFORE the scary
        # size-mismatch dump, so the user just picks the right tab.
        keys = list(ck.get("state", {}).keys())
        looks_removal = any(k.startswith("d4.") for k in keys)
        if kind == "detector" and looks_removal:
            raise ValueError("File này là model REMOVAL, nhưng bạn đang chọn ô 'Detector'. "
                             "Hãy bấm ô 'Removal — xóa & dựng nền' rồi tải lại.")
        if kind == "removal" and not looks_removal:
            raise ValueError("File này là model DETECTOR, nhưng bạn đang chọn ô 'Removal'. "
                             "Hãy bấm ô 'Detector — tìm watermark' rồi tải lại.")
        if kind == "removal":
            from training.removal import adapt_state_dict_in_ch, build_removal_net

            state = adapt_state_dict_in_ch(ck["state"], 4)
            build_removal_net(4).load_state_dict(state)
            ck["state"] = state
            ck["in_ch"] = 4
        else:
            from training.pipeline import _build_unet

            _build_unet().load_state_dict(ck["state"])
    except ValueError:
        os.remove(tmp)
        raise
    except Exception as exc:  # noqa: BLE001
        os.remove(tmp)
        raise ValueError(f"File .pt không đọc được hoặc hỏng: {str(exc)[:100]}") from exc
    # Re-write adapted checkpoint for removal so inference gets in_ch=4
    if kind == "removal":
        import torch

        torch.save(ck, tmp)
    os.replace(tmp, target)
    # Also seed a named checkpoint so resume finds it
    try:
        ep = int(ck.get("trained_epochs", 0))
        if ep > 0:
            from training.checkpoints import named_path
            import shutil

            dest = named_path(kind, ep)
            shutil.copy2(target, dest)
    except Exception:  # noqa: BLE001
        pass
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


def _run(epochs: int, resume: str | None, kind: str, target: int, base: int) -> None:
    label = "Removal" if kind == "removal" else "Detector"
    try:
        if kind == "removal":
            from training.removal import train_removal as _train
        else:
            from training.pipeline import train as _train

        def cb(p: float, loss: float):
            _set(progress=min(0.99, p), loss=round(loss, 4))

        def on_epoch(done_total: int, run_target: int, loss: float):
            _reload(kind)
            pct = (done_total - base) / max(1, target - base)
            _set(status="running", progress=min(0.99, pct), loss=loss,
                 message=f"{label}: đã lưu checkpoint vòng {done_total}/{target} "
                         f"(loss {loss}). An toàn nếu tắt máy.")

        info = _train(
            CLEAN_DIR, _model_path(kind), resume_from=resume,
            epochs=epochs, progress_cb=cb, epoch_cb=on_epoch,
            target_epochs=target, base_epochs=base,
        )
        _reload(kind)
        _write_job(kind, target, "done")
        from training.checkpoints import write_status

        write_status({
            "status": "done", "kind": kind,
            "epoch": info.get("trained_epochs", target),
            "target_epochs": target, "base_epochs": base, "progress": 1.0,
            "loss": info.get("final_loss"),
            "message": f"Xong {label}! Loss {info.get('final_loss')}, tổng "
                       f"{info.get('trained_epochs')} vòng.",
        })
        _set(status="done", progress=1.0, loss=info.get("final_loss"),
             message=f"Xong {label}! Loss {info.get('final_loss')}, tổng "
                     f"{info.get('trained_epochs')} vòng. Model đã tự tích hợp, sẵn sàng xóa.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("training failed")
        _write_job(kind, target, "done")  # a real error — don't auto-resume-loop
        _set(status="error", message=str(exc)[:300])
