"""ClearMark API — watermark removal with LaMa inpainting."""

from __future__ import annotations

import io
import logging
import os
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

from services import engine, sessions, storage, train_jobs, video_tasks
from services.lama import get_lama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clearmark")

MAX_BYTES = 10 * 1024 * 1024  # 10 MB per image
MAX_ZIP_BYTES = 100 * 1024 * 1024  # 100 MB archive
MAX_BATCH = 30
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/avif"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".bmp"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Report the ACTUAL compute device (auto-detected), not the env-var default —
    # a stale log here made it look like CPU even when the GPU was in use.
    from services.lama import _device

    dev = _device()
    gpu_name = dev.upper()
    if dev == "cuda":
        try:
            import torch

            gpu_name = f"CUDA · {torch.cuda.get_device_name(0)}"
        except Exception:  # noqa: BLE001
            pass
    logger.info("========================================")
    logger.info(" ClearMark compute device: %s", gpu_name)
    logger.info("========================================")
    logger.info("Warming up LaMa model on %s...", dev)
    try:
        get_lama()
        logger.info("LaMa ready on %s.", dev)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LaMa warm-up deferred: %s", exc)
    yield


app = FastAPI(title="ClearMark API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Không đọc được ảnh: {exc}") from exc
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA") if "A" in img.getbands() else img.convert("RGB")
    return img.convert("RGB")


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


async def _read_upload(file: UploadFile, label: str = "file") -> bytes:
    if file.content_type and file.content_type not in ALLOWED_TYPES and not file.content_type.startswith(
        "image/"
    ):
        raise HTTPException(status_code=400, detail=f"Định dạng {label} không hỗ trợ: {file.content_type}")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"{label} trống")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=400, detail=f"{label} vượt quá 10MB")
    return data


def _process_auto(raw: bytes, remove_text: bool = True) -> bytes:
    original = _load_image(raw)
    # engine routes to the GPU worker when configured, else local full-res tiler.
    result = engine.erase_auto(original, remove_text)
    return _to_png_bytes(result)


def _safe_stem(name: str, index: int) -> str:
    stem = Path(name).stem or f"image_{index + 1}"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:80]
    return safe or f"image_{index + 1}"


@app.get("/health")
def health():
    from services.lama import backend_name

    return {
        "status": "ok",
        "service": "clearmark-api",
        "inpaint": backend_name(),
        "engine": engine.backend_label(),
    }


def _truthy(v: Optional[str]) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


@app.post("/api/remove")
async def remove_watermark(
    image: UploadFile = File(...),
    remove_text: Optional[str] = Form(default="1"),
    return_mask: Optional[str] = Form(default="0"),
):
    """Auto-detect watermark regions and inpaint at full resolution.

    ``remove_text`` (default on) also targets OCR-detected text. Set to 0 to
    protect real printed text and only remove coloured logo/stamp watermarks.
    """
    raw = await _read_upload(image, "ảnh")
    try:
        png = _process_auto(raw, remove_text=_truthy(remove_text))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Xử lý thất bại: {exc}") from exc

    headers = {"X-ClearMark-Mode": "auto"}
    if return_mask in ("1", "true", "True"):
        headers["X-ClearMark-Mask"] = "1"
    return Response(content=png, media_type="image/png", headers=headers)


@app.post("/api/inpaint")
async def manual_inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
):
    """Inpaint using a user-drawn mask (white = remove)."""
    img_raw = await _read_upload(image, "ảnh")
    mask_raw = await _read_upload(mask, "mask")

    original = _load_image(img_raw)
    # Match the brushed mask to the FULL-resolution original (canvas may be scaled).
    mask_img = Image.open(io.BytesIO(mask_raw)).convert("L")
    if mask_img.size != original.size:
        mask_img = mask_img.resize(original.size, Image.Resampling.NEAREST)

    import cv2

    m = np.array(mask_img)
    _, m = cv2.threshold(m, 100, 255, cv2.THRESH_BINARY)
    # Light clean only — do NOT erode (that wiped manual brushes)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask_img = Image.fromarray(m, mode="L")

    if np.array(mask_img).max() < 128:
        raise HTTPException(
            status_code=400,
            detail="Chưa tô vùng cần xóa (hoặc nét quá mỏng). Hãy tô màu cam lên chữ mờ rồi bấm Xử lý.",
        )

    try:
        result = engine.erase(original, mask_img)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=_to_png_bytes(result),
        media_type="image/png",
        headers={"X-ClearMark-Mode": "manual"},
    )


@app.post("/api/detect-mask")
async def detect_mask(
    image: UploadFile = File(...),
    remove_text: Optional[str] = Form(default="1"),
):
    """Return the auto-generated mask as PNG (for UI preview)."""
    raw = await _read_upload(image, "ảnh")
    original = _load_image(raw)
    mask = engine.detect_mask(original, _truthy(remove_text))
    return Response(content=_to_png_bytes(mask.convert("RGB")), media_type="image/png")


def _mask_from_upload(mask_raw: bytes, size: tuple[int, int]) -> np.ndarray:
    """Uploaded brush PNG → full-res binary mask (uint8 {0,255})."""
    import cv2

    m = Image.open(io.BytesIO(mask_raw)).convert("L")
    if m.size != size:
        m = m.resize(size, Image.Resampling.NEAREST)
    arr = np.array(m)
    _, arr = cv2.threshold(arr, 100, 255, cv2.THRESH_BINARY)
    return arr


@app.post("/api/session")
async def session_create(image: UploadFile = File(...)):
    """Open a refine session; the pristine full-res image is kept server-side."""
    raw = await _read_upload(image, "ảnh")
    img = _load_image(raw)
    # Re-encode to PNG so the stored original is lossless for every refine pass.
    original_bytes = _to_png_bytes(img)
    sid = sessions.create(original_bytes, img.size)
    return {"session_id": sid, "width": img.size[0], "height": img.size[1]}


@app.post("/api/session/{sid}/erase")
async def session_erase(
    sid: str,
    mode: Optional[str] = Form(default="auto"),
    remove_text: Optional[str] = Form(default="1"),
    mask: Optional[UploadFile] = File(default=None),
):
    """
    Add to the accumulated mask, then inpaint from the PRISTINE original.

    ``mode=auto``  → run auto-detection and add it to the mask.
    a ``mask`` file (brush strokes, white = remove) is always added if present.
    Because we always start from the stored original, refining never stacks
    inpaint-on-inpaint, so detail does not degrade across passes.
    """
    try:
        original = sessions.original_image(sid)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Phiên đã hết hạn — hãy tải lại ảnh.") from exc

    added = False
    if mode is None or mode == "auto":
        auto = engine.detect_mask(original, _truthy(remove_text))
        if np.array(auto).max() >= 128:
            sessions.accumulate(sid, np.array(auto))
            added = True
    if mask is not None and mask.filename:
        mask_raw = await _read_upload(mask, "mask")
        sessions.accumulate(sid, _mask_from_upload(mask_raw, original.size))
        added = True

    acc = sessions.get_mask(sid)
    if acc is None or acc.max() < 128:
        detail = (
            "Không phát hiện watermark. Hãy tô tay vùng cần xóa rồi bấm Xử lý."
            if not added
            else "Chưa có vùng nào để xóa."
        )
        raise HTTPException(status_code=422, detail=detail)

    try:
        result = engine.erase(original, Image.fromarray(acc, mode="L"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(
        content=_to_png_bytes(result),
        media_type="image/png",
        headers={"X-ClearMark-Mode": "session", "X-ClearMark-Session": sid},
    )


@app.get("/api/session/{sid}/mask")
def session_mask(sid: str):
    """Current accumulated mask as PNG (white = will be removed) for UI overlay."""
    try:
        acc = sessions.get_mask(sid)
        w, h = sessions.size(sid)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Phiên đã hết hạn.") from exc
    if acc is None:
        acc = np.zeros((h, w), dtype=np.uint8)
    return Response(content=_to_png_bytes(Image.fromarray(acc, mode="L")), media_type="image/png")


@app.post("/api/session/{sid}/reset-mask")
def session_reset(sid: str):
    """Clear the accumulated mask (start the erase over from the original)."""
    try:
        sessions.reset_mask(sid)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Phiên đã hết hạn.") from exc
    return {"ok": True}


@app.delete("/api/session/{sid}")
def session_delete(sid: str):
    sessions.drop(sid)
    return {"ok": True}


MAX_VIDEO_BYTES = 200 * 1024 * 1024  # 200 MB clip


@app.post("/api/video/upload")
async def video_upload(video: UploadFile = File(...)):
    """Upload a clip and open an async job. Returns a task_id to poll."""
    data = await video.read()
    if not data:
        raise HTTPException(status_code=400, detail="Video trống")
    if len(data) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=400, detail="Video vượt quá 200MB")
    ct = (video.content_type or "").lower()
    if "mp4" not in ct and not (video.filename or "").lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ MP4 (H.264).")
    task = video_tasks.create(data, "mp4")
    return task.public()


@app.post("/api/video/tasks")
async def video_start(
    task_id: str = Form(...),
    mask: Optional[UploadFile] = File(default=None),
):
    """Start processing an uploaded clip (optionally with a static brush mask)."""
    mask_bytes = None
    if mask is not None and mask.filename:
        mask_bytes = await mask.read()
    try:
        video_tasks.start(task_id, mask_bytes)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Task không tồn tại hoặc đã hết hạn.") from exc
    t = video_tasks.get(task_id)
    return t.public() if t else {"task_id": task_id, "status": "QUEUED"}


@app.get("/api/video/tasks/{task_id}")
def video_status(task_id: str):
    t = video_tasks.get(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task không tồn tại hoặc đã hết hạn.")
    return t.public()


@app.get("/api/video/tasks/{task_id}/download")
def video_download(task_id: str):
    t = video_tasks.get(task_id)
    if not t or t.status != "DONE" or not t.output_key:
        raise HTTPException(status_code=404, detail="Chưa có kết quả.")
    # If storage exposes a direct URL (R2), redirect; else stream the bytes.
    direct = storage.public_url(t.output_key)
    if direct:
        from fastapi.responses import RedirectResponse

        return RedirectResponse(direct)
    data = storage.get(t.output_key)
    if data is None:
        raise HTTPException(status_code=404, detail="Kết quả đã hết hạn (quá 1 giờ).")
    return Response(
        content=data,
        media_type="video/mp4",
        headers={"Content-Disposition": 'attachment; filename="clearmark-video.mp4"'},
    )


@app.post("/api/train/upload")
async def train_upload(images: List[UploadFile] = File(...)):
    """Add clean (watermark-free) images to the training set."""
    n = 0
    for f in images:
        if not f.filename:
            continue
        data = await f.read()
        if not data or len(data) > MAX_BYTES:
            continue
        train_jobs.save_clean(f.filename, data)
        n += 1
    return {"added": n, **train_jobs.status()}


@app.post("/api/train/start")
async def train_start(
    epochs: Optional[str] = Form(default="8"),
    fresh: Optional[str] = Form(default="0"),
    kind: Optional[str] = Form(default="detector"),
):
    try:
        train_jobs.start(int(epochs or 8), fresh=_truthy(fresh), kind=kind or "detector")
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return train_jobs.status()


@app.post("/api/train/scrape")
async def train_scrape(
    count: Optional[str] = Form(default="200"),
    source: Optional[str] = Form(default="picsum"),
    api_key: Optional[str] = Form(default=None),
    query: Optional[str] = Form(default=None),
):
    """Auto-download clean stock images (picsum/pexels/unsplash; query = people/portrait)."""
    try:
        n = max(1, min(5000, int(count or 200)))
        train_jobs.scrape(n, source or "picsum", (api_key or "").strip() or None,
                          (query or "").strip() or None)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return train_jobs.status()


@app.post("/api/train/upload-model")
async def train_upload_model(
    model: UploadFile = File(...),
    kind: Optional[str] = Form(default="detector"),
):
    """Upload a previously-downloaded .pt to continue training from it."""
    data = await model.read()
    if not data:
        raise HTTPException(status_code=400, detail="File trống")
    try:
        train_jobs.set_model(data, kind or "detector")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return train_jobs.status()


@app.get("/api/train/status")
def train_status():
    return train_jobs.status()


@app.post("/api/train/clear")
def train_clear():
    train_jobs.clear_clean()
    return train_jobs.status()


@app.post("/api/train/watermarks")
async def train_watermarks(images: List[UploadFile] = File(...)):
    """Add logo/sticker PNGs (with transparency) to the watermark library to learn."""
    n = 0
    for f in images:
        if not f.filename:
            continue
        data = await f.read()
        if not data or len(data) > MAX_BYTES:
            continue
        train_jobs.save_watermark(f.filename, data)
        n += 1
    return {"added": n, **train_jobs.status()}


@app.post("/api/train/watermarks/clear")
def train_watermarks_clear():
    train_jobs.clear_watermarks()
    return train_jobs.status()


@app.get("/api/train/preview")
def train_preview():
    """Montage of freshly-generated training samples (mask outlined red)."""
    from training.pipeline import preview_samples

    if train_jobs.count_clean() < 1:
        raise HTTPException(status_code=400, detail="Hãy tải vài ảnh sạch lên trước.")
    grid = preview_samples(train_jobs.CLEAN_DIR, n=6)
    return Response(content=_to_png_bytes(Image.fromarray(grid)), media_type="image/png")


@app.get("/api/train/evaluate")
def train_evaluate():
    """Run the trained model on held-out synthetic watermarks → proof montage + metrics."""
    if not os.path.exists(train_jobs.MODEL_OUT):
        raise HTTPException(status_code=400, detail="Chưa có model. Hãy train trước.")
    from training.eval import evaluate

    montage, metrics = evaluate(train_jobs.CLEAN_DIR, n=6)
    headers = {f"X-Eval-{k}": str(v) for k, v in metrics.items()}
    return Response(content=_to_png_bytes(Image.fromarray(montage)),
                    media_type="image/png", headers=headers)


@app.get("/api/train/model")
def train_model(kind: str = "detector"):
    path = train_jobs.MODEL_REMOVAL if kind == "removal" else train_jobs.MODEL_OUT
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Chưa có model. Hãy train trước.")
    with open(path, "rb") as f:
        data = f.read()
    fname = "wm_remover.pt" if kind == "removal" else "wm_detector.pt"
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.post("/api/remove-batch")
async def remove_batch(
    images: Optional[List[UploadFile]] = File(default=None),
    archive: Optional[UploadFile] = File(default=None),
):
    """
    Process many images (multipart files and/or a .zip of images).
    Returns a ZIP of cleaned PNGs (+ errors.txt if any failed).
    """
    jobs: list[tuple[str, bytes]] = []

    if archive is not None and archive.filename:
        data = await archive.read()
        if len(data) > MAX_ZIP_BYTES:
            raise HTTPException(status_code=400, detail="File ZIP vượt quá 100MB")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    ext = Path(info.filename).suffix.lower()
                    if ext not in IMAGE_EXTS:
                        continue
                    if info.file_size > MAX_BYTES:
                        continue
                    raw = zf.read(info)
                    if raw:
                        jobs.append((Path(info.filename).name, raw))
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail="File ZIP không hợp lệ") from exc

    if images:
        for f in images:
            if not f.filename:
                continue
            raw = await f.read()
            if not raw:
                continue
            if len(raw) > MAX_BYTES:
                continue
            jobs.append((f.filename, raw))

    if not jobs:
        raise HTTPException(
            status_code=400,
            detail="Không có ảnh hợp lệ. Hãy chọn nhiều ảnh hoặc 1 file ZIP chứa ảnh.",
        )
    if len(jobs) > MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"Tối đa {MAX_BATCH} ảnh mỗi lần.")

    out_buf = io.BytesIO()
    errors: list[str] = []
    ok = 0
    with zipfile.ZipFile(out_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        used_names: set[str] = set()
        for i, (name, raw) in enumerate(jobs):
            stem = _safe_stem(name, i)
            out_name = f"{stem}_clearmark.png"
            n = 1
            while out_name in used_names:
                n += 1
                out_name = f"{stem}_{n}_clearmark.png"
            used_names.add(out_name)
            try:
                png = _process_auto(raw)
                zf.writestr(out_name, png)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}")

        if errors:
            zf.writestr("errors.txt", "\n".join(errors) + "\n")

    if ok == 0:
        raise HTTPException(
            status_code=422,
            detail="Không xử lý được ảnh nào. " + (errors[0] if errors else ""),
        )

    return Response(
        content=out_buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="clearmark-batch.zip"',
            "X-ClearMark-Mode": "batch",
            "X-ClearMark-OK": str(ok),
            "X-ClearMark-Failed": str(len(errors)),
        },
    )
