"""
ClearMark GPU worker — serverless, scale-to-zero, pay-per-second.

Deploy with Modal (https://modal.com):

    pip install modal
    modal token new
    modal deploy worker/modal_app.py

Modal prints a URL like  https://<you>--clearmark-worker-web.modal.run
Put it in the API environment:

    GPU_WORKER_URL=https://<you>--clearmark-worker-web.modal.run
    GPU_WORKER_TOKEN=<the same secret you set below>

The API (apps/api/services/engine.py) then routes detection + removal here and
falls back to local CPU automatically if the worker is unreachable.

Why Modal: pure-Python serverless GPU. The container spins up on demand, holds
the models warm for a few minutes, then scales to zero — so a low-traffic site
pays only for the seconds it actually inpaints (~<1¢/image on an L4/A10), never
for idle time. RunPod Serverless / Replicate are drop-in alternatives; the same
handler functions apply, only the wrapper differs.

Pipeline (mirrors the dewatermark.ai teardown):
  detect : Florence-2 open-vocabulary grounding ("watermark, logo, translucent
           text, stamp") -> boxes -> SAM 2 tightens each box to a pixel mask.
  erase  : predict_mode "3.0" -> LaMa full-res tiling (fast, ~1 credit)
           predict_mode "4.0" -> SDXL inpainting (hard/large, ~3 credits)
           then Real-ESRGAN sharpening ONLY over the mask to hide inpaint softness.
  video  : one static mask for the clip -> ProPainter temporal-consistent inpaint.

Everything is guarded so a missing optional model degrades instead of crashing.
"""

from __future__ import annotations

import io
import os

import modal

APP_NAME = "clearmark-worker"

# ---- container image: CUDA torch + the model stack --------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.49.0",
        "accelerate>=1.0",
        "einops",
        "timm",
        "opencv-python-headless",
        "pillow",
        "numpy<2",
        "diffusers>=0.31",
        "safetensors",
        "simple-lama-inpainting==0.1.2",
        "realesrgan==0.3.0",
        "basicsr==1.4.2",
        "huggingface_hub>=0.25",
        "python-multipart",
        "fastapi[standard]",
        "scipy",
        "av",
    )
    # ProPainter for temporal-consistent video inpainting (static mask per clip).
    .run_commands(
        "git clone --depth 1 https://github.com/sczhou/ProPainter /opt/ProPainter || true",
        "pip install -r /opt/ProPainter/requirements.txt || true",
    )
)

app = modal.App(APP_NAME)

# Cache HF weights in a persistent volume so cold starts don't re-download.
cache = modal.Volume.from_name("clearmark-cache", create_if_missing=True)
CACHE_DIR = "/cache"

# Shared secret; set with:  modal secret create clearmark-worker-token TOKEN=...
try:
    _secret = modal.Secret.from_name("clearmark-worker-token")
except Exception:  # noqa: BLE001
    _secret = modal.Secret.from_dict({"TOKEN": ""})

GPU = os.environ.get("CLEARMARK_GPU", "L4")  # L4 is cheapest that fits SDXL


@app.cls(
    image=image,
    gpu=GPU,
    volumes={CACHE_DIR: cache},
    secrets=[_secret],
    scaledown_window=180,   # keep warm 3 min after last call, then scale to zero
    timeout=600,
)
class Worker:
    @modal.enter()
    def load(self):
        import torch

        os.environ.setdefault("HF_HOME", CACHE_DIR)
        os.environ.setdefault("TORCH_HOME", CACHE_DIR)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._florence = None
        self._sam = None
        self._lama = None
        self._sdxl = None
        self._esrgan = None

    # ---- lazy model loaders (first use pays, then cached in-container) -------
    def florence(self):
        if self._florence is None:
            from transformers import AutoModelForCausalLM, AutoProcessor

            mid = "microsoft/Florence-2-large"
            self._florence = (
                AutoModelForCausalLM.from_pretrained(
                    mid, trust_remote_code=True, torch_dtype=self.dtype
                ).to(self.device).eval(),
                AutoProcessor.from_pretrained(mid, trust_remote_code=True),
            )
        return self._florence

    def sam(self):
        if self._sam is None:
            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                self._sam = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
            except Exception:  # noqa: BLE001 — SAM2 optional; box masks still work
                self._sam = False
        return self._sam or None

    def lama(self):
        if self._lama is None:
            from simple_lama_inpainting import SimpleLama

            self._lama = SimpleLama(device=self.device)
        return self._lama

    def sdxl(self):
        if self._sdxl is None:
            import torch
            from diffusers import StableDiffusionXLInpaintPipeline

            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=self.dtype,
                variant="fp16" if self.device == "cuda" else None,
            ).to(self.device)
            pipe.set_progress_bar_config(disable=True)
            self._sdxl = pipe
        return self._sdxl

    def esrgan(self):
        if self._esrgan is None:
            try:
                from realesrgan import RealESRGANer
                from basicsr.archs.rrdbnet_arch import RRDBNet

                arch = RRDBNet(3, 3, 64, 23, num_grow_ch=32, scale=4)
                self._esrgan = RealESRGANer(
                    scale=4,
                    model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                    model=arch,
                    half=self.device == "cuda",
                )
            except Exception:  # noqa: BLE001
                self._esrgan = False
        return self._esrgan or None

    # ---- detection ----------------------------------------------------------
    def detect_mask(self, image_bytes: bytes, remove_text: bool):
        import cv2
        import numpy as np
        import torch
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        W, H = img.size
        model, proc = self.florence()

        phrase = "watermark, translucent logo, stamp, repetitive overlay"
        if remove_text:
            phrase += ", text"
        prompt = "<OPEN_VOCABULARY_DETECTION>" + phrase
        inputs = proc(text=prompt, images=img, return_tensors="pt").to(self.device, self.dtype)
        with torch.inference_mode():
            ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )
        text = proc.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = proc.post_process_generation(
            text, task="<OPEN_VOCABULARY_DETECTION>", image_size=(W, H)
        )
        boxes = parsed.get("<OPEN_VOCABULARY_DETECTION>", {}).get("bboxes", [])

        mask = np.zeros((H, W), np.uint8)
        predictor = self.sam()
        if predictor is not None and boxes:
            predictor.set_image(np.array(img))
            for (x0, y0, x1, y1) in boxes:
                m, _, _ = predictor.predict(
                    box=np.array([x0, y0, x1, y1]), multimask_output=False
                )
                mask[m[0].astype(bool)] = 255
        else:
            for (x0, y0, x1, y1) in boxes:
                cv2.rectangle(mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)

        return mask, (W, H)

    # ---- removal ------------------------------------------------------------
    def _sharpen_mask_region(self, rgb, mask_bin):
        import cv2
        import numpy as np

        up = self.esrgan()
        if up is None:
            return rgb
        try:
            enh, _ = up.enhance(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), outscale=1)
            enh = cv2.cvtColor(enh, cv2.COLOR_BGR2RGB)
        except Exception:  # noqa: BLE001
            return rgb
        a = cv2.GaussianBlur((mask_bin > 0).astype(np.float32), (0, 0), 2.0)[..., None]
        return (enh.astype(np.float32) * a + rgb.astype(np.float32) * (1 - a)).clip(0, 255).astype(
            np.uint8
        )

    def erase(self, image_bytes: bytes, mask_bytes: bytes, predict_mode: str):
        import cv2
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        W, H = img.size
        m = Image.open(io.BytesIO(mask_bytes)).convert("L").resize((W, H), Image.NEAREST)
        rgb = np.array(img)
        mask_bin = (np.array(m) > 100).astype(np.uint8) * 255
        if mask_bin.max() == 0:
            return _png(img)

        if predict_mode == "4.0":
            out = self._erase_sdxl(img, Image.fromarray(mask_bin, "L"))
        else:
            out = self._erase_lama(rgb, mask_bin)

        out = self._sharpen_mask_region(out, mask_bin)
        # bit-exact original outside the mask
        final = rgb.copy()
        sel = mask_bin > 0
        final[sel] = out[sel]
        return _png(Image.fromarray(final))

    def _erase_lama(self, rgb, mask_bin):
        import cv2
        import numpy as np
        from PIL import Image

        lama = self.lama()
        grow = cv2.dilate(mask_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
        res = lama(Image.fromarray(rgb), Image.fromarray(grow, "L"))
        res = np.array(res.convert("RGB").resize((rgb.shape[1], rgb.shape[0])))
        return res

    def _erase_sdxl(self, img, mask_img):
        import numpy as np
        from PIL import Image

        pipe = self.sdxl()
        W, H = img.size
        # SDXL inpaints at 1024; work on a padded crop of the mask for detail.
        bbox = mask_img.getbbox()
        if bbox is None:
            return np.array(img)
        x0, y0, x1, y1 = bbox
        pad = 64
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
        crop = img.crop((x0, y0, x1, y1))
        cmask = mask_img.crop((x0, y0, x1, y1))
        cw, ch = crop.size
        gen = pipe(
            prompt="clean background, seamless, photorealistic, no text, no watermark",
            negative_prompt="text, watermark, logo, letters, blur, artifacts",
            image=crop.resize((1024, 1024)),
            mask_image=cmask.resize((1024, 1024)),
            num_inference_steps=25,
            strength=0.99,
            guidance_scale=7.0,
        ).images[0].resize((cw, ch))
        out = np.array(img).copy()
        out[y0:y1, x0:x1] = np.array(gen.convert("RGB"))
        return out

    # ---- Modal remote methods (called by the ASGI endpoints below) ----------
    @modal.method()
    def run_detect(self, image_bytes: bytes, remove_text: bool) -> bytes:
        import numpy as np
        from PIL import Image

        mask, _ = self.detect_mask(image_bytes, remove_text)
        return _png(Image.fromarray(mask, "L"))

    @modal.method()
    def run_erase(self, image_bytes: bytes, mask_bytes: bytes, predict_mode: str) -> bytes:
        return self.erase(image_bytes, mask_bytes, predict_mode)

    @modal.method()
    def run_erase_auto(self, image_bytes: bytes, remove_text: bool, predict_mode: str) -> bytes:
        from PIL import Image

        mask, _ = self.detect_mask(image_bytes, remove_text)
        return self.erase(image_bytes, _png(Image.fromarray(mask, "L")), predict_mode)

    # ---- video --------------------------------------------------------------
    @modal.method()
    def run_video(self, video_bytes: bytes, mask_bytes: bytes | None):
        """
        Temporal-consistent video watermark removal.

        Assumes a STATIC watermark (the common case) so one mask covers the whole
        clip: if no brush mask is supplied we auto-detect on a mid-clip frame with
        Florence-2 + SAM2. Frames are inpainted by ProPainter (propagates context
        across time → no per-frame flicker); the original audio is muxed back.
        Falls back to per-frame LaMa if the ProPainter checkout is unavailable.

        Returns (mp4_bytes, has_watermark: bool).
        """
        import glob
        import subprocess
        import tempfile

        import cv2
        import numpy as np
        from PIL import Image

        work = tempfile.mkdtemp(prefix="clearmark_vid_")
        in_path = os.path.join(work, "in.mp4")
        with open(in_path, "wb") as f:
            f.write(video_bytes)

        frames_dir = os.path.join(work, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, os.path.join(frames_dir, "%05d.png")],
            check=True, capture_output=True,
        )
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        if not frame_files:
            raise RuntimeError("Không tách được frame từ video.")
        H, W = cv2.imread(frame_files[0]).shape[:2]

        # Build one static mask for the whole clip.
        if mask_bytes:
            m = Image.open(io.BytesIO(mask_bytes)).convert("L").resize((W, H), Image.NEAREST)
            mask = (np.array(m) > 100).astype(np.uint8) * 255
        else:
            mid = frame_files[len(frame_files) // 2]
            with open(mid, "rb") as f:
                mask, _ = self.detect_mask(f.read(), remove_text=False)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        has_wm = bool(mask.max() > 0)
        if not has_wm:
            return video_bytes, False  # nothing detected → return original

        mask_dir = os.path.join(work, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        grown = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 1)
        for fp in frame_files:
            cv2.imwrite(os.path.join(mask_dir, os.path.basename(fp)), grown)

        out_frames = os.path.join(work, "out")
        os.makedirs(out_frames, exist_ok=True)
        used_propainter = self._run_propainter(frames_dir, mask_dir, out_frames)
        if not used_propainter:
            self._video_lama_fallback(frame_files, grown, out_frames)

        # Reassemble at source fps + carry original audio.
        fps = self._probe_fps(in_path)
        out_path = os.path.join(work, "out.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps),
             "-i", os.path.join(out_frames, "%05d.png"),
             "-i", in_path, "-map", "0:v", "-map", "1:a?",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", out_path],
            check=True, capture_output=True,
        )
        with open(out_path, "rb") as f:
            return f.read(), has_wm

    def _run_propainter(self, frames_dir, mask_dir, out_dir) -> bool:
        import glob
        import shutil
        import subprocess

        script = "/opt/ProPainter/inference_propainter.py"
        if not os.path.exists(script):
            return False
        try:
            res_root = os.path.join(os.path.dirname(out_dir), "pp_results")
            subprocess.run(
                ["python", script, "--video", frames_dir, "--mask", mask_dir,
                 "--output", res_root, "--save_frames", "--fp16"],
                check=True, cwd="/opt/ProPainter", capture_output=True,
            )
            produced = sorted(glob.glob(os.path.join(res_root, "**", "*.png"), recursive=True))
            src = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
            if len(produced) < len(src):
                return False
            for i, p in enumerate(produced[: len(src)]):
                shutil.copy(p, os.path.join(out_dir, os.path.basename(src[i])))
            return True
        except Exception:  # noqa: BLE001
            return False

    def _video_lama_fallback(self, frame_files, mask_grown, out_dir):
        """Per-frame LaMa with a shared static mask (deterministic → low flicker)."""
        import cv2
        import numpy as np
        from PIL import Image

        lama = self.lama()
        m = Image.fromarray(mask_grown, "L")
        for fp in frame_files:
            rgb = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
            res = lama(Image.fromarray(rgb), m)
            res = np.array(res.convert("RGB").resize((rgb.shape[1], rgb.shape[0])))
            final = rgb.copy()
            sel = mask_grown > 0
            final[sel] = res[sel]
            cv2.imwrite(os.path.join(out_dir, os.path.basename(fp)),
                        cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

    def _probe_fps(self, path) -> float:
        import subprocess

        try:
            out = subprocess.run(
                ["ffprobe", "-v", "0", "-of", "csv=p=0", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate", path],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            num, den = out.split("/")
            return max(1.0, float(num) / float(den))
        except Exception:  # noqa: BLE001
            return 24.0


def _png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# HTTP surface — matches the contract in apps/api/services/engine.py
# --------------------------------------------------------------------------- #
@app.function(image=image, secrets=[_secret], scaledown_window=120)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.responses import Response

    api = FastAPI(title="ClearMark GPU Worker")

    def _auth(authorization: str | None):
        want = os.environ.get("TOKEN", "")
        if want and authorization != f"Bearer {want}":
            raise HTTPException(status_code=401, detail="bad token")

    @api.get("/health")
    def health():
        return {"status": "ok", "gpu": GPU}

    @api.post("/detect")
    async def detect(
        image: UploadFile = File(...),
        remove_text: str = Form("0"),
        authorization: str | None = Header(default=None),
    ):
        _auth(authorization)
        data = await image.read()
        png = Worker().run_detect.remote(data, remove_text in ("1", "true", "True"))
        return Response(content=png, media_type="image/png")

    @api.post("/erase")
    async def erase(
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
        predict_mode: str = Form("3.0"),
        authorization: str | None = Header(default=None),
    ):
        _auth(authorization)
        img_b = await image.read()
        mask_b = await mask.read()
        png = Worker().run_erase.remote(img_b, mask_b, predict_mode)
        return Response(content=png, media_type="image/png")

    @api.post("/video")
    async def video(
        video: UploadFile = File(...),
        mask: UploadFile | None = File(default=None),
        authorization: str | None = Header(default=None),
    ):
        _auth(authorization)
        vb = await video.read()
        mb = await mask.read() if mask is not None else None
        out_bytes, has_wm = Worker().run_video.remote(vb, mb)
        return Response(
            content=out_bytes,
            media_type="video/mp4",
            headers={"X-Has-Watermark": "1" if has_wm else "0"},
        )

    return api
