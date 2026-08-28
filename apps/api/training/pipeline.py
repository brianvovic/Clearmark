"""
Watermark DETECTOR training — GENERAL, not one logo.

The detector must learn "watermark-ness", so we synthesise DIVERSE watermarks and
paste them on clean photos, producing (watermarked, mask) pairs for free:

  • logo/sticker PNGs from a library — the user uploads many (theirs + others');
  • random TEXT watermarks — random strings/URLs, random fonts, sizes, single OR
    multi-colour, sometimes neon-outline, rotated;
  • random sticker blobs — coloured shapes to mimic "dán sticker che".

Each training image gets 1–3 instances at random position / scale / opacity /
rotation, sometimes tiled. A small U-Net learns to segment all of them → works on
arbitrary logos, text and stickers, anywhere, at any opacity.

Provability: `preview_samples()` renders what the training data looks like, and
`evaluate()` (in eval.py) runs the trained model on held-out synthetic watermarks
and reports how many it removes.

Torch is imported lazily so the API runs without touching training.
"""

from __future__ import annotations

import glob
import logging
import os
import random
import string

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("clearmark.train")

IMG_SIZE = 384
_ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
WATERMARKS_DIR = os.path.join(_ASSETS, "watermarks")
os.makedirs(WATERMARKS_DIR, exist_ok=True)

_WORDS = ["hoalau.xyz", "SAMPLE", "©2024", "PREVIEW", "watermark", "DEMO", "mysite.com",
          "NOT FOR SALE", "@user_name", "COPYRIGHT", "xyz.com", "FREE", "★VIP★", "18+",
          "sample.net", "do not copy", "PROOF", "★", "♥", "웹사이트"]
_FONT_CACHE: list[str] | None = None
_FONT_OBJ: dict = {}


def _get_font(path: str | None, size: int):
    key = (path, size)
    f = _FONT_OBJ.get(key)
    if f is None:
        try:
            f = ImageFont.truetype(path, size) if path else ImageFont.load_default()
        except Exception:  # noqa: BLE001
            f = ImageFont.load_default()
        if len(_FONT_OBJ) < 800:
            _FONT_OBJ[key] = f
    return f


# --------------------------------------------------------------------------- #
# watermark sources
# --------------------------------------------------------------------------- #
def _fonts() -> list[str]:
    global _FONT_CACHE
    if _FONT_CACHE is not None:
        return _FONT_CACHE
    dirs = [r"C:\Windows\Fonts", "/usr/share/fonts", "/Library/Fonts",
            os.path.expanduser("~/.fonts")]
    fonts: list[str] = []
    for d in dirs:
        if os.path.isdir(d):
            for ext in ("*.ttf", "*.otf", "*.TTF"):
                fonts += glob.glob(os.path.join(d, "**", ext), recursive=True)
    _FONT_CACHE = fonts[:200] if fonts else []
    return _FONT_CACHE


def _rand_color(rng: random.Random) -> tuple[int, int, int]:
    # bias toward vivid colours (watermarks are usually saturated)
    h = rng.randint(0, 179)
    hsv = np.uint8([[[h, rng.randint(120, 255), rng.randint(180, 255)]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(r), int(g), int(b)


def _text_watermark(rng: random.Random) -> Image.Image:
    txt = rng.choice(_WORDS)
    if rng.random() < 0.3:
        txt = "".join(rng.choice(string.ascii_letters + ".·/") for _ in range(rng.randint(4, 12)))
    size = rng.randint(28, 96)
    font = _get_font(rng.choice(_fonts()) if _fonts() else None, size)
    dummy = Image.new("RGBA", (10, 10))
    d = ImageDraw.Draw(dummy)
    try:
        bbox = d.textbbox((0, 0), txt, font=font)
    except Exception:  # noqa: BLE001
        bbox = (0, 0, size * len(txt) // 2, size)
    tw, th = bbox[2] - bbox[0] + 20, bbox[3] - bbox[1] + 20
    img = Image.new("RGBA", (max(8, tw), max(8, th)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    multicolor = rng.random() < 0.4
    outline = rng.random() < 0.4
    if multicolor:
        x = 10
        for ch in txt:
            c = _rand_color(rng)
            d.text((x, 8), ch, font=font, fill=(*c, 255),
                   stroke_width=2 if outline else 0, stroke_fill=(255, 255, 255, 255))
            x += d.textlength(ch, font=font)
    else:
        c = _rand_color(rng)
        d.text((10, 8), txt, font=font, fill=(*c, 255),
               stroke_width=2 if outline else 0, stroke_fill=(255, 255, 255, 255))
    if rng.random() < 0.5:
        img = img.rotate(rng.uniform(-30, 30), expand=True, resample=Image.BICUBIC)
    return img


def _sticker_watermark(rng: random.Random) -> Image.Image:
    s = rng.randint(60, 180)
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(1, 4)):
        c = _rand_color(rng)
        a = rng.randint(120, 255)
        x0, y0 = rng.randint(0, s // 2), rng.randint(0, s // 2)
        x1, y1 = rng.randint(s // 2, s), rng.randint(s // 2, s)
        if rng.random() < 0.5:
            d.ellipse([x0, y0, x1, y1], fill=(*c, a))
        else:
            d.rounded_rectangle([x0, y0, x1, y1], radius=rng.randint(4, 20), fill=(*c, a))
    return img


def load_wm_assets() -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for f in glob.glob(os.path.join(WATERMARKS_DIR, "*")):
        try:
            out.append(np.array(Image.open(f).convert("RGBA")))
        except Exception:  # noqa: BLE001
            pass
    return out


def _pick_watermark(assets: list[np.ndarray], rng: random.Random) -> np.ndarray:
    r = rng.random()
    if assets and r < 0.5:
        return assets[rng.randrange(len(assets))]
    if r < 0.8:
        return np.array(_text_watermark(rng))
    return np.array(_sticker_watermark(rng))


# --------------------------------------------------------------------------- #
# compositing
# --------------------------------------------------------------------------- #
def _stamp(out: np.ndarray, mask: np.ndarray, wm: np.ndarray, x: int, y: int, opacity: float):
    H, W = out.shape[:2]
    th, tw = wm.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + tw), min(H, y + th)
    if x1 <= x0 or y1 <= y0:
        return
    sub = wm[y0 - y:y1 - y, x0 - x:x1 - x]
    a = (sub[..., 3:4].astype(np.float32) / 255.0) * opacity
    col = sub[..., :3].astype(np.float32)
    out[y0:y1, x0:x1] = (1 - a) * out[y0:y1, x0:x1] + a * col
    mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], (a[..., 0] > 0.05) * 255.0)


def synthesize(clean: np.ndarray, assets: list[np.ndarray], rng: random.Random):
    """Return (watermarked_uint8, mask_uint8) with 1–3 diverse watermarks."""
    H, W = clean.shape[:2]
    out = clean.astype(np.float32).copy()
    mask = np.zeros((H, W), np.float32)
    for _ in range(rng.randint(1, 3)):
        wm = _pick_watermark(assets, rng)
        lh, lw = wm.shape[:2]
        target_w = int(W * rng.uniform(0.15, 0.6))
        target_h = max(8, int(lh * target_w / max(1, lw)))
        wm_r = cv2.resize(wm, (max(8, target_w), target_h), interpolation=cv2.INTER_AREA)
        opacity = rng.uniform(0.25, 0.9)
        if rng.random() < 0.25:  # tiled
            gx, gy = int(target_w * 0.5), int(target_h * 0.8)
            for yy in range(rng.randint(0, gy), H, target_h + gy):
                for xx in range(rng.randint(0, gx), W, wm_r.shape[1] + gx):
                    _stamp(out, mask, wm_r, xx, yy, opacity)
        else:
            x = rng.randint(-target_w // 6, max(1, W - target_w + target_w // 6))
            y = rng.randint(-target_h // 6, max(1, H - target_h + target_h // 6))
            _stamp(out, mask, wm_r, x, y, opacity)
    return np.clip(out, 0, 255).astype(np.uint8), mask.astype(np.uint8)


def _list_clean(clean_dir: str) -> list[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    files: list[str] = []
    for e in exts:
        files += glob.glob(os.path.join(clean_dir, "**", e), recursive=True)
    return files


def make_sample(clean_path: str, assets: list[np.ndarray], rng: random.Random):
    img = Image.open(clean_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    clean = np.array(img)
    if rng.random() < 0.1:  # clean negative → fewer false positives
        return clean, np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
    return synthesize(clean, assets, rng)


def preview_samples(clean_dir: str, n: int = 6) -> np.ndarray:
    """A montage (RGB) of generated training samples with the mask outlined red."""
    assets = load_wm_assets()
    files = _list_clean(clean_dir)
    rng = random.Random()
    tiles = []
    for i in range(n):
        src = files[rng.randrange(len(files))] if files else None
        if src:
            clean = np.array(Image.open(src).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
        else:
            clean = np.full((IMG_SIZE, IMG_SIZE, 3), 150, np.uint8)
        img, mask = synthesize(clean, assets, random.Random(rng.randint(0, 1 << 30)))
        edge = cv2.dilate(mask, np.ones((3, 3), np.uint8)) - mask
        vis = img.copy()
        vis[edge > 0] = [255, 0, 0]
        tiles.append(vis)
    cols = 3
    rows = (n + cols - 1) // cols
    grid = np.full((rows * IMG_SIZE, cols * IMG_SIZE, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * IMG_SIZE:(r + 1) * IMG_SIZE, c * IMG_SIZE:(c + 1) * IMG_SIZE] = t
    return grid


# --------------------------------------------------------------------------- #
# model + training
# --------------------------------------------------------------------------- #
def _build_unet():
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.d1, self.d2, self.d3 = block(3, 32), block(32, 64), block(64, 128)
            self.pool = nn.MaxPool2d(2)
            self.mid = block(128, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
            self.u3 = block(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
            self.u2 = block(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
            self.u1 = block(64, 32)
            self.out = nn.Conv2d(32, 1, 1)

        def forward(self, x):
            import torch

            c1 = self.d1(x)
            c2 = self.d2(self.pool(c1))
            c3 = self.d3(self.pool(c2))
            m = self.mid(self.pool(c3))
            x = self.u3(torch.cat([self.up3(m), c3], 1))
            x = self.u2(torch.cat([self.up2(x), c2], 1))
            x = self.u1(torch.cat([self.up1(x), c1], 1))
            return self.out(x)

    return UNet()


def train(clean_dir: str, out_path: str, *, resume_from: str | None = None,
          epochs: int = 8, batch: int = 8, progress_cb=None, epoch_cb=None) -> dict:
    import torch
    from torch.utils.data import DataLoader, Dataset

    assets = load_wm_assets()
    files = _list_clean(clean_dir)
    if len(files) < 4:
        raise ValueError("Cần ít nhất vài ảnh sạch để train.")
    base_rng = random.Random(1234)

    class DS(Dataset):
        def __init__(self, paths, n):
            self.paths, self.n = paths, n

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            p = self.paths[i % len(self.paths)]
            img, mask = make_sample(p, assets, random.Random(i * 7 + base_rng.randint(0, 1 << 20)))
            x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            y = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
            return x, y

    # ~1 pass over the data per epoch (capped) so epochs finish fast on big sets
    # and each is checkpointed — better than one giant slow epoch.
    steps_per = max(2, min(len(files) * 3, 6000) // batch)
    dl = DataLoader(DS(files, steps_per * batch), batch_size=batch, shuffle=True, num_workers=0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = _build_unet().to(dev)
    # Continual / resume: warm-start from an existing checkpoint so training is
    # ADDITIVE across sessions instead of starting over. Because clean images
    # accumulate on disk, each resume trains on the FULL set (old + new) → the
    # model keeps improving without forgetting.
    prev_epochs = 0
    if resume_from and os.path.exists(resume_from):
        try:
            ck = torch.load(resume_from, map_location=dev)
            net.load_state_dict(ck["state"])
            prev_epochs = int(ck.get("trained_epochs", 0))
            logger.info("resumed from %s (%d prior epochs)", resume_from, prev_epochs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not resume from %s: %s", resume_from, exc)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    bce = torch.nn.BCEWithLogitsLoss()

    def dice(logits, y):
        p = torch.sigmoid(logits)
        return 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    net.train()
    hist, step, total = [], 0, epochs * len(dl)
    for ep in range(epochs):
        run = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = net(x)
            loss = bce(out, y) + dice(out, y)
            loss.backward()
            opt.step()
            run += float(loss.item())
            step += 1
            if progress_cb and step % 2 == 0:
                progress_cb(step / total, float(loss.item()))
        hist.append(run / len(dl))
        total_epochs = prev_epochs + ep + 1
        # CHECKPOINT AFTER EVERY EPOCH — a crash loses at most this one epoch, and
        # the model is immediately usable/integrated at its current level.
        tmp = out_path + ".tmp"
        torch.save({"state": net.state_dict(), "img_size": IMG_SIZE,
                    "trained_epochs": total_epochs}, tmp)
        os.replace(tmp, out_path)
        logger.info("epoch %d/%d loss=%.4f -> saved (%d total epochs)",
                    ep + 1, epochs, hist[-1], total_epochs)
        if epoch_cb:
            epoch_cb(total_epochs, prev_epochs + epochs, round(hist[-1], 4))
        if progress_cb:
            progress_cb((ep + 1) / epochs, hist[-1])

    return {"final_loss": round(hist[-1], 4) if hist else None, "epochs": epochs,
            "trained_epochs": prev_epochs + epochs, "resumed": prev_epochs > 0,
            "samples": len(files), "wm_assets": len(assets)}
