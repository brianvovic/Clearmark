"""
End-to-end REMOVAL model — learns watermarked → clean directly.

Unlike the detector (which only finds WHERE the watermark is, then LaMa fills the
hole), this network learns to RECONSTRUCT the clean pixels: it predicts a residual
that, added to the watermarked image, cancels the watermark and rebuilds what was
underneath. Trained on the same free synthetic pairs — synthesize() gives the
watermarked image AND we already have the clean source, so it's fully supervised.

Residual learning (clean = input + Δ) is easy to optimise and keeps untouched
areas near-identity. L1 loss, weighted higher inside the watermark so the model
spends its capacity where it matters.

Quality is DATA-BOUND: with few images it blurs; with thousands of clean stock
photos (see scrape_clean.py) + many epochs it sharpens. Weights → assets/
wm_remover.pt, used by services.wm_remover.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
from PIL import Image

from training.pipeline import IMG_SIZE, _list_clean, load_wm_assets, synthesize

logger = logging.getLogger("clearmark.removal")


def build_removal_net():
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        )

    class RemovalUNet(nn.Module):
        """Predicts a residual in [-1,1]; clean = clamp(input + residual)."""

        def __init__(self):
            super().__init__()
            self.d1, self.d2, self.d3, self.d4 = block(3, 32), block(32, 64), block(64, 128), block(128, 256)
            self.pool = nn.MaxPool2d(2)
            self.mid = block(256, 384)
            self.up4 = nn.ConvTranspose2d(384, 256, 2, 2); self.u4 = block(512, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, 2); self.u3 = block(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, 2); self.u2 = block(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, 2); self.u1 = block(64, 32)
            self.out = nn.Conv2d(32, 3, 1)
            self.act = nn.Tanh()

        def forward(self, x):
            import torch

            c1 = self.d1(x)
            c2 = self.d2(self.pool(c1))
            c3 = self.d3(self.pool(c2))
            c4 = self.d4(self.pool(c3))
            m = self.mid(self.pool(c4))
            y = self.u4(torch.cat([self.up4(m), c4], 1))
            y = self.u3(torch.cat([self.up3(y), c3], 1))
            y = self.u2(torch.cat([self.up2(y), c2], 1))
            y = self.u1(torch.cat([self.up1(y), c1], 1))
            return torch.clamp(x + self.act(self.out(y)), 0, 1)

    return RemovalUNet()


def train_removal(clean_dir: str, out_path: str, *, resume_from: str | None = None,
                  epochs: int = 10, batch: int = 6, progress_cb=None, epoch_cb=None) -> dict:
    import torch
    from torch.utils.data import DataLoader, Dataset

    assets = load_wm_assets()
    files = _list_clean(clean_dir)
    if len(files) < 4:
        raise ValueError("Cần ít nhất vài ảnh sạch để train.")
    base_rng = random.Random(4321)

    class DS(Dataset):
        def __init__(self, paths, n):
            self.paths, self.n = paths, n

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            p = self.paths[i % len(self.paths)]
            clean = np.array(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
            wm, mask = synthesize(clean, assets, random.Random(i * 11 + base_rng.randint(0, 1 << 20)))
            x = torch.from_numpy(wm).permute(2, 0, 1).float() / 255.0
            y = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
            w = torch.from_numpy((mask > 0).astype("float32")).unsqueeze(0) * 4 + 1  # weight in wm
            return x, y, w

    steps_per = max(2, min(len(files) * 3, 6000) // batch)
    dl = DataLoader(DS(files, steps_per * batch), batch_size=batch, shuffle=True, num_workers=0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_removal_net().to(dev)
    prev_epochs = 0
    if resume_from and os.path.exists(resume_from):
        try:
            ck = torch.load(resume_from, map_location=dev)
            net.load_state_dict(ck["state"])
            prev_epochs = int(ck.get("trained_epochs", 0))
            logger.info("removal resumed from %s (%d epochs)", resume_from, prev_epochs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("removal resume failed: %s", exc)
    opt = torch.optim.Adam(net.parameters(), 8e-4)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    net.train()
    hist, step, total = [], 0, epochs * len(dl)
    for ep in range(epochs):
        run = 0.0
        for x, y, w in dl:
            x, y, w = x.to(dev), y.to(dev), w.to(dev)
            opt.zero_grad()
            pred = net(x)
            loss = (torch.abs(pred - y) * w).mean()
            loss.backward()
            opt.step()
            run += float(loss.item())
            step += 1
            if progress_cb and step % 2 == 0:
                progress_cb(step / total, float(loss.item()))
        hist.append(run / len(dl))
        total_epochs = prev_epochs + ep + 1
        tmp = out_path + ".tmp"  # checkpoint every epoch (crash-safe)
        torch.save({"state": net.state_dict(), "img_size": IMG_SIZE, "trained_epochs": total_epochs}, tmp)
        os.replace(tmp, out_path)
        logger.info("removal epoch %d/%d L1=%.4f -> saved (%d total)", ep + 1, epochs, hist[-1], total_epochs)
        if epoch_cb:
            epoch_cb(total_epochs, prev_epochs + epochs, round(hist[-1], 4))
        if progress_cb:
            progress_cb((ep + 1) / epochs, hist[-1])

    return {"final_loss": round(hist[-1], 4) if hist else None, "epochs": epochs,
            "trained_epochs": prev_epochs + epochs, "resumed": prev_epochs > 0, "samples": len(files)}
