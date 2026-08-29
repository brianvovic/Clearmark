"""
End-to-end REMOVAL model — watermarked → clean.

Loss (anti-blur):
  L_total = λ_l1 * weighted_L1  +  λ_lpips * LPIPS  +  λ_gan * adversarial

Pure L1 makes the net "lazy" (average/blur to minimise pixel error). LPIPS + a
small PatchGAN force texture and sharpness inside the watermark region.

Checkpoints: every epoch → checkpoints/removal_epN.pt + assets/wm_remover.pt
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
from PIL import Image

from training.checkpoints import resolve_resume, save_epoch
from training.pipeline import IMG_SIZE, _list_clean, load_wm_assets, synthesize

logger = logging.getLogger("clearmark.removal")

# Loss weights — LPIPS is the main anti-blur signal
_L1_W = 0.6
_LPIPS_W = 0.8
_GAN_W = 0.05


def build_removal_net():
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        )

    class RemovalUNet(nn.Module):
        """Predicts residual in [-1,1]; clean = clamp(input + residual)."""

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


def build_discriminator():
    """Lightweight PatchGAN — classifies 70×70 patches as real/fake."""
    import torch.nn as nn

    def cblock(i, o, stride=2):
        return nn.Sequential(
            nn.Conv2d(i, o, 4, stride=stride, padding=1),
            nn.BatchNorm2d(o),
            nn.LeakyReLU(0.2, inplace=True),
        )

    class PatchDisc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                cblock(32, 64),
                cblock(64, 128),
                nn.Conv2d(128, 1, 4, 1, 1),
            )

        def forward(self, x):
            return self.net(x)

    return PatchDisc()


class _LpipsLoss:
    """Lazy LPIPS wrapper (AlexNet). Falls back to None if package missing."""

    def __init__(self, device: str):
        self.ok = False
        self.net = None
        try:
            import lpips

            self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
            for p in self.net.parameters():
                p.requires_grad_(False)
            self.ok = True
            logger.info("LPIPS perceptual loss enabled")
        except Exception as exc:  # noqa: BLE001
            logger.warning("LPIPS unavailable (%s) — training with L1+GAN only", exc)

    def __call__(self, pred, target):
        import torch

        if not self.ok:
            return pred.new_zeros(())
        # LPIPS expects [-1, 1]
        a = pred * 2 - 1
        b = target * 2 - 1
        return self.net(a, b).mean()


def train_removal(clean_dir: str, out_path: str, *, resume_from: str | None = None,
                  epochs: int = 10, batch: int = 6, progress_cb=None, epoch_cb=None,
                  target_epochs: int | None = None, base_epochs: int | None = None) -> dict:
    import torch
    import torch.nn.functional as F
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
            wm, mask = synthesize(clean, assets, random.Random(i * 11 + base_rng.randint(0, 1 << 20)),
                                  augment=False)
            x = torch.from_numpy(wm).permute(2, 0, 1).float() / 255.0
            y = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
            w = torch.from_numpy((mask > 0).astype("float32")).unsqueeze(0) * 6 + 1
            return x, y, w

    steps_per = max(2, min(len(files) * 3, 6000) // batch)
    dl = DataLoader(DS(files, steps_per * batch), batch_size=batch, shuffle=True, num_workers=0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_removal_net().to(dev)
    disc = build_discriminator().to(dev)
    lpips_fn = _LpipsLoss(dev)

    # Prefer explicit resume_from, else highest named checkpoint
    if not resume_from:
        resume_from, _ = resolve_resume("removal", out_path, fresh=False)

    prev_epochs = 0
    if resume_from and os.path.exists(resume_from):
        try:
            ck = torch.load(resume_from, map_location=dev)
            net.load_state_dict(ck["state"])
            prev_epochs = int(ck.get("trained_epochs", 0))
            if ck.get("optimizer"):
                pass  # loaded below after opt create
            logger.info("removal resumed from %s (%d epochs)", resume_from, prev_epochs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("removal resume failed: %s", exc)

    opt = torch.optim.Adam(net.parameters(), 5e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), 2e-4, betas=(0.5, 0.999))
    if resume_from and os.path.exists(resume_from):
        try:
            ck = torch.load(resume_from, map_location=dev)
            if ck.get("optimizer"):
                opt.load_state_dict(ck["optimizer"])
        except Exception:  # noqa: BLE001
            pass

    from training.checkpoints import read_status, write_status

    run_target = (target_epochs if target_epochs is not None else prev_epochs + epochs)
    base = base_epochs if base_epochs is not None else prev_epochs
    write_status({
        "status": "running", "kind": "removal", "epoch": prev_epochs,
        "target_epochs": run_target, "base_epochs": base, "progress": 0.0,
        "message": f"Removal: tiếp tục từ vòng {prev_epochs} → {run_target} (L1+LPIPS+GAN)",
    })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    net.train()
    disc.train()
    hist, step, total = [], 0, epochs * len(dl)

    for ep in range(epochs):
        run = 0.0
        for x, y, w in dl:
            x, y, w = x.to(dev), y.to(dev), w.to(dev)

            # ---- Generator ----
            opt.zero_grad()
            pred = net(x)
            l1 = (torch.abs(pred - y) * w).mean()
            # LPIPS on full image — forces texture / sharpness vs lazy blur
            perc = lpips_fn(pred, y)
            fake_logits = disc(pred)
            gan_g = F.binary_cross_entropy_with_logits(
                fake_logits, torch.ones_like(fake_logits)
            )
            loss_g = _L1_W * l1 + _LPIPS_W * perc + _GAN_W * gan_g
            loss_g.backward()
            opt.step()

            # ---- Discriminator ----
            opt_d.zero_grad()
            real_logits = disc(y.detach())
            fake_logits_d = disc(pred.detach())
            loss_d = 0.5 * (
                F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                + F.binary_cross_entropy_with_logits(fake_logits_d, torch.zeros_like(fake_logits_d))
            )
            loss_d.backward()
            opt_d.step()

            run += float(loss_g.item())
            step += 1
            if progress_cb and step % 2 == 0:
                frac = (ep + (step % max(1, len(dl))) / max(1, len(dl))) / max(1, epochs)
                run_done = (prev_epochs - base) + frac * epochs
                run_total = max(1, run_target - base)
                progress_cb(min(0.99, run_done / run_total), float(loss_g.item()))

        hist.append(run / len(dl))
        total_epochs = prev_epochs + ep + 1
        save_epoch(
            "removal",
            total_epochs,
            net.state_dict(),
            latest_path=out_path,
            optimizer_state=opt.state_dict(),
            loss=round(hist[-1], 4),
            target_epochs=run_target,
            extra={"img_size": IMG_SIZE, "base_epochs": base},
        )
        st = read_status()
        st["base_epochs"] = base
        st["target_epochs"] = run_target
        write_status(st)

        logger.info(
            "removal epoch %d/%d loss=%.4f (L1+LPIPS+GAN) -> ckpt ep%d",
            ep + 1, epochs, hist[-1], total_epochs,
        )
        if epoch_cb:
            epoch_cb(total_epochs, run_target, round(hist[-1], 4))
        if progress_cb:
            run_done = total_epochs - base
            run_total = max(1, run_target - base)
            progress_cb(min(0.99, run_done / run_total), hist[-1])

    write_status({
        "status": "done", "kind": "removal", "epoch": prev_epochs + epochs,
        "target_epochs": run_target, "base_epochs": base, "progress": 1.0,
        "loss": round(hist[-1], 4) if hist else None,
        "message": f"Removal xong — tổng {prev_epochs + epochs} vòng (L1+LPIPS+GAN).",
    })
    return {
        "final_loss": round(hist[-1], 4) if hist else None,
        "epochs": epochs,
        "trained_epochs": prev_epochs + epochs,
        "resumed": prev_epochs > 0,
        "samples": len(files),
        "losses": "L1+LPIPS+GAN",
    }
