"""Smoke test: real photos + synthetic watermarks through engine.erase / erase_auto.

Checks two failure modes at once:
  removal   — does the watermark actually go away? (delta vs clean inside mask)
  integrity — with a deliberately WRONG fat mask, is the photo still intact?
"""

from __future__ import annotations

import glob
import logging
import sys

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from services import engine  # noqa: E402


def _load(path: str, size: int = 512) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img)


def _photos(n: int = 3) -> list[str]:
    """Pick real photographs — the clean pool also holds manga line art, whose
    texture no fill can reconstruct, which makes pixel metrics meaningless."""
    import random

    pool = sorted(glob.glob("training/_data/clean/*"))
    random.Random(7).shuffle(pool)
    out: list[str] = []
    for p in pool:
        try:
            a = _load(p, 256).astype(np.float32)
        except Exception:  # noqa: BLE001
            continue
        sat = float(np.abs(a.max(axis=2) - a.min(axis=2)).mean())
        if sat > 28:
            out.append(p)
        if len(out) >= n:
            break
    return out


INK = np.array([255.0, 60.0, 160.0], np.float32)


def _stamp_text(clean: np.ndarray, alpha: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Pink 'gaigu' text like the real watermark."""
    gt = np.zeros(clean.shape[:2], np.uint8)
    cv2.putText(gt, "gaigu", (60, 300), cv2.FONT_HERSHEY_DUPLEX, 3.0, 255, 6, cv2.LINE_AA)
    wm = clean.astype(np.float32).copy()
    sel = gt > 0
    wm[sel] = clean[sel] * (1 - alpha) + INK * alpha
    return np.clip(wm, 0, 255).astype(np.uint8), gt


def _ink_left(img: np.ndarray, clean: np.ndarray, sel: np.ndarray) -> float:
    """How much of the watermark's own colour is still visible (0 = gone).

    Projecting onto the ink direction ignores texture the fill could not
    reconstruct and measures only what a viewer reads as 'the watermark'.
    """
    u = INK[None, :] - clean[sel].astype(np.float32)
    n = np.linalg.norm(u, axis=1, keepdims=True) + 1e-6
    u = u / n
    d = img[sel].astype(np.float32) - clean[sel].astype(np.float32)
    return float(np.maximum((d * u).sum(axis=1), 0).mean())


def _detail(x: np.ndarray) -> float:
    g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def main() -> int:
    files = _photos(3)
    ok = True

    for path in files:
        clean = _load(path)
        wm, gt = _stamp_text(clean)
        d_clean, d_wm = _detail(clean), _detail(wm)
        dil = cv2.dilate(gt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 1)

        # A correct mask stands in for a user brush, which the app trusts.
        out = np.asarray(
            engine.erase(Image.fromarray(wm), Image.fromarray(dil, mode="L"),
                         mode="smart", remove_text=True,
                         trusted=Image.fromarray(dil, mode="L")).convert("RGB")
        )
        sel = gt > 0
        before = _ink_left(wm, clean, sel)
        after = _ink_left(out, clean, sel)
        outside = float(np.abs(out[~sel].astype(int) - wm[~sel].astype(int)).mean())
        gain = 100.0 * (1.0 - after / max(before, 1e-3))
        keep_wm = 100.0 * _detail(out) / max(d_clean, 1e-3)
        print(f"[removal] {path.split(chr(92))[-1]:26s} "
              f"ink_before={before:6.1f} ink_after={after:6.1f} removed={gain:5.1f}% "
              f"outside_change={outside:.2f} detail={keep_wm:5.1f}%")
        if gain < 80:
            ok = False
            print("   !! watermark still visible")
        if outside > 1.0:
            ok = False
            print("   !! pixels outside the mask were modified")
        if not 70 <= keep_wm <= 135:
            ok = False
            print("   !! image detail was damaged or amplified")

        # End to end: detection included — this is what the user actually clicks
        try:
            auto = np.asarray(
                engine.erase_auto(Image.fromarray(wm), True, mode="smart").convert("RGB")
            )
            a_after = _ink_left(auto, clean, sel)
            a_gain = 100.0 * (1.0 - a_after / max(before, 1e-3))
            a_detail = 100.0 * _detail(auto) / max(d_clean, 1e-3)
            print(f"[auto]    detect+erase: removed={a_gain:5.1f}% detail={a_detail:5.1f}%")
            if a_gain < 60:
                ok = False
                print("   !! auto pipeline left the watermark")
            if not 70 <= a_detail <= 135:
                ok = False
                print("   !! auto pipeline damaged the photo")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[auto]    FAILED: {exc}")

        # Integrity: a WRONG mask covering the middle third of the frame
        bad = np.zeros(clean.shape[:2], np.uint8)
        bad[120:400, 100:420] = 255
        wrong = np.asarray(
            engine.erase(Image.fromarray(clean), Image.fromarray(bad, mode="L"),
                         mode="pro").convert("RGB")
        )
        b = bad > 0
        drift = float(np.abs(wrong[b].astype(int) - clean[b].astype(int)).mean())
        d0, d1 = _detail(clean[120:400, 100:420]), _detail(wrong[120:400, 100:420])
        keep = 100.0 * d1 / max(d0, 1e-3)
        print(f"[integrity] fat wrong mask: drift={drift:5.1f} detail_kept={keep:5.1f}%")
        if drift > 30 or keep < 55:
            ok = False
            print("   !! fat mask destroyed the photo")

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
