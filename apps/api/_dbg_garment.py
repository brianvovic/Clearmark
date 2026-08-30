"""Unit-level check of the garment safety net (no models, runs in a second)."""
from __future__ import annotations

import logging

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from services import ink  # noqa: E402
from tools_smoke_removal import _body_with_bikini, _load, _photos, _stamp_text  # noqa: E402

def features(before: np.ndarray, after: np.ndarray, tag: str) -> None:
    """Same measurements the net makes, printed so thresholds can be chosen."""
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16)).max(axis=2)
    changed = (diff > 6).astype(np.uint8) * 255
    skin = ink.skin_mask(before) > 0
    clusters = cv2.morphologyEx(
        changed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    n, labels, st, _ = cv2.connectedComponentsWithStats((clusters > 0).astype(np.uint8), 8)
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))
    for i in range(1, n):
        if int(st[i, cv2.CC_STAT_AREA]) < 400:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        sel = comp > 0
        local = cv2.bitwise_and(changed, comp)
        tc = float(cv2.distanceTransform(comp // 255, cv2.DIST_L2, 3).max() * 2)
        tl = float(cv2.distanceTransform(local // 255, cv2.DIST_L2, 3).max() * 2)
        ring = (cv2.dilate(comp, ring_k, 1) > 0) & (~sel)
        print(f"  min_thick={ink.garment_thickness(skin.astype(np.uint8) * 255):5.1f}")
        print(f"  {tag:22s} area={st[i, cv2.CC_STAT_AREA]:7d} thick_cluster={tc:6.1f} "
              f"thick_changed={tl:5.1f} fill={float((local[sel] > 0).mean()):4.2f} "
              f"ring_skin={skin[ring].mean() if ring.sum() else 0:4.2f}")


wm, clean, suit, gt = _body_with_bikini()

# 1) A pass that repainted the swimsuit as skin must be rolled back.
wiped = cv2.inpaint(wm, cv2.dilate(suit, np.ones((5, 5), np.uint8), 1), 12, cv2.INPAINT_TELEA)
features(wm, wiped, "swimsuit-wiped")
kept = ink.revert_worn_garments(wm, wiped)
back = float(np.abs(kept[suit > 0].astype(int) - wm[suit > 0].astype(int)).mean())
print(f"swimsuit repaint reverted: residual_change={back:5.2f} (want ~0)")

# 2) The same pass removing the pink text must NOT be rolled back.
erased = wm.copy()
erased[gt > 0] = clean[gt > 0]
kept2 = ink.revert_worn_garments(wm, erased)
still = float(np.abs(kept2[gt > 0].astype(int) - erased[gt > 0].astype(int)).mean())
print(f"text removal preserved:    rollback={still:5.2f} (want ~0)")

# 3) Same on real photos: removing 'gaigu' must survive the net.
for p in _photos(3):
    c = _load(p)
    w2, g2 = _stamp_text(c)
    e2 = w2.copy()
    e2[g2 > 0] = c[g2 > 0]
    features(w2, e2, "text/" + p.split(chr(92))[-1][:14])
    k2 = ink.revert_worn_garments(w2, e2)
    roll = float(np.abs(k2[g2 > 0].astype(int) - e2[g2 > 0].astype(int)).mean())
    print(f"  {p.split(chr(92))[-1]:28s} rollback={roll:5.2f} (want ~0)")
