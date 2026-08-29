"""
Per-epoch named checkpoints + durable status.json for crash-safe resume.

Layout:
  assets/checkpoints/
    detector_ep26.pt
    removal_ep68.pt
    status.json          # live progress for UI + auto-resume
  assets/wm_detector.pt  # always = latest detector (symlink-like copy)
  assets/wm_remover.pt   # always = latest removal
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

logger = logging.getLogger("clearmark.checkpoints")

_ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
CKPT_DIR = os.path.join(_ASSETS, "checkpoints")
STATUS_PATH = os.path.join(_ASSETS, "status.json")
os.makedirs(CKPT_DIR, exist_ok=True)

_EP_RE = re.compile(r"^(detector|removal)_ep(\d+)\.pt$", re.I)


def status_path() -> str:
    return STATUS_PATH


def named_path(kind: str, epoch: int) -> str:
    kind = "removal" if kind == "removal" else "detector"
    return os.path.join(CKPT_DIR, f"{kind}_ep{int(epoch)}.pt")


def latest_named(kind: str) -> tuple[str | None, int]:
    """Return (path, epoch) of the highest numbered named checkpoint, or (None, 0)."""
    kind = "removal" if kind == "removal" else "detector"
    best_ep, best_path = 0, None
    try:
        for name in os.listdir(CKPT_DIR):
            m = _EP_RE.match(name)
            if not m or m.group(1).lower() != kind:
                continue
            ep = int(m.group(2))
            if ep >= best_ep:
                best_ep, best_path = ep, os.path.join(CKPT_DIR, name)
    except OSError:
        pass
    return best_path, best_ep


def write_status(payload: dict[str, Any]) -> None:
    data = dict(payload)
    data["updated"] = time.time()
    tmp = STATUS_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, STATUS_PATH)
    except OSError as exc:
        logger.warning("status.json write failed: %s", exc)


def read_status() -> dict[str, Any]:
    try:
        with open(STATUS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def save_epoch(
    kind: str,
    epoch: int,
    state: dict,
    *,
    latest_path: str,
    optimizer_state: dict | None = None,
    loss: float | None = None,
    target_epochs: int | None = None,
    extra: dict | None = None,
) -> str:
    """
    Atomically save:
      1) named checkpoint  checkpoints/{kind}_ep{N}.pt
      2) rolling latest    assets/wm_*.pt
    and update status.json progress %.
    """
    kind = "removal" if kind == "removal" else "detector"
    payload = {
        "state": state,
        "trained_epochs": int(epoch),
        "kind": kind,
        "loss": loss,
        **(extra or {}),
    }
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state

    named = named_path(kind, epoch)
    os.makedirs(os.path.dirname(named), exist_ok=True)
    tmp_n = named + ".tmp"
    import torch

    torch.save(payload, tmp_n)
    os.replace(tmp_n, named)

    # Rolling "latest" used by inference + Train tiếp
    os.makedirs(os.path.dirname(latest_path), exist_ok=True)
    tmp_l = latest_path + ".tmp"
    torch.save(payload, tmp_l)
    os.replace(tmp_l, latest_path)

    # Progress for Next.js: done/target within this run + cumulative epoch
    st = read_status()
    target = int(target_epochs or st.get("target_epochs") or epoch)
    base = int(st.get("base_epochs") or 0)
    run_done = max(0, epoch - base)
    run_total = max(1, target - base)
    pct = min(0.99, run_done / run_total) if target > base else 1.0
    write_status({
        **st,
        "status": "running",
        "kind": kind,
        "epoch": epoch,
        "target_epochs": target,
        "base_epochs": base,
        "progress": pct,
        "loss": loss,
        "latest_checkpoint": named,
        "message": f"{kind}: đã lưu vòng {epoch}/{target} (checkpoint an toàn).",
    })
    logger.info("checkpoint saved %s (epoch %d)", named, epoch)
    return named


def resolve_resume(kind: str, latest_path: str, fresh: bool = False) -> tuple[str | None, int]:
    """
    Pick the best weights to resume from:
      named checkpoint with highest epoch  >  rolling latest  >  None
    """
    if fresh:
        return None, 0
    named, ep = latest_named(kind)
    if named and ep > 0:
        return named, ep
    if os.path.exists(latest_path):
        try:
            import torch

            ck = torch.load(latest_path, map_location="cpu")
            return latest_path, int(ck.get("trained_epochs", 0))
        except Exception:  # noqa: BLE001
            return latest_path, 0
    return None, 0
