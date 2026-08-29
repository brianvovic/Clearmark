"""
High-precision watermark mask.

Lesson from user samples: recall-first CRAFT+residual paints body parts
(nipples, lips, fabric folds) → inpaint smears skin and fades color.

New policy — precision first:
1) EasyOCR readtext (must recognize Latin letters) on contrast-boosted image
2) Red/colored stamp logos (small, compact)
3) Optional thin residual strokes ONLY inside OCR boxes
Never spray residual across the whole frame.
"""

from __future__ import annotations

import logging
import re
import threading

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("clearmark.mask")

_ocr = None
_ocr_lock = threading.Lock()
_WORD = re.compile(r"[A-Za-zÀ-ỹĂăÂâÊêÔôƠơƯưĐđ]{2,}")


def _get_ocr():
    global _ocr
    if _ocr is not None:
        return _ocr
    with _ocr_lock:
        if _ocr is not None:
            return _ocr
        try:
            import easyocr

            gpu = False
            try:
                import torch

                gpu = torch.cuda.is_available()
            except ImportError:
                pass
            logger.info("Loading EasyOCR (gpu=%s)...", gpu)
            _ocr = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("EasyOCR unavailable: %s", exc)
            _ocr = False
        return _ocr


def _boost(rgb: np.ndarray) -> list[np.ndarray]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(4.5, (8, 8)).apply(l)
    boosted = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), 14.0)
    amp = cv2.normalize(cv2.absdiff(gray, bg), None, 0, 255, cv2.NORM_MINMAX)
    amp = cv2.cvtColor(amp, cv2.COLOR_GRAY2RGB)
    return [boosted, amp, rgb]


def _ocr_word_boxes(rgb: np.ndarray) -> list[tuple[np.ndarray, str]]:
    reader = _get_ocr()
    if not reader:
        return []
    h, w = rgb.shape[:2]
    found: list[tuple[np.ndarray, str]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for src in _boost(rgb):
        try:
            results = reader.readtext(
                src,
                detail=1,
                paragraph=False,
                low_text=0.25,
                text_threshold=0.5,
                link_threshold=0.3,
                canvas_size=1600,
                mag_ratio=1.2,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR read failed: %s", exc)
            continue
        for item in results:
            if len(item) < 3:
                continue
            bbox, text, conf = item[0], str(item[1]), float(item[2])
            if conf < 0.15:
                continue
            if not _WORD.search(text):
                continue
            # Skip ultra-long OCR (paragraphs / false)
            if len(text) > 24:
                continue
            pts = np.array(bbox, dtype=np.int32)
            x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
            x1, y1 = int(pts[:, 0].max()), int(pts[:, 1].max())
            bw, bh = max(1, x1 - x0), max(1, y1 - y0)
            area = bw * bh
            if area > 0.12 * h * w:
                continue
            # Prefer word-like geometry (not a round blob)
            aspect = max(bw, bh) / max(1.0, min(bw, bh))
            if aspect < 1.4 and area > 0.01 * h * w:
                continue
            key = (x0 // 8, y0 // 8, x1 // 8, y1 // 8)
            if key in seen:
                continue
            seen.add(key)
            found.append((pts, text))
            logger.info("OCR hit '%s' conf=%.2f", text, conf)
    return found


def _strokes_in_box(rgb: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Thin stroke mask inside one OCR box (black/top-hat)."""
    h, w = rgb.shape[:2]
    box = np.zeros((h, w), dtype=np.uint8)
    x0 = max(0, int(pts[:, 0].min()) - 2)
    y0 = max(0, int(pts[:, 1].min()) - 2)
    x1 = min(w - 1, int(pts[:, 0].max()) + 2)
    y1 = min(h - 1, int(pts[:, 1].max()) + 2)
    cv2.fillPoly(box, [pts.reshape(-1, 1, 2)], 255)
    cv2.rectangle(box, (x0, y0), (x1, y1), 255, -1)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    enh = cv2.createCLAHE(3.0, (8, 8)).apply(gray)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, (x1 - x0) // 2), 3))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(7, (y1 - y0) // 2)))
    bh = cv2.max(
        cv2.morphologyEx(enh, cv2.MORPH_BLACKHAT, kh),
        cv2.morphologyEx(enh, cv2.MORPH_BLACKHAT, kv),
    )
    th = cv2.max(
        cv2.morphologyEx(enh, cv2.MORPH_TOPHAT, kh),
        cv2.morphologyEx(enh, cv2.MORPH_TOPHAT, kv),
    )
    thr = max(5, int(max(bh.mean(), th.mean()) + 0.8 * max(bh.std(), th.std())))
    strokes = ((bh >= thr) | (th >= thr)).astype(np.uint8) * 255
    strokes = cv2.bitwise_and(strokes, box)
    # If stroke extract failed, fall back to the box itself (still limited)
    if float((strokes > 0).sum()) < 0.02 * max(1, (box > 0).sum()):
        return box
    strokes = cv2.dilate(strokes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
    return cv2.bitwise_and(strokes, box)


def _colored_logo_mask(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Looser red (hotel stamps are often washed out)
    red1 = cv2.inRange(hsv, (0, 35, 50), (18, 255, 255))
    red2 = cv2.inRange(hsv, (155, 35, 50), (180, 255, 255))
    cand = cv2.bitwise_or(red1, red2)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    out = np.zeros((h, w), dtype=np.uint8)
    img_area = h * w
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 25 or area > 0.025 * img_area:
            continue
        if max(bw, bh) > 0.28 * max(h, w):
            continue
        fill = area / max(1, bw * bh)
        if fill < 0.12:
            continue
        out[labels == i] = 255
    if out.max():
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
        logger.info("logo coverage=%.3f%%", 100.0 * float((out > 0).mean()))
    return out


_face_cascades = None


def _cascade_ctor():
    """cv2.CascadeClassifier moved to cv2.objdetect in OpenCV 5; support both."""
    if hasattr(cv2, "CascadeClassifier"):
        return cv2.CascadeClassifier
    objd = getattr(cv2, "objdetect", None)
    return getattr(objd, "CascadeClassifier", None) if objd else None


def _get_face_cascades():
    """Frontal + profile Haar cascades shipped with OpenCV (no download)."""
    global _face_cascades
    if _face_cascades is None:
        ctor = _cascade_ctor()
        cs = []
        base = getattr(getattr(cv2, "data", None), "haarcascades", None)
        if ctor and base:
            for n in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
                try:
                    c = ctor(base + n)
                    if not c.empty():
                        cs.append(c)
                except Exception:  # noqa: BLE001
                    pass
        if not cs:
            logger.info("Haar cascades unavailable (cv2 %s) — skin-tone protection only",
                        cv2.__version__)
        _face_cascades = cs
    return _face_cascades


def protect_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Build a KEEP-OUT mask (uint8 {0,255}) of regions auto-removal must never touch:
    detected faces (expanded to cover the whole head) plus strong skin tone.

    This is the safety net for the #1 complaint — auto detection grabbing a face
    (reddish skin/lips looked like a red logo) and LaMa blurring it. Any auto mask
    is AND-NOT'd with this before inpainting. Manual brushing bypasses it, so the
    user can still paint over skin deliberately.
    """
    h, w = rgb.shape[:2]
    keep = np.zeros((h, w), np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    for c in _get_face_cascades():
        try:
            faces = c.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                       minSize=(max(24, w // 25), max(24, h // 25)))
        except Exception:  # noqa: BLE001
            continue
        for (x, y, fw, fh) in faces:
            # expand the box to hair/chin/neck so we never nibble the face edge
            ex, ey = int(fw * 0.45), int(fh * 0.6)
            cv2.rectangle(keep, (max(0, x - ex), max(0, y - ey)),
                          (min(w, x + fw + ex), min(h, y + fh + ey)), 255, -1)
    # Strong skin tone (YCrCb) — protects hands/body even without a face hit.
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 138, 77), (255, 173, 127))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return cv2.bitwise_or(keep, skin)


def _apply_face_protection(mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Zero mask only on face/head boxes — keep strokes on body skin for peel."""
    h, w = rgb.shape[:2]
    face = np.zeros((h, w), np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    for c in _get_face_cascades():
        try:
            faces = c.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=5,
                minSize=(max(24, w // 25), max(24, h // 25)),
            )
        except Exception:  # noqa: BLE001
            continue
        for (x, y, fw, fh) in faces:
            ex, ey = int(fw * 0.45), int(fh * 0.6)
            cv2.rectangle(
                face,
                (max(0, x - ex), max(0, y - ey)),
                (min(w, x + fw + ex), min(h, y + fh + ey)),
                255, -1,
            )
    if face.max() == 0:
        return mask
    out = mask.copy()
    out[face > 0] = 0
    removed = int((mask > 0).sum() - (out > 0).sum())
    if removed > 0:
        logger.info("face protection removed %d masked px", removed)
    return out


def _apply_protection(mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Legacy full face+skin strip — prefer ``_apply_face_protection`` for OCR/text."""
    keep = protect_mask(rgb)
    protected = cv2.bitwise_and(mask, cv2.bitwise_not(keep))
    removed = int((mask > 0).sum() - (protected > 0).sum())
    if removed > 0:
        logger.info("protection removed %d masked px over faces/skin", removed)
    return protected


def neon_watermark_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Detect bright, saturated NON-SKIN coloured strokes — the neon/semi-transparent
    logo+text watermark type (magenta/pink/cyan/blue/green outlines) that sits over
    skin or any background. Returns uint8 {0,255}.

    Skin is low/medium saturation with an orange-red hue; neon watermark strokes
    are high saturation AND high "colourfulness" (channel spread) AND their hue is
    cyan/blue/green/magenta — well away from skin. So we can carve out exactly the
    watermark lines without masking the skin they lie on.
    """
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    colorful = (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)).astype(np.int16)

    # Per-hue rules. Skin hue in OpenCV is ~0-33 (orange-red); NONE of the bands
    # below occur in skin, so cyan/blue can be caught even when very pale, while
    # magenta/pink (closer to skin-red) needs more saturation to stay safe.
    green = (H >= 40) & (H < 85) & (S >= 50) & (colorful >= 35)
    cyan = (H >= 85) & (H < 105) & (S >= 18) & (V >= 85)          # pale bright cyan outlines
    blue = (H >= 105) & (H < 135) & (S >= 25) & (V >= 85)
    magenta = (H >= 135) & (H <= 168) & (S >= 35) & (colorful >= 30)
    core = (colorful >= 70) & (H >= 40) & (H <= 168)               # any very vivid neon
    raw = (green | cyan | blue | magenta | core).astype(np.uint8) * 255

    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 175, 130))
    on_person = float((skin > 0).mean()) > 0.08
    if on_person:
        # Watermark sits ON skin. Drop bluish background (tiles/walls) that also
        # trips the cyan/blue rule by gating to a grown skin region.
        near_skin = cv2.dilate(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)), 1)
        raw = cv2.bitwise_and(raw, near_skin)
    if raw.max() == 0:
        return raw

    # Build the watermark BAND envelope from the coloured strokes, then inside it
    # mask everything that is NOT skin — this captures the bright near-white NEON
    # CORE (which has no colour, so the colour rules miss it and it survives as a
    # ghost outline after inpaint) as well as the coloured glow.
    band = cv2.dilate(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), 1)
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    non_skin = cv2.bitwise_not(skin) if on_person else np.full_like(raw, 255)
    core = cv2.bitwise_and(band, non_skin)
    m = cv2.bitwise_or(raw, core)

    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 60 or bw * bh > 0.30 * h * w:
            continue
        out[labels == i] = 255
    if out.max():
        # grow a little to swallow the soft glow halo just outside the strokes
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 1)
        logger.info("neon watermark coverage=%.3f%%", 100.0 * float((out > 0).mean()))
    return out


def build_auto_mask(image: Image.Image, remove_text: bool = True) -> Image.Image:
    """
    Auto watermark mask.

    ``remove_text``: when False, skip OCR text detection entirely and only mask
    coloured logo/stamp watermarks. This is the dewatermark.ai-style opt-in:
    real printed text on shirts, packaging or signs is never touched unless the
    user explicitly asks to remove text.
    """
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    boxes: list[tuple[np.ndarray, str]] = []
    if remove_text:
        boxes = _ocr_word_boxes(rgb)
        for pts, text in boxes:
            strokes = _strokes_in_box(rgb, pts)
            mask = cv2.bitwise_or(mask, strokes)

    logo = _colored_logo_mask(rgb)
    mask = cv2.bitwise_or(mask, logo)

    # Face only — skin/body text must stay so erase can peel it.
    # (Full skin wipe here was why "gaigu" on body never reached removal.)
    mask = _apply_face_protection(mask, rgb)

    # Hard cap — prevents whole-image smear
    cov = float((mask > 0).mean())
    if cov > 0.12:
        # keep largest few components only
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        parts = sorted(
            ((int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, num)),
            reverse=True,
        )
        out = np.zeros_like(mask)
        budget = int(0.12 * h * w)
        used = 0
        for area, i in parts:
            if used + area > budget:
                continue
            out[labels == i] = 255
            used += area
        mask = out
        cov = float((mask > 0).mean())

    logger.info("auto mask coverage=%.3f%% words=%d", 100.0 * cov, len(boxes))
    return Image.fromarray(mask, mode="L")
