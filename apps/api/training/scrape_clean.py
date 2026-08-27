"""
Fetch CLEAN (watermark-free) images to build the training set.

Sources:
  • "picsum" (default, NO API key): Lorem Picsum — free CC0 photos sourced from
    Unsplash. Perfect for bulk training data, no sign-up.
  • "pexels": official Pexels API (needs a free key: https://www.pexels.com/api/).
  • "unsplash": official Unsplash API (needs a free key:
    https://unsplash.com/developers).

Downloaded images go straight into the training clean-set. Use only for training
your own model on watermark-free stock photos.

CLI:  python -m training.scrape_clean --count 500
      python -m training.scrape_clean --count 1000 --source pexels --key <KEY>
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger("clearmark.scrape")

SIZE = 512


def _save(dirpath: str, idx: int, data: bytes) -> None:
    with open(os.path.join(dirpath, f"stock_{int(time.time())}_{idx}.jpg"), "wb") as f:
        f.write(data)


def _valid_image(data: bytes) -> bool:
    from io import BytesIO

    from PIL import Image

    try:
        Image.open(BytesIO(data)).verify()
        return len(data) > 3000
    except Exception:  # noqa: BLE001
        return False


def download(count: int, out_dir: str, *, source: str = "picsum",
             api_key: str | None = None, progress_cb=None) -> dict:
    import httpx

    os.makedirs(out_dir, exist_ok=True)
    got, tries = 0, 0
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        if source == "picsum":
            while got < count and tries < count * 3:
                tries += 1
                try:
                    r = client.get(f"https://picsum.photos/{SIZE}/{SIZE}",
                                   params={"random": tries})
                    if r.status_code == 200 and _valid_image(r.content):
                        _save(out_dir, tries, r.content)
                        got += 1
                        if progress_cb:
                            progress_cb(got / count, got)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("picsum fetch failed: %s", exc)

        elif source == "pexels":
            if not api_key:
                raise ValueError("Pexels cần API key.")
            page = 1
            while got < count and page < 200:
                r = client.get("https://api.pexels.com/v1/curated",
                               params={"per_page": 80, "page": page},
                               headers={"Authorization": api_key})
                if r.status_code != 200:
                    raise ValueError(f"Pexels API lỗi {r.status_code}: {r.text[:120]}")
                for photo in r.json().get("photos", []):
                    if got >= count:
                        break
                    url = photo.get("src", {}).get("large") or photo.get("src", {}).get("medium")
                    try:
                        img = client.get(url)
                        if img.status_code == 200 and _valid_image(img.content):
                            _save(out_dir, got, img.content)
                            got += 1
                            if progress_cb:
                                progress_cb(got / count, got)
                    except Exception:  # noqa: BLE001
                        pass
                page += 1

        elif source == "unsplash":
            if not api_key:
                raise ValueError("Unsplash cần Access Key.")
            while got < count:
                r = client.get("https://api.unsplash.com/photos/random",
                               params={"count": 30, "client_id": api_key})
                if r.status_code != 200:
                    raise ValueError(f"Unsplash API lỗi {r.status_code}: {r.text[:120]}")
                for photo in r.json():
                    if got >= count:
                        break
                    url = photo.get("urls", {}).get("regular")
                    try:
                        img = client.get(url)
                        if img.status_code == 200 and _valid_image(img.content):
                            _save(out_dir, got, img.content)
                            got += 1
                            if progress_cb:
                                progress_cb(got / count, got)
                    except Exception:  # noqa: BLE001
                        pass
        else:
            raise ValueError(f"Nguồn không hỗ trợ: {source}")

    return {"downloaded": got, "source": source}


if __name__ == "__main__":
    import argparse

    from services.train_jobs import CLEAN_DIR

    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--source", default="picsum", choices=["picsum", "pexels", "unsplash"])
    p.add_argument("--key", default=None)
    p.add_argument("--out", default=CLEAN_DIR)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    info = download(a.count, a.out, source=a.source, api_key=a.key,
                    progress_cb=lambda p, n: print(f"\r{n}/{a.count}", end=""))
    print("\n", info)
