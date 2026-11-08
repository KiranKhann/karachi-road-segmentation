"""Re-download the Karachi road photographs listed in data/karachi_raw/MANIFEST.csv from Wikimedia Commons
and downscale them to a 1600 px long edge (the resolution used by notebook 02).

Usage:  python scripts/fetch_karachi_images.py
"""
import csv, io, time, urllib.request
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "karachi_raw"
UA = {"User-Agent": "KarachiSegmentationProject/1.0 (educational; https://github.com/)"}
MAX_SIDE = 1600

rows = list(csv.DictReader(open(RAW / "MANIFEST.csv", encoding="utf-8")))
for r in rows:
    dest = RAW / r["file"]
    if dest.exists() and dest.stat().st_size > 0:
        print("exists ", r["file"]); continue
    for attempt in range(3):
        try:
            data = urllib.request.urlopen(urllib.request.Request(r["downloaded_url"], headers=UA), timeout=300).read()
            im = Image.open(io.BytesIO(data)).convert("RGB")
            s = MAX_SIDE / max(im.size)
            if s < 1: im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            im.save(dest, quality=92)
            print(f"saved   {r['file']}  {im.size}  [{r['license']}] {r['commons_title']}")
            break
        except Exception as e:
            print("retry  ", r["file"], e); time.sleep(3)
print(f"{sum((RAW / r['file']).exists() for r in rows)}/{len(rows)} images present in {RAW}")
