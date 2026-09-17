"""
Download additional diverse negative-class images using Picsum Photos.
Uses higher seed range (2001-2400) and specific Picsum portrait IDs
to ensure the negative class contains human/person photos.
All downloads use urllib (standard library) — no external dependencies.
"""

import os
import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

OUT_DIR = os.path.join("data_non_sweet", "not_somali_sweet")
NUM_THREADS = 12

# ---------------------------------------------------------------------------
# Picsum IDs that are portraits / contain people (verified from picsum.photos)
# ---------------------------------------------------------------------------
PORTRAIT_IDS = [
    10, 21, 22, 30, 47, 64, 65, 91, 100, 102,
    110, 119, 127, 129, 143, 151, 157, 163, 177, 185,
    195, 196, 200, 213, 221, 225, 229, 236, 241, 247,
    256, 262, 264, 268, 280, 292, 300, 305, 313, 316,
    317, 318, 325, 337, 338, 339, 342, 348, 349, 354,
    357, 366, 367, 375, 376, 379, 381, 382, 383, 389,
    390, 392, 397, 399, 400, 401, 402, 403, 404, 405,
    408, 409, 410, 411, 413, 414, 415, 416, 417, 418,
    420, 421, 424, 425, 426, 428, 429, 430, 431, 432,
    433, 436, 437, 440, 441, 442, 443, 444, 445, 446,
    450, 453, 454, 455, 456, 460, 461, 462, 463, 464,
    465, 466, 467, 468, 469, 470, 471, 472, 473, 474,
    475, 476, 477, 478, 479, 480, 481, 482, 483, 484,
    485, 486, 487, 488, 489, 490, 491, 492, 493, 494,
]

def download_picsum_by_id(pic_id, out_name):
    """Download a Picsum photo by its specific photo ID."""
    url = f"https://picsum.photos/id/{pic_id}/224/224"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.save(os.path.join(OUT_DIR, out_name), "JPEG")
        return True, pic_id
    except Exception as e:
        return False, f"ID {pic_id}: {e}"

def download_picsum_random(seed, out_name):
    """Download a random Picsum photo by seed."""
    url = f"https://picsum.photos/224/224?random={seed}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read()
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.save(os.path.join(OUT_DIR, out_name), "JPEG")
        return True, seed
    except Exception as e:
        return False, f"seed {seed}: {e}"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    existing = len([f for f in os.listdir(OUT_DIR) if f.endswith(".jpg")])
    print(f"Existing negative images: {existing}")

    success = 0
    failures = []

    # 1. Specific portrait IDs from Picsum
    print(f"\nStep 1: Downloading {len(PORTRAIT_IDS)} portrait images by ID...")
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as ex:
        futs = {ex.submit(download_picsum_by_id, pid, f"portrait_{pid:04d}.jpg"): pid
                for pid in PORTRAIT_IDS}
        for fut in as_completed(futs):
            ok, result = fut.result()
            if ok:
                success += 1
            else:
                failures.append(result)
    print(f"  Portrait IDs downloaded: {success}/{len(PORTRAIT_IDS)}")

    # 2. High-range random seeds (2001-2350) for diverse extra content
    EXTRA_SEEDS = list(range(2001, 2351))
    print(f"\nStep 2: Downloading {len(EXTRA_SEEDS)} extra diverse images (seeds 2001-2350)...")
    extra_ok = 0
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as ex:
        futs = {ex.submit(download_picsum_random, s, f"diverse_{s}.jpg"): s
                for s in EXTRA_SEEDS}
        for fut in as_completed(futs):
            ok, result = fut.result()
            if ok:
                extra_ok += 1
                if extra_ok % 100 == 0:
                    print(f"  {extra_ok}/{len(EXTRA_SEEDS)} extra images downloaded...")
            else:
                failures.append(result)
    print(f"  Extra diverse images downloaded: {extra_ok}/{len(EXTRA_SEEDS)}")

    total = len([f for f in os.listdir(OUT_DIR) if f.endswith(".jpg")])
    print(f"\nTotal negative class images: {total}")
    if failures:
        print(f"Failures: {len(failures)} — first 3: {failures[:3]}")

if __name__ == "__main__":
    main()
