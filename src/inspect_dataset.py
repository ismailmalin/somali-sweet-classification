"""
Data inspection: finds corrupt, non-RGB, and duplicate images.
Run from project root: python src/inspect_data.py
"""

import os
from pathlib import Path
from PIL import Image
import hashlib
from collections import defaultdict

DATA_DIR = Path("data")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

def hash_file(filepath):
    """Return MD5 hash of a file's contents."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    corrupt = []
    non_rgb = []
    duplicates = defaultdict(list)  # hash -> [path, ...]
    total = 0

    for split in DATA_DIR.iterdir():
        if not split.is_dir():
            continue
        for ext in VALID_EXTENSIONS:
            for img_path in split.rglob(f"*{ext}"):
                total += 1
                # Check if file can be opened as image
                try:
                    with Image.open(img_path) as img:
                        img.verify()  # verifies file integrity
                    # Re-open for mode check (verify can leave file in broken state)
                    with Image.open(img_path) as img:
                        if img.mode not in ("RGB", "L", "RGBA"):
                            non_rgb.append(str(img_path))
                except (IOError, OSError) as e:
                    corrupt.append(str(img_path))
                    continue

                # Hash for duplicates
                file_hash = hash_file(img_path)
                duplicates[file_hash].append(str(img_path))

    print(f"Total images scanned: {total}")
    print(f"Corrupt/unreadable: {len(corrupt)}")
    if corrupt:
        print("  Corrupt files:")
        for f in corrupt[:10]:  # show first 10
            print(f"    {f}")

    print(f"Non‑RGB images (may need conversion): {len(non_rgb)}")
    if non_rgb:
        print("  Non-RGB files:")
        for f in non_rgb[:10]:
            print(f"    {f}")

    duplicate_groups = {h: paths for h, paths in duplicates.items() if len(paths) > 1}
    num_dups = sum(len(paths) - 1 for paths in duplicate_groups.values())
    print(f"Duplicate files (excess copies): {num_dups}")
    if duplicate_groups:
        print("  Duplicate groups (first 5):")
        for h, paths in list(duplicate_groups.items())[:5]:
            print(f"    hash {h}: {len(paths)} copies")
            for p in paths:
                print(f"      {p}")

    if not corrupt and not non_rgb and num_dups == 0:
        print("\n✅ Dataset looks clean. Ready for training.")
    else:
        print("\n❌ Issues found. Fix them before training (remove corrupt, delete duplicates, convert non-RGB).")

if __name__ == "__main__":
    main()