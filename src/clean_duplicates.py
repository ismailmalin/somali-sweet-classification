"""
Clean duplicate images: removes cross-class duplicates and intra-class duplicates.
Run a dry-run first: python src/clean_duplicates.py --dry-run
Then apply: python src/clean_duplicates.py
"""

import os
from pathlib import Path
from PIL import Image
import hashlib
import argparse
from collections import defaultdict

DATA_DIR = Path("data")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

def hash_file(filepath):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def main(dry_run=True):
    # hash -> list of (class_name, path)
    hash_map = defaultdict(list)
    total_images = 0

    for class_dir in DATA_DIR.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for ext in VALID_EXTENSIONS:
            for img_path in class_dir.rglob(f"*{ext}"):
                # Skip checkpoints
                if '.ipynb_checkpoints' in img_path.parts:
                    continue
                total_images += 1
                file_hash = hash_file(img_path)
                hash_map[file_hash].append((class_name, str(img_path)))

    to_delete = []
    # Process cross-class duplicates first
    for file_hash, entries in hash_map.items():
        if len(entries) <= 1:
            continue
        # If all entries are in the same class, keep first, delete rest
        classes_involved = {cls for cls, _ in entries}
        if len(classes_involved) == 1:
            # Intra-class duplicate: keep one, delete others
            first_path = entries[0][1]
            for cls, path in entries[1:]:
                to_delete.append(path)
            print(f"Intra-class dup in {classes_involved}: kept {first_path}, removing {len(entries)-1} copies")
        else:
            # Cross-class duplicate: alert and remove from all classes except the one with most images (or keep in first class)
            # For safety, we'll keep the occurrence in the class that appears first alphabetically,
            # but strongly recommend manual inspection.
            print(f"\n⚠️  CROSS-CLASS DUPLICATE: {file_hash}")
            print(f"   Appears in: {set(classes_involved)}")
            # Sort classes alphabetically, keep the first
            sorted_entries = sorted(entries, key=lambda x: x[0])
            keep_class, keep_path = sorted_entries[0]
            for cls, path in entries:
                if cls != keep_class:
                    to_delete.append(path)
                    print(f"   Will remove from {cls}: {path}")
            print(f"   Keeping in {keep_class}: {keep_path}")

    print(f"\nTotal images scanned: {total_images}")
    print(f"Total duplicates to remove: {len(to_delete)}")

    if dry_run:
        print("\n[DRY RUN] No files deleted. Run without --dry-run to apply changes.")
    else:
        for path in to_delete:
            os.remove(path)
        print(f"\n✅ Deleted {len(to_delete)} duplicate files.")

        # Also delete any remaining .ipynb_checkpoints folders
        for ckpt in DATA_DIR.rglob('.ipynb_checkpoints'):
            import shutil
            shutil.rmtree(ckpt)
            print(f"Removed checkpoint folder: {ckpt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without deleting')
    args = parser.parse_args()
    main(dry_run=args.dry_run)