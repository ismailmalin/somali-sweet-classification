"""
Script to download non-sweet images from Picsum Photos to act as the negative class dataset.
"""

import os
import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

OUT_DIR = os.path.join("data_non_sweet", "not_somali_sweet")
NUM_IMAGES = 400
NUM_THREADS = 16

def download_single_image(index):
    url = f"https://picsum.photos/224/224?random={index}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = response.read()
        
        # Verify it's a valid image using PIL
        img = Image.open(io.BytesIO(data))
        img.verify()
        
        # Re-open (verify leaves image in closed/invalid state)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        
        filename = f"non_sweet_{index:04d}.jpg"
        filepath = os.path.join(OUT_DIR, filename)
        img.save(filepath, "JPEG")
        return True, index
    except Exception as e:
        return False, f"Index {index} failed: {e}"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Starting download of {NUM_IMAGES} images into '{OUT_DIR}' using {NUM_THREADS} threads...")
    
    success_count = 0
    failures = []
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(download_single_image, i): i for i in range(1, NUM_IMAGES + 1)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                success, result = future.result()
                if success:
                    success_count += 1
                    if success_count % 50 == 0:
                        print(f"Downloaded {success_count}/{NUM_IMAGES} images...")
                else:
                    failures.append(result)
            except Exception as e:
                failures.append(f"Unexpected error for index {idx}: {e}")
                
    print(f"\nDownload completed. Successfully downloaded: {success_count}/{NUM_IMAGES}")
    if failures:
        print(f"Failed downloads: {len(failures)}")
        print("First 5 failures:")
        for f in failures[:5]:
            print(f"  - {f}")

if __name__ == "__main__":
    main()
