import os

data_dir = "data"
classes = ["Halwo", "buskud", "doolshe"]
min_images = 150

for cls in classes:
    path = os.path.join(data_dir, cls)
    count = len(os.listdir(path)) if os.path.isdir(path) else 0
    print(f"{cls}: {count} images")
    if count < min_images:
        print(f"  ⚠️ WARNING: {cls} has fewer than {min_images} images!")

total = sum(len(os.listdir(os.path.join(data_dir, cls))) for cls in classes if os.path.isdir(os.path.join(data_dir, cls)))
print(f"\nTotal images: {total}")