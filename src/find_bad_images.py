"""Try to load and transform every image; report any that fail."""
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

DATA_DIR = Path("data")
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

bad_files = []
for img_path in DATA_DIR.rglob("*.*"):
    if not img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}:
        continue
    try:
        img = Image.open(img_path).convert("RGB")
        transform(img)  # force actual processing
    except Exception as e:
        bad_files.append((img_path, str(e)))
        print(f"❌ {img_path}   → {e}")

if not bad_files:
    print("✅ All images are OK. The crash may be due to another cause.")
else:
    print(f"\n{len(bad_files)} problematic files found. Delete them or move them out of data/.")