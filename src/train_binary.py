"""
Training script for the Stage 1 Binary Classifier (Somali Sweet vs Not Somali Sweet)
Uses transfer learning with MobileNetV2.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

# ----------------------------
# 1. Configuration
# ----------------------------
SWEET_DIR = "data"
NON_SWEET_DIR = "data_non_sweet"
BATCH_SIZE = 32
EPOCHS = 15
LR = 5e-4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------
# 2. Binary Dataset Definition
# ----------------------------
class BinarySweetDataset(Dataset):
    """
    Combines sweet images (label 1) and non-sweet images (label 0).
    """
    def __init__(self, sweet_dir, non_sweet_dir):
        self.samples = []
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        
        # 1. Sweet images (label 1)
        sweet_path = Path(sweet_dir)
        for root, _, files in os.walk(sweet_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_exts:
                    self.samples.append((os.path.join(root, file), 1))
                    
        # 2. Non-sweet images (label 0)
        non_sweet_path = Path(non_sweet_dir)
        for root, _, files in os.walk(non_sweet_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_exts:
                    self.samples.append((os.path.join(root, file), 0))
                    
        print(f"Dataset summary:")
        print(f"  - Somali Sweet (label 1): {sum(1 for _, l in self.samples if l == 1)} images")
        print(f"  - Not Somali Sweet (label 0): {sum(1 for _, l in self.samples if l == 0)} images")
        print(f"  - Total images: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            # Open image as RGB
            img = Image.open(path).convert("RGB")
        except Exception as e:
            # If load fails, try to fallback to a neighboring index
            print(f"Warning: Failed to load {path} ({e}). Falling back.")
            return self.__getitem__((idx + 1) % len(self.samples))
        return img, label

# ----------------------------
# 3. Data Loading & Splits
# ----------------------------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

full_dataset = BinarySweetDataset(SWEET_DIR, NON_SWEET_DIR)
num_classes = 2
classes = ["not_somali_sweet", "somali_sweet"]

# Train/Val split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

class TransformedSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

train_dataset = TransformedSubset(train_subset, train_transform)
val_dataset = TransformedSubset(val_subset, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# 4. Class Weights (for loss function)
# ----------------------------
train_labels = [y for _, y in train_dataset]
class_counts = torch.bincount(torch.tensor(train_labels))
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum() * num_classes
class_weights = class_weights.to(DEVICE)
print(f"Class weights: {class_weights}")

# ----------------------------
# 5. Model Setup
# ----------------------------
# Initialize MobileNetV2 with pre-trained weights
model = models.mobilenet_v2(weights='IMAGENET1K_V1')

# Freeze all backbone layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace head with a 2-class classifier
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, num_classes)
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# ----------------------------
# 6. Training Loop
# ----------------------------
os.makedirs("models", exist_ok=True)

best_val_acc = 0.0
best_epoch = 0

for epoch in range(1, EPOCHS + 1):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss / train_total
    train_acc = 100.0 * train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / val_total
    val_acc = 100.0 * val_correct / val_total

    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), "models/binary_mobilenet.pth")
        print(f"  -> New best model saved at epoch {epoch}")

print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

# ----------------------------
# 7. Evaluation
# ----------------------------
print("\nEvaluating on validation set...")
model.load_state_dict(torch.load("models/binary_mobilenet.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Stage 1 Binary Classifier")
plt.savefig("confusion_matrix_binary.png")
plt.close()
print("Confusion matrix saved to confusion_matrix_binary.png")
