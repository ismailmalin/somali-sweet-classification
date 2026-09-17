"""
Tuned training script for Somali Sweet Classifier.
Improves baseline by:
- Fine-tuning last few MobileNetV2 layers
- Lower learning rate
- Stronger augmentation for minority classes (buskud, doolshe)
- Learning rate scheduler
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 1. Configuration
# ----------------------------
DATA_DIR = "data"
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4                # lower learning rate for fine-tuning
SEED = 42
UNFREEZE_BLOCKS = 3      # unfreeze the last 3 blocks of MobileNetV2 features
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------
# 2. Transforms (with class-conditional extra augs)
# ----------------------------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

base_train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

extra_aug = transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # tiny shift
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Class indices (alphabetical from ImageFolder)
# buskud=0, doolshe=1, halwo=2
MINORITY_CLASSES = {0, 1}   # buskud & doolshe

class ClassConditionalDataset(torch.utils.data.Dataset):
    """Wraps a dataset and applies extra augmentation to specified classes."""
    def __init__(self, dataset, base_transform, extra_transform, minority_classes):
        self.dataset = dataset
        self.base_transform = base_transform
        self.extra_transform = extra_transform
        self.minority_classes = minority_classes

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        # Apply base transforms always
        x = self.base_transform(x)
        # Apply extra transforms only for minority classes
        if y in self.minority_classes:
            x = self.extra_transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)

# ----------------------------
# 3. Data Loading & Splitting
# ----------------------------
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None)
classes = full_dataset.classes
num_classes = len(classes)
print(f"Classes: {classes}")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_dataset = ClassConditionalDataset(train_subset, base_train_tf, extra_aug, MINORITY_CLASSES)
val_dataset = ClassConditionalDataset(val_subset, val_tf, None, set())   # no extra aug on val

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# 4. Class Weights
# ----------------------------
train_labels = [full_dataset[idx][1] for idx in train_subset.indices]
class_counts = torch.bincount(torch.tensor(train_labels))
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum() * num_classes
class_weights = class_weights.to(DEVICE)
print(f"Class weights: {class_weights}")

# ----------------------------
# 5. Model Setup (Fine-Tuning)
# ----------------------------
model = models.mobilenet_v2(weights='IMAGENET1K_V1')

# First freeze everything
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last UNFREEZE_BLOCKS blocks of features
# MobileNetV2 features are organized in a list of blocks (children of .features)
features = list(model.features.children())
total_blocks = len(features)
unfreeze_from = max(0, total_blocks - UNFREEZE_BLOCKS)
print(f"Total feature blocks: {total_blocks}, unfreezing last {UNFREEZE_BLOCKS} (indices {unfreeze_from}..{total_blocks-1})")
for i, block in enumerate(features):
    if i >= unfreeze_from:
        for param in block.parameters():
            param.requires_grad = True

# Replace classifier
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, num_classes)
)
# Classifier is always trainable (we never froze it)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
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

    print(f"Epoch {epoch:2d}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

    # Scheduler step
    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), "models/tuned_mobilenet.pth")
        print(f"  -> New best model saved (epoch {epoch})")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, "models/tuned_checkpoint.pth")

print(f"\nTuned training completed. Best val accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

# ----------------------------
# 7. Evaluation & Comparison with Baseline
# ----------------------------
print("\nEvaluating tuned model on validation set...")
model.load_state_dict(torch.load("models/tuned_mobilenet.pth"))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nTuned Model Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Tuned MobileNetV2")
plt.savefig("confusion_matrix_tuned.png")
plt.close()
print("Tuned confusion matrix saved to confusion_matrix_tuned.png")

# Compare with baseline
if os.path.exists("models/baseline_mobilenet.pth"):
    print("\nComparing with baseline model...")
    baseline_model = models.mobilenet_v2(weights=None, num_classes=num_classes)
    baseline_model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, num_classes)
    )
    baseline_model.load_state_dict(torch.load("models/baseline_mobilenet.pth"))
    baseline_model.to(DEVICE)
    baseline_model.eval()

    baseline_preds, baseline_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = baseline_model(inputs)
            _, preds = torch.max(outputs, 1)
            baseline_preds.extend(preds.cpu().numpy())
            baseline_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import accuracy_score
    baseline_acc = accuracy_score(baseline_labels, baseline_preds) * 100
    tuned_acc = accuracy_score(all_labels, all_preds) * 100
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    print(f"Tuned accuracy:    {tuned_acc:.2f}%")
    print(f"Improvement:       {tuned_acc - baseline_acc:.2f}%")
else:
    print("Baseline model not found. Skipping comparison.")