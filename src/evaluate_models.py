import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from PIL import Image
from pathlib import Path
import numpy as np

# ----------------------------
# Configurations
# ----------------------------
SWEET_DIR = "data"
NON_SWEET_DIR = "data_non_sweet"
BATCH_SIZE = 32
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ----------------------------
# Dataset Helpers
# ----------------------------
class BinarySweetDataset(Dataset):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.samples))
        return img, label

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

# ----------------------------
# 1. Evaluate Stage 1 Binary Classifier
# ----------------------------
def evaluate_binary():
    print("=== Stage 1: Binary Classifier Evaluation ===")
    binary_model_path = "models/binary_mobilenet.pth"
    if not os.path.exists(binary_model_path):
        print(f"Error: {binary_model_path} not found.")
        return
        
    full_dataset = BinarySweetDataset(SWEET_DIR, NON_SWEET_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_subset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    val_dataset = TransformedSubset(val_subset, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 2)
    )
    model.load_state_dict(torch.load(binary_model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    classes = ["not_somali_sweet", "somali_sweet"]
    print("Validation Accuracy:", accuracy_score(all_labels, all_preds))
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    print()

# ----------------------------
# 2. Evaluate Stage 2 Multi-Class Classifiers
# ----------------------------
class ClassConditionalDataset(Dataset):
    def __init__(self, dataset, base_transform):
        self.dataset = dataset
        self.base_transform = base_transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = self.base_transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)

def evaluate_multiclass():
    print("=== Stage 2: Multi-Class Classifiers Evaluation ===")
    import torchvision.datasets as datasets
    
    full_dataset = datasets.ImageFolder(root=SWEET_DIR, transform=None)
    classes = full_dataset.classes
    num_classes = len(classes)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_subset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    val_dataset = ClassConditionalDataset(val_subset, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Baseline Model
    baseline_path = "models/baseline_mobilenet.pth"
    if os.path.exists(baseline_path):
        print("\n--- Baseline Model ---")
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
        model.load_state_dict(torch.load(baseline_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        print("Validation Accuracy:", accuracy_score(all_labels, all_preds))
        print(classification_report(all_labels, all_preds, target_names=classes))
        
    # Tuned Model
    tuned_path = "models/tuned_mobilenet.pth"
    if os.path.exists(tuned_path):
        print("\n--- Tuned Model ---")
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
        model.load_state_dict(torch.load(tuned_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        print("Validation Accuracy:", accuracy_score(all_labels, all_preds))
        print(classification_report(all_labels, all_preds, target_names=classes))
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:\n", cm)

if __name__ == '__main__':
    evaluate_binary()
    evaluate_multiclass()
