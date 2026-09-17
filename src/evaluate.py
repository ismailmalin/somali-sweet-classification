"""
Reusable evaluation script for Somali Sweet Classifier.
Usage:
  python src/evaluate.py --model models/baseline_mobilenet.pth
  python src/evaluate.py --model models/tuned_mobilenet.pth
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

def get_val_loader(data_dir="data", batch_size=32):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    full_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
    # Use the same 80/20 split with seed 42 for consistency
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False), full_dataset.classes

def load_model(model_path, num_classes, device):
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model .pth file')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader, classes = get_val_loader()
    num_classes = len(classes)
    print(f"Classes: {classes}")

    model = load_model(args.model, num_classes, device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\nEvaluation for model: {args.model}")
    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {os.path.basename(args.model)}")
    out_name = f"confusion_matrix_{os.path.splitext(os.path.basename(args.model))[0]}.png"
    plt.savefig(out_name)
    plt.close()
    print(f"Confusion matrix saved to {out_name}")

if __name__ == "__main__":
    main()