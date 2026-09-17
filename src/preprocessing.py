"""
Production-ready image preprocessing for Somali Sweet Classifier.
Use the same transforms for training, validation, and inference.
"""

import torchvision.transforms as transforms

# ImageNet stats (required for MobileNetV2 pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transforms():
    """Augmentations for training set."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_val_transforms():
    """No augmentation for validation/evaluation."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_inference_transforms():
    """Same as val – single image preprocessing for prediction."""
    return get_val_transforms()