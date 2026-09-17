import torchvision.datasets as datasets
import torchvision.transforms as transforms

data_dir = "data"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root=data_dir,
    transform=transform
)

print("Classes:")
print(dataset.classes)

print("\nClass to Index:")
print(dataset.class_to_idx)

print("\nTotal Images:")
print(len(dataset))

img, label = dataset[0]

print("\nSample Image Shape:")
print(img.shape)

print("\nSample Label:")
print(dataset.classes[label])

print("\nDataset loaded successfully!")