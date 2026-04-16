from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_dataloaders(path, batch_size=32):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Error: The directory '{path}' was not found.\n"
            "Please ensure your dataset is structured as follows:\n"
            "dataset/\n"
            "  train/\n"
            "    dog/ (contains dog images)\n"
            "    cat/ (contains cat images)\n"
        )

    train_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    dataset = datasets.ImageFolder(root=path, transform=train_transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return train_loader, val_loader



