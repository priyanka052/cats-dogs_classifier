import torch
import torch.nn as nn
import torch.optim as optim
import os

from model import DogCatCNN
from dataset import get_dataloaders
from utils import calculate_accuracy

# ✅ CHECK PATH
path = r"C:\Users\Priyanka\Desktop\PetImages"
print("Path exists:", os.path.exists(path))

# ✅ LOAD DATA
train_loader, val_loader = get_dataloaders(path)

# 🔥 DEBUG (IMPORTANT)
print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))

# Check classes
print("Classes:", train_loader.dataset.dataset.classes)

# ❌ STOP IF DATA NOT LOADED
if len(train_loader) == 0:
    print("❌ ERROR: Dataset not loading properly")
    exit()

# ✅ MODEL
model = DogCatCNN()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

print("🚀 Training started...\n")

# ✅ TRAIN LOOP
for epoch in range(3):

    model.train()
    total_loss = 0

    for images, labels in train_loader:

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    val_acc = calculate_accuracy(model, val_loader)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

# ✅ SAVE MODEL
torch.save(model.state_dict(), "model.pth")

print("\n✅ Model saved as model.pth")