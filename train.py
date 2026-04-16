import torch
import torch.nn as nn
import torch.optim as optim

from model import DogCatCNN
from dataset import get_dataloaders
from utils import calculate_accuracy

train_loader, val_loader = get_dataloaders("dataset/train")

model = DogCatCNN()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(10):

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

torch.save(model.state_dict(), "model.pth")

correct = 0
total = 0

model.eval()   # IMPORTANT

with torch.no_grad():
    for images, labels in val_loader:

        outputs = model(images)
        _, predicted = torch.max(outputs,1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Validation Accuracy:", correct/total)