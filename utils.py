import torch
def calculate_accuracy(model, loader):
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs,1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total