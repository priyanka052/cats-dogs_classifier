import torch
import torch.nn as nn

class DogCatCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,64,3)
        self.conv2 = nn.Conv2d(64,128,3)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128,2)

    def forward(self,x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = torch.flatten(x,1)

        x = self.dropout(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x