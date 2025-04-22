import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Define the layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3 input channels (RGB) -> 32 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 -> 64 filters
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64 -> 128 filters (new layer)
        
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # Fully connected layer after flattening the output
        self.fc2 = nn.Linear(512, 4)  # Output layer for 4 classes (e.g., Healthy, Unhealthy, etc.)

        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        # Apply 1st convolution + ReLU + MaxPool
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # Apply 2nd convolution + ReLU + MaxPool
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Apply 3rd convolution + ReLU + MaxPool
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten the output for fully connected layers
        x = x.view(-1, 128 * 16 * 16)  # Flatten to [B, 128 * 16 * 16]

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer (raw logits)

        return x
