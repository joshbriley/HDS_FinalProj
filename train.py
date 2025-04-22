import torch
import torch.nn as nn
import torch.optim as optim
from loader import CervicalCellDataset, transform_train, transform_test  # Import the Dataset and transforms from loader.py
import matplotlib.pyplot as plt
from model import SimpleCNN  # Assuming your CNN model is in a separate file

def train(model, trainloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for inputs, labels in trainloader:
            optimizer.zero_grad()  # Zero out the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(trainloader)}")


if __name__ == '__main__':
    # Initialize Dataset and DataLoader for training
    dataset_path_train = "../final_proj_dataset/isbi2025-ps3c-train-dataset"  # Replace with the correct training path
    dataset_train = CervicalCellDataset(root_dir=dataset_path_train, transform=transform_train, is_test=False)
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=2)

    # Initialize Dataset and DataLoader for testing (no labels)
    dataset_path_test =  "../final_proj_dataset/isbi2025-ps3c-test-dataset"
    dataset_test = CervicalCellDataset(root_dir=dataset_path_test, transform=transform_test, is_test=True)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=2)

    # Define the model, loss function, and optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, trainloader, criterion, optimizer, num_epochs=10)