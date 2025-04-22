import os
import cv2  # OpenCV for image loading
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CervicalCellDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        """
        Custom PyTorch Dataset to load cervical cell images (PNG, JPG).
        
        Args:
            root_dir (str): Path to the dataset directory (folders named by class labels).
            transform (callable, optional): Transformations to apply to the images.
            is_test (bool, optional): Whether the dataset is for testing (i.e., no labels).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

        if self.is_test:
            # Test data: load images from the test folder (no labels)
            self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.labels = None  # No labels for test data
        else:
            # Training data: load images from subfolders named by class labels
            self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])  
            self.image_paths = []
            self.labels = {}

            # Collect all image paths and assign labels based on folder names
            for label, class_name in enumerate(self.classes):
                class_dir = os.path.join(root_dir, class_name)
                for file in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, file)
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Accept PNG, JPG
                        self.image_paths.append(file_path)
                        self.labels[file_path] = label  # Store the label

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, file_path):
        """
        Load an image from a given file path.
        Supports PNG and JPG.
        """
        image = cv2.imread(file_path)  # Load PNG/JPG
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        return Image.fromarray(image)  # Convert to PIL Image

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label (if not test data).
        """
        img_path = self.image_paths[idx]
        image = self.load_image(img_path)  # Load image
        
        if self.is_test:
            label = None  # No label for test data
        else:
            label = self.labels[img_path]  # Get label for training data

        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image, label


# Define Transformations with Data Augmentation for the Training Dataset
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to a fixed size (128x128)
    transforms.RandomRotation(30),  # Random rotation between -30 and +30 degrees
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomVerticalFlip(),  # Random vertical flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # Random crop with resize
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Random affine transformations (rotation, scaling, translation)
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image to [-1, 1]
])

# Define Transformations for the Test Dataset (without augmentation)
transform_test = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to a fixed size (128x128)
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image to [-1, 1]
])

# Create Dataset and DataLoader for training
dataset_path_train = "../final_proj_dataset/isbi2025-ps3c-train-dataset"  # Path to the training dataset
dataset_train = CervicalCellDataset(root_dir=dataset_path_train, transform=transform_train, is_test=False)
trainloader = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=2)

# Create Dataset and DataLoader for testing (no labels)
dataset_path_test = "../final_proj_dataset/isbi2025-ps3c-test-dataset"  # Path to the test dataset
dataset_test = CervicalCellDataset(root_dir=dataset_path_test, transform=transform_test, is_test=True)
testloader = DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=2)