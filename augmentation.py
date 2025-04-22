import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import random

# Define augmentation pipeline
augmentation = A.Compose([
    A.Rotate(limit=30, p=0.5),  # Random rotation between -30° to 30°
    A.HorizontalFlip(p=0.5),  # Random horizontal flip
    A.VerticalFlip(p=0.5),  # Random vertical flip
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Contrast adjustment
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Adding Gaussian noise
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),  # Elastic deformation
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), shear=(-10, 10), p=0.5)  # Affine transformations
])

# Function to apply augmentation
def augment_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply augmentation
    augmented = augmentation(image=image)['image']
    
    return image, augmented

# Example demonstration
image_path = "isbi2025_ps3c_train_image_03449.png"  # Replace with actual image path
original, augmented = augment_image(image_path)

# Display before and after augmentation
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(augmented)
plt.title("Augmented Image")
plt.axis("off")

plt.show()
