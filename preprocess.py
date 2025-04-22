import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Define parameters
IMAGE_SIZE = (128, 128)  # Resize dimensions
CROP_SIZE = (100, 100)   # Cropping dimensions

# Function to load and preprocess an image
def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Resize to consistent size
    resized = cv2.resize(image, IMAGE_SIZE)
    
    # Random Cropping
    x = np.random.randint(0, resized.shape[1] - CROP_SIZE[1])
    y = np.random.randint(0, resized.shape[0] - CROP_SIZE[0])
    cropped = resized[y:y + CROP_SIZE[0], x:x + CROP_SIZE[1]]

    # Convert to float and normalize
    normalized = cropped / 255.0  # Scale pixel values to [0,1]
    
    return normalized

# Example demonstration
image_path = "isbi2025_ps3c_train_image_03449.png"  # Replace with actual image path
processed_image = preprocess_image(image_path)

# Display before and after preprocessing
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(processed_image)
plt.title("Processed Image")
plt.axis("off")

plt.show()
