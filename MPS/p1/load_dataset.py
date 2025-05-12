import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Path to dataset
dataset_path = r"C:\Users\Raghavi\OneDrive\Desktop\me\project1\p1\letters"

# Parameters
img_size = (64, 64)  # Resize images to this size
num_classes = 26  # A-Z letters

# Data storage
images = []
labels = []

# Load images directly from files (since there are no folders)
for img_name in sorted(os.listdir(dataset_path), key=str.lower):
    img_path = os.path.join(dataset_path, img_name)

    if img_name.endswith(".jpg") or img_name.endswith(".png"):  # Ensure it's an image file
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        img = cv2.resize(img, img_size)  # Resize to (64, 64)
        
        images.append(img)
        labels.append(ord(img_name[0]) - ord('a'))  # Convert 'a' to 0, 'b' to 1, ..., 'z' to 25

# Convert lists to numpy arrays
images = np.array(images).reshape(-1, img_size[0], img_size[1], 1)  # Add channel dimension
labels = np.array(labels)

# One-hot encode labels
labels = to_categorical(labels, num_classes)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Save processed data
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("✅ Dataset loaded and saved successfully!")
