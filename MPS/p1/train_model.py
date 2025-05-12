import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 🔹 Define dataset path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "static", "letters")  # Update dataset path

# 🔹 Image settings
IMG_SIZE = 64
NUM_CLASSES = 26  # A-Z letters

# 🔹 Augmentation settings
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,  # Hand signs are not flipped in reality
    fill_mode='nearest'
)

def load_dataset():
    """
    Loads images from dataset and applies preprocessing.
    """
    images = []
    labels = []
    label_map = {chr(i + 97): i for i in range(26)}  # {'a':0, 'b':1, ..., 'z':25}

    print(f"🔍 Checking images in: {DATASET_DIR}")

    for file in os.listdir(DATASET_DIR):
        img_path = os.path.join(DATASET_DIR, file)

        if os.path.isfile(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 64x64
            img = img / 255.0  # Normalize pixel values
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            
            label = label_map.get(file[0].lower(), None)  # Extract label from filename
            if label is not None:
                images.append(img)
                labels.append(label)

    if len(images) == 0:
        raise ValueError("❌ No images loaded! Check dataset path or file format.")

    return np.array(images), np.array(labels)

# 🔹 Load dataset
X, y = load_dataset()

# 🔹 Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Apply augmentation
datagen.fit(X_train)

# 🔹 Define CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 🔹 Train model
model = create_model()
print("✅ Training started...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# 🔹 Save model
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
model.save(MODEL_PATH)
print(f"✅ Model trained and saved as {MODEL_PATH}")
