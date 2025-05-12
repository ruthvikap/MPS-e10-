import cv2
import numpy as np
import tensorflow as tf
import os

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Define labels for sign recognition
LABELS = list("abcdefghijklmnopqrstuvwxyz")

def predict_sign(frame):
    """Predict sign language letter from an image frame."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (64, 64))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension

    prediction = model.predict(img)
    predicted_label = LABELS[np.argmax(prediction)]
    
    return predicted_label

def start_camera():
    """Open webcam and recognize signs in real-time."""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera feed not available.")
            break

        sign = predict_sign(frame)
        cv2.putText(frame, f"Predicted: {sign}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Sign to Text Converter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera()
