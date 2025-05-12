import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Define labels for each sign (A-Z)
LABELS = list("abcdefghijklmnopqrstuvwxyz")

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def detect_hand(frame):
    """
    Detects the hand using Mediapipe and extracts the hand region.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box of the hand
            h, w, c = frame.shape
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x, x_min)
                y_min = min(y, y_min)
                x_max = max(x, x_max)
                y_max = max(y, y_max)

            # Expand bounding box slightly
            x_min = max(0, x_min - 20)
            y_min = max(0, y_min - 20)
            x_max = min(w, x_max + 20)
            y_max = min(h, y_max + 20)

            hand_crop = frame[y_min:y_max, x_min:x_max]
            return hand_crop

    return None  # No hand detected

def predict_sign(frame):
    """
    Detects hand, processes image, and predicts the sign language letter.
    """
    hand = detect_hand(frame)

    if hand is None:
        print("❌ No hand detected, skipping frame...")
        return None

    try:
        img = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)

        # 🛑 **DEBUG: Show the processed hand image**
        cv2.imshow("Processed Hand Image", img[0, :, :, 0])  # Display grayscale image
        cv2.waitKey(1)  # Let the window update

        prediction = model.predict(img)
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)

        if confidence > 0.85:  # Only accept high-confidence predictions
            predicted_label = LABELS[predicted_index]
            print(f"✅ Prediction: {predicted_label} (Confidence: {confidence:.2f})")
            return predicted_label
        else:
            print("⚠️ Low confidence, waiting for clearer input...")
            return None

    except Exception as e:
        print(f"🚨 Error in processing: {e}")
        return None

def start_camera():
    """
    Opens webcam and starts real-time sign recognition.
    """
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        sign = predict_sign(frame)

        # Draw hand landmarks for visualization
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if sign:
            cv2.putText(frame, f"Predicted: {sign}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Sign to Text Converter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera()
