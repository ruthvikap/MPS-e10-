import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load pre-trained sign language model
model = tf.keras.models.load_model('sign_language_model.h5')

# Define MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Define class labels for ISL signs
labels = ['Hello', 'Thank You', 'Yes', 'No', 'I Love You', 'Please', 'Sorry']

# Open camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract hand landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
            
            # Convert landmarks to NumPy array
            landmarks = np.array(landmarks).reshape(1, -1)
            
            # Predict sign
            prediction = model.predict(landmarks)
            sign = labels[np.argmax(prediction)]
            
            # Display predicted text
            cv2.putText(frame, sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Sign Language Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
