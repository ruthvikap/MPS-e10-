from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import os
import sys
import pickle
import mediapipe as mp
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated.")

# Add p1 directory to path
sys.path.append(os.path.join(os.getcwd(), "p1"))

# Import your custom sign processing functions
from p1.sign_to_text import predict_sign
from p1.main6 import process_text

# Initialize Flask app and SocketIO
app = Flask(__name__, static_folder="static")
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Load ML model for real-time recognition
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    print("Error loading the model:", e)
    model = None

# Label dictionary for prediction output
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello',
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please',
    32: 'You are welcome.'
}

# Home page
@app.route("/")
def index():
    return render_template("index.html")

# TEXT → SIGN route
@app.route("/process", methods=["POST"])
def process():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    signs = process_text(text)
    if not signs:
        return jsonify({"error": "No matching sign found"}), 404

    return jsonify({"signs": signs})

# IMAGE → TEXT route
@app.route("/sign-to-text", methods=["POST"])
def recognize_sign():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    predicted_sign = predict_sign(frame)
    return jsonify({"text": predicted_sign})

# REAL-TIME WEBCAM STREAM → TEXT via socket
@socketio.on('connect')
def handle_connect():
    print('Client connected')

def generate_frames():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    prediction_proba = model.predict_proba([np.asarray(data_aux)])
                    confidence = max(prediction_proba[0])
                    predicted_character = labels_dict[int(prediction[0])]

                    socketio.emit('prediction', {
                        'text': predicted_character,
                        'confidence': confidence
                    })

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, f"{predicted_character} ({confidence*100:.2f}%)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                except Exception as e:
                    pass

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# TRANSLATION: text to Hindi (default)
from deep_translator import GoogleTranslator

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text input'}), 400

    text = data['text']
    to_lang = data.get('to', 'hi')  # default target language: Hindi

    try:
        translated = GoogleTranslator(source='auto', target=to_lang).translate(text)
        return jsonify({'translated_text': translated})
    except Exception as e:
        return jsonify({'error': f'Translation failed: {str(e)}'}), 500

# Run the app
if __name__ == "__main__":
    socketio.run(app, debug=True)
