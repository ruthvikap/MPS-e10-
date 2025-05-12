from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf
import os
import sys

# Ensure Python finds the `p1` directory
sys.path.append(os.path.join(os.getcwd(), "p1"))

# Import sign-to-text processing function
from sign_to_text import predict_sign
from main6 import process_text  # ✅ Ensure this is correctly imported!

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Load the UI

@app.route("/process", methods=["POST"])
def process():
    """
    Converts text to sign language images/GIFs.
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    signs = process_text(text)  # ✅ Ensure `process_text()` returns correct paths

    return jsonify({"signs": signs})

@app.route("/sign-to-text", methods=["POST"])
def recognize_sign():
    """
    Receives an image from the frontend, processes it, and returns the predicted sign.
    """
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    # Read image in OpenCV format
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Predict the sign
    predicted_sign = predict_sign(frame)

    return jsonify({"text": predicted_sign})

if __name__ == "__main__":
    app.run(debug=True)
