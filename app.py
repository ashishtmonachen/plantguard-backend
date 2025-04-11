#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify
from utils import predict_disease
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "🌿 PlantGuard AI backend is running"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    label, confidence = predict_disease(image_file)

    # Mapping model output to remedy-compatible keys
    label_remap = {
        "Tomato Bacterial Spot": "Tomato___Bacterial_spot",
        "Tomato Early Blight": "Tomato___Early_blight",
        "Tomato Late Blight": "Tomato___Late_blight",
        "Tomato Leaf Mold": "Tomato___Leaf_Mold",
        "Tomato Septoria Leaf Spot": "Tomato___Septoria_leaf_spot",
        "Tomato Spider Mites": "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato Target Spot": "Tomato___Target_Spot",
        "Tomato Yellow Leaf Curl Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato Mosaic Virus": "Tomato___Tomato_mosaic_virus",
        "Tomato Healthy": "Tomato___Healthy"
    }

    remedies = {
        "Tomato___Bacterial_spot": "Remove infected leaves and apply copper-based bactericides.",
        "Tomato___Early_blight": "Use fungicide sprays like chlorothalonil and remove debris.",
        "Tomato___Late_blight": "Destroy infected plants. Apply preventive fungicides early.",
        "Tomato___Leaf_Mold": "Improve air circulation and use fungicide with mancozeb.",
        "Tomato___Septoria_leaf_spot": "Remove lower infected leaves and apply fungicide.",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Spray with neem oil or insecticidal soap.",
        "Tomato___Target_Spot": "Use drip irrigation and rotate crops yearly.",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Remove infected plants and control whiteflies.",
        "Tomato___Tomato_mosaic_virus": "Remove and destroy infected plants. Avoid handling when wet.",
        "Tomato___Healthy": "No issues detected. Keep your plant healthy! 🌿"
    }

    remedy_key = label_remap.get(label.strip(), "")
    remedy = remedies.get(remedy_key, "No specific remedy available.")

    return jsonify({
        'label': label,
        'confidence': round(confidence * 100, 2),
        'remedy': remedy
    })

if __name__ == '__main__':
    app.run(debug=True)
