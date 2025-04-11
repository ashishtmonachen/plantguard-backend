#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify
from utils import predict_disease
from flask_cors import CORS
import openai

app = Flask(__name__)
CORS(app)

# 🔐 Hardcoded OpenAI key (ONLY for demo purposes — don’t do this in production)
openai.api_key = "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"  # Replace with your real API key

@app.route('/')
def home():
    return "🌿 PlantGuard AI backend is running"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    label, confidence = predict_disease(image_file)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a plant disease expert. Give short and actionable remedies."},
                {"role": "user", "content": f"What is the treatment for {label} in tomato plants?"}
            ],
            max_tokens=100
        )
        remedy = response.choices[0].message.content.strip()
    except Exception as e:
        remedy = "Remedy not available at the moment. Please try again later."

    return jsonify({
        'label': label,
        'confidence': round(confidence * 100, 2),
        'remedy': remedy
    })

if __name__ == '__main__':
    app.run(debug=True)
