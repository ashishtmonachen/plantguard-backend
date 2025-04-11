#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify
from utils import predict_disease
from flask_cors import CORS


# In[2]:


app = Flask(__name__)
CORS(app)


# In[3]:


@app.route('/')
def home():
    return "🌿 PlantGuard AI backend is running"


# In[4]:


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    label, confidence = predict_disease(image_file)

    # Example remedies — you can expand this
    remedies = {
        "Tomato___Early_blight": "Use a fungicide and remove infected leaves.",
        "Tomato___Late_blight": "Improve air flow and avoid overhead watering.",
        "Tomato___Healthy": "No issues detected. Keep your plant healthy! 🌿",
    }

    return jsonify({
        'label': label,
        'confidence': round(confidence * 100, 2),
        'remedy': remedies.get(label, "No specific remedy available.")
    })

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




