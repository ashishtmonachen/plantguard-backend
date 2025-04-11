#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os


# In[9]:


MODEL_PATH = 'C:/Users/ashis/Downloads/AIDI 2000/plantguard-ai/model/plant_disease_model.h5'
LABEL_MAP_PATH = 'C:/Users/ashis/Downloads/AIDI 2000/plantguard-ai/model/class_indices.json'


# In[10]:


# Load model and class indices
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(LABEL_MAP_PATH, 'r') as f:
    class_indices = json.load(f)
label_map = {v: k for k, v in class_indices.items()}


# In[11]:


def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)


# In[12]:


def predict_disease(image_file):
    processed_image = preprocess_image(image_file)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))
    label = label_map[predicted_class]
    return label, confidence


# In[ ]:




