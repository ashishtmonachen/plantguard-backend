import io
import json
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model and class label map
MODEL_PATH = 'model/plant_disease_model.h5'
LABEL_MAP_PATH = 'model/class_indices.json'

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(LABEL_MAP_PATH, 'r') as f:
    class_indices = json.load(f)
label_map = {v: k for k, v in class_indices.items()}

def preprocess_image(image_file):
    image_file.seek(0)
    image_bytes = image_file.read()
    print("Image byte size:", len(image_bytes))  # For logs

    if not image_bytes:
        raise ValueError("Uploaded image is empty or unreadable.")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_disease(image_file):
    # Load model only when needed (saves memory)
    model = tf.keras.models.load_model("model/plant_disease_model.h5", compile=False)
    with open("model/class_indices.json", 'r') as f:
        class_indices = json.load(f)
    label_map = {v: k for k, v in class_indices.items()}

    processed_image = preprocess_image(image_file)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))
    label = label_map[predicted_class]
    return label, confidence
