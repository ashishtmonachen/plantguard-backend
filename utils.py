import numpy as np
from PIL import Image
import json
import tensorflow as tf
import io

# Paths
MODEL_PATH = "model/plant_disease_model.tflite"
LABEL_MAP_PATH = "model/class_indices.json"

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label map
with open(LABEL_MAP_PATH, "r") as f:
    class_indices = json.load(f)
label_map = {v: k for k, v in class_indices.items()}

def preprocess_image(image_file):
    image_file.seek(0)
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_disease(image_file):
    processed_image = preprocess_image(image_file)

    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = int(np.argmax(output_data))
    confidence = float(np.max(output_data))
    label = label_map[predicted_class]

    return label, confidence
