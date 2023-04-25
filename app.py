from flask import Flask, request, jsonify, render_template
import numpy as np
import io
from PIL import Image
import tensorflow as tf
import os

from functools import lru_cache

app = Flask(__name__)

# Load the trained model
@lru_cache(maxsize=1)
def get_model():
    return tf.keras.models.load_model('my_model.h5')

species = ['Cheetah', 'Leopard', 'Lion', 'Puma', 'Tiger']

@app.route('/form')
def index():
    return render_template('./form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    img_bytes = request.files['image'].read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # Preprocess the image
    img = img.resize((64, 64), Image.LANCZOS)  # Resize the image to (64, 64)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class probabilities
    with tf.device('/CPU:0'):  # Use CPU if no GPU is available
        model = get_model()
        prob = model.predict(img_array)[0]  # Use eager execution mode

    result = {species[i]: float(prob[i]) for i in range(len(species))}
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the $PORT environment variable
    app.run(host='0.0.0.0', port=port)  # Start the Flask app on the specified port
