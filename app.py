from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)

# Trained Model load karein (YAHAN BADALNA THA)
model = tf.keras.models.load_model('har_model.h5', compile=False)

@app.route('/')
def home():
    return "HAR Backend is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['sensor_data']
        input_array = np.array(data).reshape(1, 128, 9) 
        prediction = model.predict(input_array)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        return jsonify({'status': 'success', 'activity': predicted_class})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)