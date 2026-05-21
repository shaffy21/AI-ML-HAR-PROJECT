from flask import Flask, request, jsonify
from flask_cors import CORS
import tf_keras as keras  # Legacy Keras ko direct import kiya
import numpy as np

app = Flask(__name__)
CORS(app)

# Model load karne ka try block
try:
    model = keras.models.load_model('har_model.h5', compile=False)
    print("Model loaded successfully via tf_keras!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/')
def home():
    if model is None:
        return "HAR Backend is Live, but Model file integration pending inside tf_keras!", 500
    return "HAR Backend is Running successfully with Model loaded!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model is not loaded properly'})
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