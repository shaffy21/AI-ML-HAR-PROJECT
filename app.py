from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np

# Naye Keras ke serializers ko bypass karne ke liye registry fix
try:
    from keras.src.saving import serialization_lib
    serialization_lib.safe_mode = False
except:
    pass

app = Flask(__name__)
CORS(app)

# Model load karne ka ek naya aur absolute safe tareeqa
try:
    # custom_objects mein humne pass-through de diya hai taaki quantization error na aaye
    model = tf.keras.models.load_model(
        'har_model.h5', 
        custom_objects={'Dense': tf.keras.layers.Dense}, 
        compile=False
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/')
def home():
    if model is None:
        return "Backend is Live, but Model integration pending. Check logs!", 500
    return "HAR Backend is Running successfully with Model loaded! 🎉🚀"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'})
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