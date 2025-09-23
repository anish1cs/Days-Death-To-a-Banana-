import os
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
from flask import Flask, request, jsonify, render_template
import io

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. SETUP ---
app = Flask(__name__)

# --- Model Paths ---
MODEL_DIR = 'models'
RESNET_FE_NAME = os.path.join(MODEL_DIR, 'cnn_resnet50_feature_extractor.keras')
LGBM_MODEL_NAME = os.path.join(MODEL_DIR, 'lgbm_regressor.pkl')

# --- Globals ---
feature_extractor_resnet = None
lgbm_model = None


# --- 2. MODEL LOADING ---
def load_models():
    global feature_extractor_resnet, lgbm_model
    print("--- Loading ResNet + LGBM ---")

    if not os.path.exists(RESNET_FE_NAME) or not os.path.exists(LGBM_MODEL_NAME):
        raise FileNotFoundError("FATAL: Model files not found in /models")

    feature_extractor_resnet = tf.keras.models.load_model(RESNET_FE_NAME)
    lgbm_model = joblib.load(LGBM_MODEL_NAME)

    print("✅ ResNet extractor and LGBM model loaded.")


# --- 3. PREDICTION ---
def predict_days(image: Image.Image):
    # Preprocess for ResNet
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)[:, :, :3]
    img_batch = np.expand_dims(img_array, axis=0)

    preprocessed = tf.keras.applications.resnet50.preprocess_input(img_batch)
    features = feature_extractor_resnet.predict(preprocessed, verbose=0)

    prediction = lgbm_model.predict(features)[0]
    return prediction


# --- 4. ROUTES ---
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        days = predict_days(image)
        return jsonify({'prediction': f'{days:.1f}'})
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': 'Server could not process the image'}), 500


# --- 5. APP START ---
load_models()  # load once at startup

if __name__ == '__main__':
    app.run(debug=True)
