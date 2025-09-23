import os
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
from flask import Flask, request, jsonify, render_template
import io

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Flask app ---
app = Flask(__name__)

# --- Model paths ---
MODEL_DIR = 'models'
RESNET_FE_NAME = os.path.join(MODEL_DIR, 'cnn_resnet50_feature_extractor.keras')
MOBILENET_FE_NAME = os.path.join(MODEL_DIR, 'cnn_mobilenetv2_feature_extractor.keras')
LGBM_RESNET_BASE = os.path.join(MODEL_DIR, 'lgbm_resnet_fold_{}.pkl')
LGBM_MOBILENET_BASE = os.path.join(MODEL_DIR, 'lgbm_mobilenet_fold_{}.pkl')
N_SPLITS = 5

# --- Global placeholders ---
feature_extractor_resnet = None
feature_extractor_mobilenet = None
models_lgbm_resnet = []
models_lgbm_mobilenet = []


# --- Load Models (Lazy) ---
def load_all_models():
    """
    Loads CNN feature extractors + LightGBM models into memory.
    Called only when needed (lazy load).
    """
    global feature_extractor_resnet, feature_extractor_mobilenet
    global models_lgbm_resnet, models_lgbm_mobilenet

    if feature_extractor_resnet is not None and feature_extractor_mobilenet is not None:
        return  # already loaded

    print("--- Lazy-loading all models. This may take a moment... ---")

    # Load CNN extractors
    if os.path.exists(RESNET_FE_NAME) and os.path.exists(MOBILENET_FE_NAME):
        feature_extractor_resnet = tf.keras.models.load_model(RESNET_FE_NAME)
        feature_extractor_mobilenet = tf.keras.models.load_model(MOBILENET_FE_NAME)
        print("✅ CNN feature extractors loaded.")
    else:
        raise FileNotFoundError("❌ Could not find the .keras feature extractor files.")

    # Load LightGBM models
    for i in range(1, N_SPLITS + 1):
        models_lgbm_resnet.append(joblib.load(LGBM_RESNET_BASE.format(i)))
        models_lgbm_mobilenet.append(joblib.load(LGBM_MOBILENET_BASE.format(i)))

    print(f"✅ Loaded {len(models_lgbm_resnet) + len(models_lgbm_mobilenet)} LGBM models.")
    print("--- Models ready! ---")


# --- Prediction logic ---
def predict_ensemble(image: Image.Image):
    """Take an image and return prediction (days left)."""
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)[:, :, :3]
    img_batch = np.expand_dims(img_array, axis=0)

    # Extract features
    preprocessed_resnet = tf.keras.applications.resnet50.preprocess_input(img_batch.copy())
    features_resnet = feature_extractor_resnet.predict(preprocessed_resnet, verbose=0)

    preprocessed_mobilenet = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch.copy())
    features_mobilenet = feature_extractor_mobilenet.predict(preprocessed_mobilenet, verbose=0)

    # Predict with LightGBM
    preds_resnet = [model.predict(features_resnet)[0] for model in models_lgbm_resnet]
    preds_mobilenet = [model.predict(features_mobilenet)[0] for model in models_lgbm_mobilenet]

    all_predictions = preds_resnet + preds_mobilenet
    return np.mean(all_predictions)


# --- Routes ---
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def handle_prediction():
    global feature_extractor_resnet, feature_extractor_mobilenet

    # Lazy load here
    if feature_extractor_resnet is None or feature_extractor_mobilenet is None:
        load_all_models()

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        prediction = predict_ensemble(image)
        return jsonify({'prediction': f'{prediction:.1f}'})
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({'error': 'Server could not process the image'}), 500


# --- Local run ---
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
