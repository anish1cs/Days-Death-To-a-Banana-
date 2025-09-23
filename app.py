import os
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
from flask import Flask, request, jsonify, render_template
import io

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. SETUP AND CONFIGURATION ---
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# --- Model Configuration ---
MODEL_DIR = 'models'
RESNET_FE_NAME = os.path.join(MODEL_DIR, 'cnn_resnet50_feature_extractor.keras')
MOBILENET_FE_NAME = os.path.join(MODEL_DIR, 'cnn_mobilenetv2_feature_extractor.keras')
LGBM_RESNET_BASE = os.path.join(MODEL_DIR, 'lgbm_resnet_fold_{}.pkl')
LGBM_MOBILENET_BASE = os.path.join(MODEL_DIR, 'lgbm_mobilenet_fold_{}.pkl')
N_SPLITS = 5

# --- Global variables ---
feature_extractor_resnet = None
feature_extractor_mobilenet = None
models_lgbm_resnet = []
models_lgbm_mobilenet = []


# --- 2. MODEL LOADING ---
def load_all_models():
    global feature_extractor_resnet, feature_extractor_mobilenet, models_lgbm_resnet, models_lgbm_mobilenet
    
    print("--- Loading all models. This may take a moment... ---")

    # Load CNN Feature Extractors
    if os.path.exists(RESNET_FE_NAME) and os.path.exists(MOBILENET_FE_NAME):
        feature_extractor_resnet = tf.keras.models.load_model(RESNET_FE_NAME)
        feature_extractor_mobilenet = tf.keras.models.load_model(MOBILENET_FE_NAME)
        print("✅ CNN feature extractors loaded successfully.")
    else:
        raise FileNotFoundError("FATAL ERROR: Could not find the .keras feature extractor files.")

    # Load LightGBM models
    for i in range(1, N_SPLITS + 1):
        models_lgbm_resnet.append(joblib.load(LGBM_RESNET_BASE.format(i)))
        models_lgbm_mobilenet.append(joblib.load(LGBM_MOBILENET_BASE.format(i)))

    print(f"✅ Loaded {len(models_lgbm_resnet) + len(models_lgbm_mobilenet)} LGBM models.")
    print("--- All models loaded successfully! ---")


# --- 3. PREDICTION LOGIC ---
def predict_ensemble(image: Image.Image):
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)[:, :, :3]
    img_batch = np.expand_dims(img_array, axis=0)

    # Extract ResNet features
    preprocessed_resnet = tf.keras.applications.resnet50.preprocess_input(img_batch.copy())
    features_resnet = feature_extractor_resnet.predict(preprocessed_resnet, verbose=0)

    # Extract MobileNet features
    preprocessed_mobilenet = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch.copy())
    features_mobilenet = feature_extractor_mobilenet.predict(preprocessed_mobilenet, verbose=0)

    # Ensemble predictions
    preds_resnet = [model.predict(features_resnet)[0] for model in models_lgbm_resnet]
    preds_mobilenet = [model.predict(features_mobilenet)[0] for model in models_lgbm_mobilenet]

    all_predictions = preds_resnet + preds_mobilenet
    final_prediction = np.mean(all_predictions)

    return final_prediction


# --- 4. FLASK ROUTES ---
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
        prediction = predict_ensemble(image)
        return jsonify({'prediction': f'{prediction:.1f}'})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Server could not process the image'}), 500

# ✅ Health-check route for Render
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


# --- 5. START APP ---
load_all_models()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
