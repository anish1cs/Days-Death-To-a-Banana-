import os
import numpy as np
import tensorflow as tf
import joblib
import lightgbm as lgb
from PIL import Image
from flask import Flask, request, jsonify, render_template
import io

# Suppress TensorFlow warnings for a cleaner output, making the server logs easier to read.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. SETUP AND CONFIGURATION ---
# Initialize the Flask application. Flask is the engine that runs our web server.
app = Flask(__name__)

# --- Model Configuration ---
# Define the locations of all the model files.
# These paths are relative to this app.py file and MUST match your folder structure.
MODEL_DIR = 'models'
RESNET_FE_NAME = os.path.join(MODEL_DIR, 'cnn_resnet50_feature_extractor.keras')
MOBILENET_FE_NAME = os.path.join(MODEL_DIR, 'cnn_mobilenetv2_feature_extractor.keras')
LGBM_RESNET_BASE = os.path.join(MODEL_DIR, 'lgbm_resnet_fold_{}.pkl')
LGBM_MOBILENET_BASE = os.path.join(MODEL_DIR, 'lgbm_mobilenet_fold_{}.pkl')
N_SPLITS = 5  # The number of folds you used during training. This is crucial.

# --- Global variables to hold the loaded models ---
# We load the models into memory once when the server starts to make predictions fast.
feature_extractor_resnet = None
feature_extractor_mobilenet = None
models_lgbm_resnet = []
models_lgbm_mobilenet = []


# --- 2. MODEL LOADING ---
def load_all_models():
    """
    Loads all 12 parts of the complex ensemble model into memory.
    This function is called only once when the server starts up.
    """
    global feature_extractor_resnet, feature_extractor_mobilenet, models_lgbm_resnet, models_lgbm_mobilenet
    
    print("--- Loading all models. This may take a moment... ---")
    
    # Load CNN Feature Extractors (The "Expert Eyes")
    if os.path.exists(RESNET_FE_NAME) and os.path.exists(MOBILENET_FE_NAME):
        feature_extractor_resnet = tf.keras.models.load_model(RESNET_FE_NAME)
        feature_extractor_mobilenet = tf.keras.models.load_model(MOBILENET_FE_NAME)
        print("✅ CNN feature extractors loaded successfully.")
    else:
        raise FileNotFoundError("FATAL ERROR: Could not find the .keras feature extractor files in the 'models' folder.")

    # Load the 5 LightGBM models for the ResNet branch
    for i in range(1, N_SPLITS + 1):
        model_path = LGBM_RESNET_BASE.format(i)
        if os.path.exists(model_path):
            models_lgbm_resnet.append(joblib.load(model_path))
        else:
            raise FileNotFoundError(f"FATAL ERROR: Could not find the ResNet model for fold {i} at {model_path}")
    print(f"✅ {len(models_lgbm_resnet)} ResNet-based LGBM models loaded.")
    
    # Load the 5 LightGBM models for the MobileNetV2 branch
    for i in range(1, N_SPLITS + 1):
        model_path = LGBM_MOBILENET_BASE.format(i)
        if os.path.exists(model_path):
            models_lgbm_mobilenet.append(joblib.load(model_path))
        else:
            raise FileNotFoundError(f"FATAL ERROR: Could not find the MobileNet model for fold {i} at {model_path}")
    print(f"✅ {len(models_lgbm_mobilenet)} MobileNet-based LGBM models loaded.")
    
    print("--- All models loaded successfully! The web app is ready. ---")


# --- 3. PREDICTION LOGIC ---
def predict_ensemble(image: Image.Image):
    """
    Takes a user-uploaded image and returns the final ensemble prediction.
    This logic is a direct translation of the prediction pipeline from your Jupyter Notebook.
    """
    # Resize and prepare the image in the required 224x224 format
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)[:, :, :3] # Ensure 3 color channels (RGB)
    img_batch = np.expand_dims(img_array, axis=0)

    # Preprocess the image for each CNN type and extract the feature vectors
    preprocessed_resnet = tf.keras.applications.resnet50.preprocess_input(img_batch.copy())
    features_resnet = feature_extractor_resnet.predict(preprocessed_resnet, verbose=0)

    preprocessed_mobilenet = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch.copy())
    features_mobilenet = feature_extractor_mobilenet.predict(preprocessed_mobilenet, verbose=0)
    
    # Get predictions from all 10 LightGBM models
    preds_resnet = [model.predict(features_resnet)[0] for model in models_lgbm_resnet]
    preds_mobilenet = [model.predict(features_mobilenet)[0] for model in models_lgbm_mobilenet]
    
    # Combine all predictions and calculate the final average
    all_predictions = preds_resnet + preds_mobilenet
    final_prediction = np.mean(all_predictions)
    
    return final_prediction


# --- 4. FLASK WEB ROUTES ---
# This section defines the URLs for our web app.

@app.route('/', methods=['GET'])
def home():
    """Renders the main homepage (index.html) when a user visits the root URL."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """
    Handles the image file upload from the webpage, calls the prediction function,
    and returns the result as JSON data.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
        
    try:
        # Read the image file from the user's upload into memory
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get the prediction from our powerful ensemble model
        prediction = predict_ensemble(image)
        
        # Return the final result to the webpage
        return jsonify({'prediction': f'{prediction:.1f}'})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Server could not process the image'}), 500


# --- 5. START THE APPLICATION ---
# This block runs only when you execute "python app.py" in the terminal.
if __name__ == '__main__':
    # Load the models into memory first when the application starts
    load_all_models()
    # Run the Flask web server
    # debug=True allows the server to auto-reload when you save changes to this file.
    app.run(debug=True)

