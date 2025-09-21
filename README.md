Banana Ripeness Prediction Web App
This project serves a sophisticated, dual-CNN hybrid ensemble model (ResNet50 + MobileNetV2) through a simple and modern web interface built with Flask. The application allows a user to upload an image of a banana and receive an AI-powered prediction of its remaining edible days.

Project Structure
The project is organized into the following directories:

/models/: Contains all 12 pre-trained model files required for the ensemble prediction.

/static/: Holds static assets like background images for the webpage.

/templates/: Contains the index.html file for the user interface.

app.py: The core Flask web server that handles logic and predictions.

requirements.txt: A list of all necessary Python packages.

Setup Instructions
Follow these steps to set up and run the application on your local machine.

1. Place Your Trained Models
This is the most critical step. From your Jupyter Notebook project, copy all 12 of your trained model files into the models/ folder of this web app project. The application will not run without them.

The required files are:

cnn_resnet50_feature_extractor.keras

cnn_mobilenetv2_feature_extractor.keras

All lgbm_resnet_fold_*.pkl files (5 files)

All lgbm_mobilenet_fold_*.pkl files (5 files)

2. Create a Virtual Environment
Using a virtual environment is a best practice to keep project dependencies isolated and avoid conflicts.

# In the VS Code terminal, navigate to your project folder and create the environment
python -m venv venv

# Activate the environment (on Windows PowerShell)
.\venv\Scripts\Activate.ps1

# On macOS/Linux, you would use:
# source venv/bin/activate

After activation, you will see (venv) at the beginning of your terminal prompt. This indicates that the virtual environment is active.

3. Install Required Libraries
Once your virtual environment is active, install all the necessary Python packages using the requirements.txt file.

# This command reads the requirements.txt file and installs everything at once
pip install -r requirements.txt

4. (Optional) Add a Background Image
You can place a background image named background.jpg into the static/ folder to improve the app's visual appeal. The index.html file is already configured to use it if it exists.

How to Run the Web App
Make sure your virtual environment is active ((venv) should be visible in your terminal).

Run the Flask application from the main project directory:

python app.py
