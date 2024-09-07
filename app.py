from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os
import pytesseract
from PIL import Image
import re

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and label encoder trained with SMOTE
model = joblib.load('eat_if_model_smote.pkl')
label_encoder = joblib.load('label_encoder_smote.pkl')

# Define the mapping of categories to satirical messages
category_messages = {
    "Balanced": "Eat if you want to stay balanced. Because, why not?",
    "Indulgent": "Eat if you want to gain weight. lol.",
    "Nourishing": "Eat if you want to stay healthy. You know, like a grown-up."
}

# Home page route that renders index.html
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route for manual inputs
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Received data:", data)  # Log the incoming data to check if it's being received
    
    # Extract features from the request data
    input_data = {
        "Calories": data.get('Calories', None),
        "Protein": data.get('Protein', None),
        "Carbohydrate": data.get('Carbohydrate', None),
        "Total fat": data.get('Total fat', None)
    }
    
    df = pd.DataFrame([input_data])
    
    # Make a prediction
    prediction = model.predict(df)
    category = label_encoder.inverse_transform(prediction)[0]
    message = category_messages.get(category, "Unknown category... Just eat whatever you want!")
    
    return jsonify({'category': category, 'message': message})

# Upload route for image processing
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Open the image using PIL
        img = Image.open(file.stream)

        # Use Tesseract to extract text from the image
        extracted_text = pytesseract.image_to_string(img)
        print("Extracted Text:", extracted_text)

        # Use regex to extract the nutritional values from the text
        nutrition_data = extract_nutritional_info(extracted_text)

        if not nutrition_data:
            return jsonify({'error': 'Unable to extract nutritional information.'}), 400

        return jsonify(nutrition_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def extract_nutritional_info(text):
    # Simple regex patterns to extract calories, protein, carbohydrates, and total fat
    nutrition_info = {
        'Calories': extract_value(text, r'Calories\s*(\d+)'),
        'Protein': extract_value(text, r'Protein\s*(\d+\.?\d*)'),
        'Carbohydrate': extract_value(text, r'Carbohydrate\s*(\d+\.?\d*)'),
        'Total Fat': extract_value(text, r'Total\s*Fat\s*(\d+\.?\d*)')
    }
    
    return nutrition_info if any(nutrition_info.values()) else None

def extract_value(text, pattern):
    match = re.search(pattern, text, re.IGNORECASE)
    return float(match.group(1)) if match else None

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the PORT environment variable provided by Heroku
    app.run(host='0.0.0.0', port=port)
