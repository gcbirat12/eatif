import pytesseract
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Load the model and label encoder trained with SMOTE
model = joblib.load('eat_if_model_smote.pkl')
label_encoder = joblib.load('label_encoder_smote.pkl')

# Define the mapping of categories to satirical messages
category_messages = {
    "Balanced": "Eat if you want to stay balanced. Because, why not?",
    "Indulgent": "Eat if you want to gain weight. lol.",
    "Nourishing": "Eat if you want to stay healthy. You know, like a grown-up."
}

# Define the nutritional label keywords and their variations
label_variations = {
    "Calories": ["Calories", "Total Calories"],
    "Protein": ["Protein", "Total Protein"],
    "Carbohydrate": ["Carbohydrate", "Total Carbohydrate"],
    "Total fat": ["Fat", "Total Fat"]
}

# Home page route that renders index.html
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Received data:", data)
    
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

# Helper function to extract nutritional values
def extract_nutritional_values(text):
    extracted_values = {
        "Calories": None,
        "Protein": None,
        "Carbohydrate": None,
        "Total fat": None
    }
    
    # Loop over each label and its variations
    for label, variations in label_variations.items():
        for variation in variations:
            if variation in text:
                try:
                    # Extract the number after the variation
                    number = text.split(variation)[-1].strip().split()[0]
                    extracted_values[label] = float(number)
                except (IndexError, ValueError):
                    continue
    return extracted_values

# Image upload route
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join('/tmp', filename)
    file.save(filepath)

    # Extract text from image using OCR
    image = Image.open(filepath)
    ocr_text = pytesseract.image_to_string(image)

    # Extract nutritional values from the OCR text
    nutritional_values = extract_nutritional_values(ocr_text)

    # Predict using the extracted values
    df = pd.DataFrame([nutritional_values])
    prediction = model.predict(df)
    category = label_encoder.inverse_transform(prediction)[0]
    message = category_messages.get(category, "Unknown category... Just eat whatever you want!")

    return jsonify({
        "nutritional_values": nutritional_values,
        "category": category,
        "message": message
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
