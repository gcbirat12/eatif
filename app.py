from flask import Flask, request, jsonify, render_template
import pytesseract
from PIL import Image
import os
import re

app = Flask(__name__)

# Set the TESSDATA_PREFIX environment variable
os.environ["TESSDATA_PREFIX"] = "/app/tessdata/"

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route for form inputs
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        calories = data.get('Calories')
        protein = data.get('Protein')
        carbohydrate = data.get('Carbohydrate')
        total_fat = data.get('Total fat')

        # Simple logic based on the input for the Eat If feature
        if calories and calories < 300:
            message = "Eat If... it fits within your daily intake."
        else:
            message = "Eat If... you're ready for some extra energy!"

        return jsonify({"message": message})

    except Exception as e:
        return jsonify({"error": str(e)})

# Route to handle image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        try:
            # Open the image file
            img = Image.open(file.stream)

            # Use pytesseract to do OCR on the image
            extracted_text = pytesseract.image_to_string(img)
            
            # Try to extract relevant nutritional information using regex
            calories = extract_value(extracted_text, ['Total Calories', 'Calories'])
            total_fat = extract_value(extracted_text, ['Total Fat', 'Fat'])
            protein = extract_value(extracted_text, ['Total Protein', 'Protein'])
            carbohydrate = extract_value(extracted_text, ['Total Carbohydrate', 'Carbohydrates'])

            # Create a dictionary to return the results
            result = {
                "Calories": calories if calories is not None else "Not found",
                "Total Fat": total_fat if total_fat is not None else "Not found",
                "Protein": protein if protein is not None else "Not found",
                "Carbohydrates": carbohydrate if carbohydrate is not None else "Not found"
            }

            # Perform Eat If prediction
            if calories and float(calories) < 300:
                result["Eat If Prediction"] = "Eat If... it fits within your daily intake."
            else:
                result["Eat If Prediction"] = "Eat If... you're ready for some extra energy!"
            
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)})

def extract_value(text, keywords):
    """Extracts a numeric value from the text based on provided keywords."""
    for keyword in keywords:
        match = re.search(rf'{keyword}[\s:]*([0-9]+)', text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

if __name__ == '__main__':
    app.config['SECRET_KEY'] = 'yoursecretkey'
    app.run(debug=True)
