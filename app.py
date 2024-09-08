from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import joblib
import pandas as pd
import os

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey')

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
    # Get previous entries from session if they exist
    previous_entries = session.get('entries', [])
    return render_template('index.html', previous_entries=previous_entries)

# Prediction route
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

    # Store entry in session
    if 'entries' not in session:
        session['entries'] = []
    
    session['entries'].append(input_data)
    session.modified = True  # Mark session as modified to ensure the new entry is saved
    
    return jsonify({'category': category, 'message': message})

# Route to clear the log of previous entries
@app.route('/clear_entries', methods=['POST'])
def clear_entries():
    session.pop('entries', None)  # Remove previous entries from session
    return jsonify({'message': 'Entries cleared'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the PORT environment variable provided by Heroku
    app.run(host='0.0.0.0', port=port)
