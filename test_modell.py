import joblib
from sklearn.metrics import classification_report, accuracy_score

# Load the model, label encoder, and test data
model = joblib.load('eat_if_model_smote.pkl')
label_encoder = joblib.load('label_encoder_smote.pkl')
X_test, y_test = joblib.load('test_data.pkl')

# Encode the test labels
y_test_encoded = label_encoder.transform(y_test)

# Make predictions on the test data
y_pred_encoded = model.predict(X_test)

# Decode the predicted labels
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
