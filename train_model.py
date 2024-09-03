from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib

# Load the training dataset
train_df = pd.read_csv('train_df.csv')

# Define the features to be used in the model
numeric_features = ['Calories', 'Protein', 'Carbohydrate', 'Total fat']

# Separate features and target
X = train_df[numeric_features]
y = train_df['Health Category']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the training dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Preprocess the data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Label encode the target variable
label_encoder = LabelEncoder()
y_resampled_encoded = label_encoder.fit_transform(y_resampled)

# Define the model pipeline with fewer trees to speed up training
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))  # Reduced number of trees
])

# Train the model on the resampled data
model.fit(X_resampled, y_resampled_encoded)

# Save the model, label encoder, and the test set
joblib.dump(model, 'eat_if_model_smote.pkl')
joblib.dump(label_encoder, 'label_encoder_smote.pkl')
joblib.dump((X_test, y_test), 'test_data.pkl')  # Save the test data for later use

print("Model, label encoder, and test data saved successfully.")
