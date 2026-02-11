import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model
MODEL_PATH = os.path.join(BASE_DIR, "model", "phishing_rf_model.pkl")
model = joblib.load(MODEL_PATH)

print("Model loaded successfully")

# Load dataset
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
df = pd.read_csv(DATA_PATH)

# Prepare features (same as training)
drop_cols = ["label", "URL", "Domain", "FILENAME"]
X = df.drop(columns=drop_cols)
y = df["label"]

non_numeric_cols = X.select_dtypes(include=["object"]).columns
X = X.drop(columns=non_numeric_cols)

# Train-test split (same logic as training)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Predict
y_pred = model.predict(X_test)

# Evaluation metrics
print("\n--- MODEL EVALUATION ---\n")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
