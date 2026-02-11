import os
import joblib
import pandas as pd

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model
MODEL_PATH = os.path.join(BASE_DIR, "model", "phishing_rf_model.pkl")
model = joblib.load(MODEL_PATH)

print("Model loaded successfully")

# Load dataset (only to get feature names)
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
df = pd.read_csv(DATA_PATH)

# Prepare features exactly like training
drop_cols = ["label", "URL", "Domain", "FILENAME"]
X = df.drop(columns=drop_cols)

non_numeric_cols = X.select_dtypes(include=["object"]).columns
X = X.drop(columns=non_numeric_cols)

# Extract feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

# Display results
print("\n--- FEATURE IMPORTANCE (Top 15) ---\n")
for feature, score in importance.head(15).items():
    print(f"{feature:30s} : {score:.5f}")
