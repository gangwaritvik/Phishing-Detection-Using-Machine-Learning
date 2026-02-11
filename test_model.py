import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model
MODEL_PATH = os.path.join(BASE_DIR, "model", "phishing_rf_model.pkl")
model = joblib.load(MODEL_PATH)

# Load dataset
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
df = pd.read_csv(DATA_PATH)

# Prepare features
drop_cols = ["label", "URL", "Domain", "FILENAME"]
X = df.drop(columns=drop_cols)

non_numeric_cols = X.select_dtypes(include=["object"]).columns
X = X.drop(columns=non_numeric_cols)

# Random sample (changes every run)
sample_df = df.sample(5)
sample_X = X.loc[sample_df.index]
sample_urls = sample_df["URL"]
actual_labels = sample_df["label"]

# Predict
predictions = model.predict(sample_X)

print("\n--- RANDOM URL PREDICTIONS ---\n")

for url, actual, pred in zip(sample_urls, actual_labels, predictions):
    print("URL:", url)
    print("Actual Label   :", "Phishing" if actual == 1 else "Legitimate")
    print("Model Prediction:", "Phishing" if pred == 1 else "Legitimate")
    print("-" * 80)
