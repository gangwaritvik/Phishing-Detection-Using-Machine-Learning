import os
import time
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#LOAD DATASET
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Dataset Shape:", df.shape)


#SEPARATE FEATURES & LABEL

y = df["label"]

drop_cols = ["label", "URL", "Domain", "FILENAME"]
X = df.drop(columns=drop_cols)

print("\nInitial feature shape:", X.shape)


#REMOVE NON-NUMERIC COLUMNS

non_numeric_cols = X.select_dtypes(include=["object"]).columns
print("\nNon-numeric columns detected:", list(non_numeric_cols))

X = X.drop(columns=non_numeric_cols)

print("Final feature shape after dropping non-numeric columns:", X.shape)


#TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

#TRAIN RANDOM FOREST MODEL

print("\nTraining Random Forest model...")

start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

end_time = time.time()

print("Training completed")
print(f"Training time: {end_time - start_time:.2f} seconds")

#SAVE MODEL

MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "phishing_rf_model.pkl")
joblib.dump(rf_model, MODEL_PATH)

print("\nModel saved successfully at:", MODEL_PATH)
