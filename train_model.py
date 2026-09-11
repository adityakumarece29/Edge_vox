import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# PATHS
# =========================================================

DATASET_DIR = r"C:\Users\adi\OneDrive\Documents\PS26172"

DATA_FILE = os.path.join(
    DATASET_DIR,
    "mfcc_dataset_correct.npz"
)

MODEL_FILE = os.path.join(
    DATASET_DIR,
    "kws_model_correct_final.pkl"
)

SCALER_FILE = os.path.join(
    DATASET_DIR,
    "kws_scaler_correct_final.pkl"
)
# =========================================================
# LOAD DATASET
# =========================================================

print("=" * 60)
print("             PS26172 ML MODEL TRAINING")
print("=" * 60)
print()

print("Loading dataset...")

data = np.load(DATA_FILE)

X = data["X"]
y = data["y"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================================================
# CHECK DATA
# =========================================================

print()
print("Class distribution:")

print("KEYWORD :", np.sum(y == 0))
print("UNKNOWN :", np.sum(y == 1))
print("SILENCE :", np.sum(y == 2))

# =========================================================
# FLATTEN MFCC
# =========================================================

# 97 x 13 = 1261 features

X_flat = X.reshape(
    X.shape[0],
    -1
)

print()
print("Flattened X shape:", X_flat.shape)

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_flat,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print()
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

# =========================================================
# FEATURE SCALING
# =========================================================

print()
print("Scaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# =========================================================
# MLP MODEL
# =========================================================

print()
print("Creating ML model...")

model = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20
)

# =========================================================
# TRAIN
# =========================================================

print()
print("Training model...")
print("Please wait...")

model.fit(
    X_train_scaled,
    y_train
)

print()
print("Training complete!")

# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(
    X_test_scaled
)

# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print("=" * 60)
print("                  RESULTS")
print("=" * 60)

print()
print(
    "Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print()
print("Classification Report:")
print()

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "KEYWORD",
            "UNKNOWN",
            "SILENCE"
        ],
        zero_division=0
    )
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("Confusion Matrix:")
print()

print("              Predicted")
print("             K    U    S")
print()
print(
    "Actual K   ",
    cm[0][0],
    cm[0][1],
    cm[0][2]
)

print(
    "Actual U   ",
    cm[1][0],
    cm[1][1],
    cm[1][2]
)

print(
    "Actual S   ",
    cm[2][0],
    cm[2][1],
    cm[2][2]
)

# =========================================================
# MODEL INFORMATION
# =========================================================

print()
print("=" * 60)
print("               MODEL INFORMATION")
print("=" * 60)

print()
print("Input features :", X_flat.shape[1])
print("Hidden neurons  :", 32)
print("Output classes  :", 3)
print("Iterations      :", model.n_iter_)

# =========================================================
# SAVE MODEL + SCALER
# =========================================================

import joblib

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    scaler,
    SCALER_FILE
)

print()
print("Model saved:")
print(MODEL_FILE)

print()
print("Scaler saved:")
print(SCALER_FILE)

# =========================================================
# FINAL
# =========================================================

print()
print("=" * 60)
print("              TRAINING COMPLETE")
print("=" * 60)

print()
print("Classes:")
print("0 = KEYWORD")
print("1 = UNKNOWN")
print("2 = SILENCE")

print()
print("STATUS: ML MODEL READY")
print("STATUS: SCALER READY")
print("STATUS: ESP32 DEPLOYMENT READY")

print()