import numpy as np
import joblib

from sklearn.metrics import confusion_matrix

# Load dataset
data = np.load("mfcc_dataset.npz")
X = data["X"]
y = data["y"]

# Flatten MFCC
X = X.reshape(X.shape[0], -1)

# Load trained model and scaler
model = joblib.load("kws_model.pkl")
scaler = joblib.load("kws_scaler.pkl")

# Normalize
X_scaled = scaler.transform(X)

# Predictions
pred = model.predict(X_scaled)
prob = model.predict_proba(X_scaled)

classes = ["KEYWORD", "UNKNOWN", "SILENCE"]

print("================================")
print("        PS26172 MODEL TEST")
print("================================")

# Show every sample
for i in range(len(X)):
    actual = classes[y[i]]
    predicted = classes[pred[i]]
    confidence = np.max(prob[i]) * 100

    status = "✓" if y[i] == pred[i] else "✗"

    print(
        f"{status} Sample {i+1:02d} | "
        f"Actual: {actual:7s} | "
        f"Predicted: {predicted:7s} | "
        f"Confidence: {confidence:.1f}%"
    )

# Confusion matrix
cm = confusion_matrix(y, pred)

print("\n================================")
print("       CONFUSION MATRIX")
print("================================")

print("              Predicted")
print("             K   U   S")
print(f"Actual K   {cm[0,0]:2d}  {cm[0,1]:2d}  {cm[0,2]:2d}")
print(f"       U   {cm[1,0]:2d}  {cm[1,1]:2d}  {cm[1,2]:2d}")
print(f"       S   {cm[2,0]:2d}  {cm[2,1]:2d}  {cm[2,2]:2d}")

accuracy = np.mean(pred == y) * 100

print("\nOverall accuracy on full dataset:", f"{accuracy:.2f}%")
print("\nSTATUS: MODEL TEST COMPLETE")