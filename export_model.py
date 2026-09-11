import os
import numpy as np
import joblib

# ============================================================
# PS26172 - EXPORT TRAINED MODEL TO ESP32 HEADER
# 400-SAMPLE FINAL MODEL
# ============================================================

DATASET_DIR = r"C:\Users\adi\OneDrive\Documents\PS26172"

MODEL_FILE = os.path.join(
    DATASET_DIR,
    "kws_model_correct_final.pkl"
)

SCALER_FILE = os.path.join(
    DATASET_DIR,
    "kws_scaler_correct_final.pkl"
)

HEADER_FILE = os.path.join(
    DATASET_DIR,
    "kws_model_correct_final.h"
)

print("=" * 60)
print("        PS26172 MODEL EXPORT FOR ESP32")
print("             400-SAMPLE FINAL MODEL")
print("=" * 60)

# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading trained model...")

if not os.path.exists(MODEL_FILE):
    print("ERROR: Model file not found!")
    print(MODEL_FILE)
    raise SystemExit

if not os.path.exists(SCALER_FILE):
    print("ERROR: Scaler file not found!")
    print(SCALER_FILE)
    raise SystemExit

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

print("Model loaded successfully!")
print("Scaler loaded successfully!")

# ============================================================
# GET MODEL PARAMETERS
# ============================================================

W1 = model.coefs_[0]
b1 = model.intercepts_[0]

W2 = model.coefs_[1]
b2 = model.intercepts_[1]

mean = scaler.mean_
scale = scaler.scale_

print()
print("Model shapes:")
print("W1    :", W1.shape)
print("b1    :", b1.shape)
print("W2    :", W2.shape)
print("b2    :", b2.shape)
print("Mean  :", mean.shape)
print("Scale :", scale.shape)

# ============================================================
# SAFETY CHECK
# ============================================================

if W1.shape != (1261, 32):
    raise ValueError(
        f"Unexpected W1 shape: {W1.shape}"
    )

if b1.shape != (32,):
    raise ValueError(
        f"Unexpected b1 shape: {b1.shape}"
    )

if W2.shape != (32, 3):
    raise ValueError(
        f"Unexpected W2 shape: {W2.shape}"
    )

if b2.shape != (3,):
    raise ValueError(
        f"Unexpected b2 shape: {b2.shape}"
    )

if mean.shape != (1261,):
    raise ValueError(
        f"Unexpected scaler mean shape: {mean.shape}"
    )

if scale.shape != (1261,):
    raise ValueError(
        f"Unexpected scaler scale shape: {scale.shape}"
    )

print()
print("Model dimension check: PASSED")

# ============================================================
# CONVERT NUMPY ARRAY TO C++ FLOAT ARRAY
# ============================================================

def array_to_cpp(name, array):

    array = np.asarray(
        array,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # 1D ARRAY
    # --------------------------------------------------------

    if array.ndim == 1:

        values = ", ".join(
            f"{x:.9g}f"
            for x in array
        )

        return (
            f"const float {name}[{array.shape[0]}] = {{\n"
            f"    {values}\n"
            f"}};\n\n"
        )

    # --------------------------------------------------------
    # 2D ARRAY
    # --------------------------------------------------------

    elif array.ndim == 2:

        rows = []

        for row in array:

            values = ", ".join(
                f"{x:.9g}f"
                for x in row
            )

            rows.append(
                "    {" + values + "}"
            )

        return (
            f"const float {name}"
            f"[{array.shape[0]}]"
            f"[{array.shape[1]}] = {{\n"
            + ",\n".join(rows)
            + "\n};\n\n"
        )

    else:

        raise ValueError(
            "Unsupported array dimension"
        )

# ============================================================
# CREATE ESP32 HEADER
# ============================================================

print()
print("Creating ESP32 header...")

with open(
    HEADER_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "// =====================================================\n"
    )

    f.write(
        "// PS26172 - KWS MODEL FOR ESP32\n"
    )

    f.write(
        "// 400-SAMPLE FINAL MODEL\n"
    )

    f.write(
        "// Auto-generated from sklearn MLPClassifier\n"
    )

    f.write(
        "// =====================================================\n\n"
    )

    # --------------------------------------------------------
    # HEADER GUARD
    # --------------------------------------------------------

    f.write(
        "#ifndef KWS_MODEL_CORRECT_FINAL_H\n"
    )

    f.write(
        "#define KWS_MODEL_CORRECT_FINAL_H\n\n"
    )

    f.write(
        "#include <Arduino.h>\n\n"
    )

    # --------------------------------------------------------
    # MODEL DIMENSIONS
    # --------------------------------------------------------

    f.write(
        "// Model dimensions\n"
    )

    f.write(
        "const int INPUT_SIZE = 1261;\n"
    )

    f.write(
        "const int HIDDEN_SIZE = 32;\n"
    )

    f.write(
        "const int OUTPUT_SIZE = 3;\n\n"
    )

    # --------------------------------------------------------
    # CLASS LABELS
    # --------------------------------------------------------

    f.write(
        "// Class labels\n"
    )

    f.write(
        "// 0 = KEYWORD\n"
    )

    f.write(
        "// 1 = UNKNOWN\n"
    )

    f.write(
        "// 2 = SILENCE\n\n"
    )

    # --------------------------------------------------------
    # FIRST LAYER
    # --------------------------------------------------------

    f.write(
        "// First layer weights\n"
    )

    f.write(
        array_to_cpp(
            "W1",
            W1
        )
    )

    f.write(
        "// First layer bias\n"
    )

    f.write(
        array_to_cpp(
            "b1",
            b1
        )
    )

    # --------------------------------------------------------
    # SECOND LAYER
    # --------------------------------------------------------

    f.write(
        "// Second layer weights\n"
    )

    f.write(
        array_to_cpp(
            "W2",
            W2
        )
    )

    f.write(
        "// Second layer bias\n"
    )

    f.write(
        array_to_cpp(
            "b2",
            b2
        )
    )

    # --------------------------------------------------------
    # STANDARD SCALER
    # --------------------------------------------------------

    f.write(
        "// StandardScaler mean\n"
    )

    f.write(
        array_to_cpp(
            "SCALER_MEAN",
            mean
        )
    )

    f.write(
        "// StandardScaler scale\n"
    )

    f.write(
        array_to_cpp(
            "SCALER_SCALE",
            scale
        )
    )

    # --------------------------------------------------------
    # END HEADER
    # --------------------------------------------------------

    f.write(
        "#endif\n"
    )

# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 60)
print("              EXPORT COMPLETE")
print("=" * 60)

print()
print("Created:")
print(HEADER_FILE)

print()
print("Model:")
print("Input  :", 1261)
print("Hidden :", 32)
print("Output :", 3)

print()
print("Classes:")
print("0 = KEYWORD")
print("1 = UNKNOWN")
print("2 = SILENCE")

print()
print("STATUS: ESP32 MODEL HEADER READY")
print("=" * 60)