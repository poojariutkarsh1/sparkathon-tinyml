import tensorflow as tf
import numpy as np
import os

MODEL_FILE = "data/warehouse_cnn.keras"
DATA_FILE = "data/training_dataset.npz"
OUTPUT_FILE = "data/warehouse_cnn_int8.tflite"

print("======================================")
print(" TINYML MODEL CONVERSION")
print("======================================")
print()

# --------------------------------------------------
# Load trained Keras model
# --------------------------------------------------

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_FILE)

print("Model loaded.")
print()

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

data = np.load(DATA_FILE, allow_pickle=True)

X = data["X"]
sources = data["window_sources"]

print("Dataset loaded.")
print("X shape:", X.shape)
print()

# --------------------------------------------------
# Load normalization values
# --------------------------------------------------

norm = np.load("data/normalization.npz")

mean = norm["mean"]
std = norm["std"]

print("Normalization values loaded.")
print()

# --------------------------------------------------
# Normalize dataset
# --------------------------------------------------

X_normalized = (X - mean) / std

# --------------------------------------------------
# Representative dataset
#
# Use only the training-side sessions.
# The held-out test session is:
#
# raw_dataset_20261004_191314.csv
# --------------------------------------------------

representative_indices = []

for i, source in enumerate(sources):
    source = str(source)

    if "raw_dataset_20261004_191314.csv" not in source:
        representative_indices.append(i)

print("Representative samples:", len(representative_indices))
print()

def representative_dataset():
    for i in representative_indices:
        sample = X_normalized[i:i+1].astype(np.float32)
        yield [sample]

# --------------------------------------------------
# Convert to TensorFlow Lite
# --------------------------------------------------

print("Starting INT8 quantization...")
print()

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# --------------------------------------------------
# Save model
# --------------------------------------------------

with open(OUTPUT_FILE, "wb") as f:
    f.write(tflite_model)

print()
print("======================================")
print(" CONVERSION COMPLETE")
print("======================================")
print()

print("Output file:")
print(OUTPUT_FILE)

print()

print("Model size:")
print(f"{len(tflite_model) / 1024:.2f} KB")

print()
print("Input type:  INT8")
print("Output type: INT8")

print()
print("======================================")