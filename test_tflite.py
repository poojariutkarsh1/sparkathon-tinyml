import tensorflow as tf
import numpy as np

MODEL_FILE = "data/warehouse_cnn_int8.tflite"
DATA_FILE = "data/training_dataset.npz"
NORM_FILE = "data/normalization.npz"

CLASS_NAMES = [
    "NORMAL",
    "STATIONARY",
    "ABNORMAL_VIBRATION"
]

TEST_SESSION = "raw_dataset_20261004_191314.csv"

print("======================================")
print(" INT8 TFLITE MODEL TEST")
print("======================================")
print()

# --------------------------------------------------
# Load TFLite model
# --------------------------------------------------

print("Loading INT8 TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=MODEL_FILE
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded.")
print()

print("Input:")
print("  Shape:", input_details[0]["shape"])
print("  Type:", input_details[0]["dtype"])
print("  Quantization:", input_details[0]["quantization"])

print()

print("Output:")
print("  Shape:", output_details[0]["shape"])
print("  Type:", output_details[0]["dtype"])
print("  Quantization:", output_details[0]["quantization"])

print()

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

data = np.load(
    DATA_FILE,
    allow_pickle=True
)

X = data["X"]
y = data["y"]
sources = data["window_sources"]

# --------------------------------------------------
# Load normalization values
# --------------------------------------------------

norm = np.load(NORM_FILE)

mean = norm["mean"]
std = norm["std"]

# --------------------------------------------------
# Select held-out test session
# --------------------------------------------------

test_indices = []

for i, source in enumerate(sources):

    source = str(source)

    if TEST_SESSION in source:
        test_indices.append(i)

print("Held-out test windows:", len(test_indices))
print()

# --------------------------------------------------
# Run inference
# --------------------------------------------------

correct = 0

print("======================================")
print(" TEST PREDICTIONS")
print("======================================")
print()

for i in test_indices:

    sample = X[i:i+1].astype(np.float32)

    # Same normalization used during training
    sample = (sample - mean) / std

    # ----------------------------------------------
    # Quantize FLOAT → INT8
    # ----------------------------------------------

    input_scale, input_zero_point = \
        input_details[0]["quantization"]

    input_data = np.round(
        sample / input_scale
    ) + input_zero_point

    input_data = np.clip(
        input_data,
        -128,
        127
    ).astype(np.int8)

    # ----------------------------------------------
    # Run model
    # ----------------------------------------------

    interpreter.set_tensor(
        input_details[0]["index"],
        input_data
    )

    interpreter.invoke()

    output_data = interpreter.get_tensor(
        output_details[0]["index"]
    )

    # ----------------------------------------------
    # Dequantize INT8 → FLOAT
    # ----------------------------------------------

    output_scale, output_zero_point = \
        output_details[0]["quantization"]

    probabilities = (
        output_data.astype(np.float32)
        - output_zero_point
    ) * output_scale

    probabilities = probabilities[0]

    predicted_class = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_class]
    )

    actual_class = int(y[i])

    if predicted_class == actual_class:
        correct += 1

    print(
        f"{str(sources[i]):45s} | "
        f"Actual: {CLASS_NAMES[actual_class]:20s} | "
        f"Predicted: {CLASS_NAMES[predicted_class]:20s} | "
        f"Confidence: {confidence * 100:.2f}%"
    )

# --------------------------------------------------
# Final result
# --------------------------------------------------

accuracy = (
    correct / len(test_indices)
) * 100

print()
print("======================================")
print(" INT8 TEST RESULT")
print("======================================")

print()
print(
    f"Correct: {correct}/{len(test_indices)}"
)

print(
    f"Accuracy: {accuracy:.2f}%"
)

print()
print("======================================")
