import os

INPUT_FILE = "data/warehouse_cnn_int8.tflite"
OUTPUT_FILE = "model/warehouse_model.h"

ARRAY_NAME = "warehouse_cnn_int8_model"

print("======================================")
print(" TFLITE → C HEADER CONVERSION")
print("======================================")
print()

print("Input:")
print(INPUT_FILE)

print("Output:")
print(OUTPUT_FILE)

print()

if not os.path.exists(INPUT_FILE):
    print("ERROR: TFLite model not found!")
    print(INPUT_FILE)
    exit()

with open(INPUT_FILE, "rb") as f:
    model_data = f.read()

with open(OUTPUT_FILE, "w") as f:

    f.write("#ifndef WAREHOUSE_MODEL_H\n")
    f.write("#define WAREHOUSE_MODEL_H\n\n")

    f.write("#include <stdint.h>\n\n")

    f.write(
        f"const unsigned char {ARRAY_NAME}[] = {{\n"
    )

    for i, byte in enumerate(model_data):

        if i % 12 == 0:
            f.write("  ")

        f.write(f"0x{byte:02x}")

        if i != len(model_data) - 1:
            f.write(", ")

        if (i + 1) % 12 == 0:
            f.write("\n")

    f.write("\n};\n\n")

    f.write(
        f"const unsigned int {ARRAY_NAME}_len = "
        f"{len(model_data)};\n\n"
    )

    f.write("#endif\n")

print("======================================")
print(" CONVERSION COMPLETE")
print("======================================")
print()

print("Model bytes:", len(model_data))
print("Header file:", OUTPUT_FILE)

print()
print("======================================")