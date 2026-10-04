import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/clean_dataset.csv"
OUTPUT_FILE = "data/ml_dataset.npz"

ML_CLASSES = [
    "NORMAL",
    "STATIONARY",
    "ABNORMAL_VIBRATION"
]

df = pd.read_csv(INPUT_FILE)

print("======================================")
print(" PREPARING ML DATASET")
print("======================================")
print()

X = []
y = []

label_map = {
    "NORMAL": 0,
    "STATIONARY": 1,
    "ABNORMAL_VIBRATION": 2
}

for class_name in ML_CLASSES:

    class_df = df[
        df["class_name"] == class_name
    ]

    for window_id, window in class_df.groupby("window_id"):

        if len(window) != 100:
            continue

        samples = window[
            ["ax", "ay", "az", "gx", "gy", "gz"]
        ].values

        X.append(samples)
        y.append(label_map[class_name])

        print(
            f"Added: {class_name:18s} "
            f"Window {int(window_id):02d}"
        )

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print()
print("--------------------------------------")
print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)
print("--------------------------------------")

np.savez(
    OUTPUT_FILE,
    X=X,
    y=y
)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("Class counts:")

for class_name, label in label_map.items():
    print(
        f"{class_name:18s}: "
        f"{np.sum(y == label)} windows"
    )

print()
print("======================================")
