import pandas as pd
import glob
import os

DATA_FOLDER = "data"
OUTPUT_FILE = os.path.join(DATA_FOLDER, "clean_dataset.csv")

KEEP_WINDOWS = {
    ("raw_dataset_20261004_185708.csv", 1),   # NORMAL
    ("raw_dataset_20261004_185708.csv", 5),   # STATIONARY
    ("raw_dataset_20261004_185708.csv", 7),   # NORMAL
    ("raw_dataset_20261004_185708.csv", 8),   # STATIONARY
    ("raw_dataset_20261004_185708.csv", 10),  # VIBRATION
    ("raw_dataset_20261004_185708.csv", 11),  # NORMAL
    ("raw_dataset_20261004_185708.csv", 12),  # STATIONARY
    ("raw_dataset_20261004_185708.csv", 14),  # VIBRATION
    ("raw_dataset_20261004_185708.csv", 15),  # NORMAL
    ("raw_dataset_20261004_185708.csv", 16),  # STATIONARY
    ("raw_dataset_20261004_185708.csv", 18),  # VIBRATION
    ("raw_dataset_20261004_185708.csv", 20),  # IMPACT
    ("raw_dataset_20261004_185528.csv", 2),   # VIBRATION
    ("raw_dataset_20261004_185708.csv", 4),   # VIBRATION
}

files = glob.glob(
    os.path.join(DATA_FOLDER, "*.csv")
)

clean_parts = []

for filepath in files:

    filename = os.path.basename(filepath)

    try:
        df = pd.read_csv(filepath)
    except Exception:
        continue

    if df.empty:
        continue

    for window_id, window in df.groupby("window_id"):

        key = (filename, int(window_id))

        if key not in KEEP_WINDOWS:
            continue

        clean_parts.append(window)


if not clean_parts:
    print("ERROR: No windows selected.")
    exit()


clean_df = pd.concat(
    clean_parts,
    ignore_index=True
)

# Keep only the six MPU6050 channels and class information.
clean_df = clean_df[
    [
        "window_id",
        "timestamp_ms",
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz",
        "class_id",
        "class_name"
    ]
]

clean_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("======================================")
print(" CLEAN DATASET CREATED")
print("======================================")
print()

print("Output:")
print(OUTPUT_FILE)

print()
print("Rows:", len(clean_df))
print("Windows:", len(clean_df) // 100)

print()
print("Class counts:")

print(
    clean_df["class_name"]
    .value_counts()
)

print()
print("======================================")