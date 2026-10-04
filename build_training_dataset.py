import pandas as pd
import numpy as np
import glob
import os


DATA_FOLDER = "data"
OUTPUT_FILE = os.path.join(
    DATA_FOLDER,
    "training_dataset.npz"
)


# --------------------------------------------------
# APPROVED WINDOWS
# --------------------------------------------------
#
# Format:
# ("CSV filename", window number)
#
# IMPACT windows are deliberately excluded.
#

APPROVED_WINDOWS = {

    # ----------------------------------------------
    # Original approved windows
    # ----------------------------------------------

    ("raw_dataset_20261004_185708.csv", 1),
    ("raw_dataset_20261004_185708.csv", 5),
    ("raw_dataset_20261004_185708.csv", 7),
    ("raw_dataset_20261004_185708.csv", 8),
    ("raw_dataset_20261004_185708.csv", 10),
    ("raw_dataset_20261004_185708.csv", 11),
    ("raw_dataset_20261004_185708.csv", 12),
    ("raw_dataset_20261004_185708.csv", 14),
    ("raw_dataset_20261004_185708.csv", 15),
    ("raw_dataset_20261004_185708.csv", 16),
    ("raw_dataset_20261004_185708.csv", 18),

    ("raw_dataset_20261004_185528.csv", 2),

    # ----------------------------------------------
    # New 191314 recording
    # ----------------------------------------------

    ("raw_dataset_20261004_191314.csv", 1),
    ("raw_dataset_20261004_191314.csv", 2),
    ("raw_dataset_20261004_191314.csv", 3),
    ("raw_dataset_20261004_191314.csv", 4),
    ("raw_dataset_20261004_191314.csv", 5),
    ("raw_dataset_20261004_191314.csv", 6),

    # ----------------------------------------------
    # New 191551 recording
    # W07 rejected
    # ----------------------------------------------

    ("raw_dataset_20261004_191551.csv", 1),
    ("raw_dataset_20261004_191551.csv", 2),
    ("raw_dataset_20261004_191551.csv", 3),
    ("raw_dataset_20261004_191551.csv", 4),
    ("raw_dataset_20261004_191551.csv", 5),
    ("raw_dataset_20261004_191551.csv", 6),

    ("raw_dataset_20261004_191551.csv", 8),
    ("raw_dataset_20261004_191551.csv", 9),
    ("raw_dataset_20261004_191551.csv", 10),
    ("raw_dataset_20261004_191551.csv", 11),
    ("raw_dataset_20261004_191551.csv", 12),
    ("raw_dataset_20261004_191551.csv", 13),
    ("raw_dataset_20261004_191551.csv", 14),
    ("raw_dataset_20261004_191551.csv", 15),
    ("raw_dataset_20261004_191551.csv", 16),
    ("raw_dataset_20261004_191551.csv", 17),
    ("raw_dataset_20261004_191551.csv", 18),
    ("raw_dataset_20261004_191551.csv", 19),
    ("raw_dataset_20261004_191551.csv", 20),
    ("raw_dataset_20261004_191551.csv", 21),
    ("raw_dataset_20261004_191551.csv", 22),
    ("raw_dataset_20261004_191551.csv", 23),
    ("raw_dataset_20261004_191551.csv", 24),
    ("raw_dataset_20261004_191551.csv", 25),
    ("raw_dataset_20261004_191551.csv", 26),
    ("raw_dataset_20261004_191551.csv", 27),
    ("raw_dataset_20261004_191551.csv", 28),
    ("raw_dataset_20261004_191551.csv", 29),
    ("raw_dataset_20261004_191551.csv", 30),
}


# --------------------------------------------------
# CLASS LABELS
# --------------------------------------------------

LABEL_MAP = {
    "NORMAL": 0,
    "STATIONARY": 1,
    "ABNORMAL_VIBRATION": 2
}


# --------------------------------------------------
# STORAGE
# --------------------------------------------------

X = []
y = []

window_sources = []


# --------------------------------------------------
# READ RAW FILES
# --------------------------------------------------

files = glob.glob(
    os.path.join(
        DATA_FOLDER,
        "raw_dataset_*.csv"
    )
)

print("======================================")
print(" BUILDING TRAINING DATASET")
print("======================================")
print()

for filepath in files:

    filename = os.path.basename(filepath)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print("Could not read:", filename)
        continue

    if df.empty:
        continue

    # IMPORTANT:
    # Group by window ID inside THIS file only.
    for window_id, window in df.groupby("window_id"):

        key = (
            filename,
            int(window_id)
        )

        if key not in APPROVED_WINDOWS:
            continue

        # Only the three ML classes
        class_name = window["class_name"].iloc[0]

        if class_name not in LABEL_MAP:
            continue

        # Every window must contain exactly 100 samples
        if len(window) != 100:
            print(
                "Skipping incomplete window:",
                filename,
                "Window",
                window_id
            )
            continue

        # ------------------------------------------
        # Six MPU6050 channels
        # ------------------------------------------

        samples = window[
            [
                "ax",
                "ay",
                "az",
                "gx",
                "gy",
                "gz"
            ]
        ].values

        samples = samples.astype(
            np.float32
        )

        X.append(samples)

        y.append(
            LABEL_MAP[class_name]
        )

        window_sources.append(
            f"{filename}:W{int(window_id):02d}"
        )

        print(
            f"Added: "
            f"{filename:35s} "
            f"W{int(window_id):02d} "
            f"{class_name}"
        )


# --------------------------------------------------
# CONVERT TO NUMPY
# --------------------------------------------------

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int64
)


# --------------------------------------------------
# CHECK DATASET
# --------------------------------------------------

print()
print("--------------------------------------")

print(
    "Dataset shape:",
    X.shape
)

print(
    "Labels shape:",
    y.shape
)

print(
    "Total windows:",
    len(X)
)

print("--------------------------------------")


if len(X) == 0:
    print()
    print("ERROR: No training windows found.")
    exit()


# --------------------------------------------------
# CLASS COUNTS
# --------------------------------------------------

print()
print("Class counts:")

for class_name, label in LABEL_MAP.items():

    count = np.sum(
        y == label
    )

    print(
        f"{class_name:20s}: {count}"
    )


# --------------------------------------------------
# SAVE
# --------------------------------------------------

np.savez(
    OUTPUT_FILE,
    X=X,
    y=y,
    window_sources=np.array(
        window_sources
    )
)


print()
print("======================================")
print(" TRAINING DATASET SAVED")
print("======================================")
print()

print(
    "Output:",
    OUTPUT_FILE
)

print()
print(
    "Input shape:",
    X.shape[1:]
)

print(
    "Number of classes:",
    len(LABEL_MAP)
)

print()
print("======================================")