import pandas as pd
import os
import glob

# ============================================================
# SETTINGS
# ============================================================

DATA_FOLDER = "data"

EXPECTED_SAMPLES = 100

# ============================================================
# FIND CSV FILES
# ============================================================

csv_files = sorted(
    glob.glob(
        os.path.join(DATA_FOLDER, "*.csv")
    )
)

print("======================================")
print(" WAREHOUSE TinyML DATASET INSPECTOR")
print("======================================")
print()

print("Data folder:", DATA_FOLDER)
print("CSV files found:", len(csv_files))
print()

if len(csv_files) == 0:
    print("ERROR: No CSV files found.")
    print("Check that your CSV files are inside the data folder.")
    exit()

# ============================================================
# STORAGE
# ============================================================

class_counts = {}

valid_windows = []
invalid_windows = []

total_rows = 0

# ============================================================
# PROCESS EACH CSV
# ============================================================

for filepath in csv_files:

    filename = os.path.basename(filepath)

    print("--------------------------------------")
    print("File:", filename)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print("ERROR reading file:", e)
        continue

    required_columns = [
        "window_id",
        "timestamp_ms",
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz",
        "ir",
        "class_id",
        "class_name"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("ERROR: Missing columns:")
        print(missing_columns)
        continue

    total_rows += len(df)

    print("Rows:", len(df))

    # --------------------------------------------------------
    # CHECK EACH WINDOW
    # --------------------------------------------------------

    for window_id, window in df.groupby("window_id"):

        sample_count = len(window)

        class_names = window["class_name"].dropna().unique()

        if len(class_names) == 1:
            class_name = class_names[0]
        else:
            class_name = "UNKNOWN"

        # Check whether window has exactly 100 samples
        if sample_count == EXPECTED_SAMPLES and class_name != "UNKNOWN":

            valid_windows.append(
                (filename, window_id, class_name)
            )

            if class_name not in class_counts:
                class_counts[class_name] = 0

            class_counts[class_name] += 1

        else:

            invalid_windows.append(
                (
                    filename,
                    window_id,
                    sample_count,
                    class_name
                )
            )

# ============================================================
# SUMMARY
# ============================================================

print()
print("======================================")
print(" DATASET SUMMARY")
print("======================================")
print()

print("Total CSV files:", len(csv_files))
print("Total rows:", total_rows)
print()

print("Valid windows:")
print()

classes = [
    "NORMAL",
    "STATIONARY",
    "IMPACT",
    "ABNORMAL_VIBRATION"
]

for class_name in classes:

    count = class_counts.get(class_name, 0)

    print(
        f"{class_name:20s}: {count}"
    )

# ============================================================
# INVALID WINDOWS
# ============================================================

print()
print("======================================")
print(" INVALID WINDOWS")
print("======================================")
print()

if len(invalid_windows) == 0:

    print("No malformed windows found.")

else:

    for item in invalid_windows:

        filename, window_id, sample_count, class_name = item

        print(
            f"{filename} | "
            f"Window {window_id} | "
            f"Rows: {sample_count} | "
            f"Class: {class_name}"
        )

# ============================================================
# TOTAL
# ============================================================

print()
print("======================================")
print(" TOTAL VALID WINDOWS")
print("======================================")
print()

print(
    "Total:",
    len(valid_windows)
)

print()
print("Inspection complete.")