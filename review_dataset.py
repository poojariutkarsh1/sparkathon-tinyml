import pandas as pd
import os
import glob
import numpy as np

DATA_FOLDER = "data"

files = glob.glob(
    os.path.join(DATA_FOLDER, "*.csv")
)

print("======================================")
print(" WAREHOUSE TinyML DATASET REVIEW")
print("======================================")
print()

all_windows = []

for filepath in files:

    try:
        df = pd.read_csv(filepath)
    except Exception:
        continue

    if df.empty:
        continue

    required_columns = [
        "window_id",
        "ax", "ay", "az",
        "gx", "gy", "gz",
        "class_name"
    ]

    if not all(col in df.columns for col in required_columns):
        continue

    filename = os.path.basename(filepath)

    for window_id, window in df.groupby("window_id"):

        if len(window) != 100:
            continue

        accel = np.sqrt(
            window["ax"]**2 +
            window["ay"]**2 +
            window["az"]**2
        )

        gyro = np.sqrt(
            window["gx"]**2 +
            window["gy"]**2 +
            window["gz"]**2
        )

        all_windows.append({
            "file": filename,
            "window_id": int(window_id),
            "class": window["class_name"].iloc[0],

            "accel_mean": accel.mean(),
            "accel_std": accel.std(),
            "accel_range": accel.max() - accel.min(),
            "accel_peak": accel.max(),

            "gyro_mean": gyro.mean(),
            "gyro_std": gyro.std(),
            "gyro_peak": gyro.max()
        })


print("WINDOW SUMMARY")
print("--------------------------------------")

for i, w in enumerate(all_windows, start=1):

    print(
        f"{i:02d} | "
        f"{w['file']} | "
        f"Window {w['window_id']:02d} | "
        f"{w['class']:18s} | "
        f"Accel STD: {w['accel_std']:.2f} | "
        f"Accel Range: {w['accel_range']:.2f} | "
        f"Gyro Peak: {w['gyro_peak']:.2f}"
    )


print()
print("======================================")
print("TOTAL WINDOWS:", len(all_windows))
print("======================================")