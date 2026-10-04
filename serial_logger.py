
import serial
import csv
import os
import time

# ============================================================
# SETTINGS
# ============================================================

PORT = "COM3"
BAUD_RATE = 115200

OUTPUT_FOLDER = "data"

# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

filename = time.strftime("raw_dataset_%Y%m%d_%H%M%S.csv")
filepath = os.path.join(OUTPUT_FOLDER, filename)

# ============================================================
# START
# ============================================================

print("======================================")
print(" WAREHOUSE TinyML DATA LOGGER")
print("======================================")
print()
print("Port:", PORT)
print("Baud:", BAUD_RATE)
print("Output:", filepath)
print()

print("Connecting to ESP32...")

ser = serial.Serial(
    PORT,
    BAUD_RATE,
    timeout=1
)

# ESP32 resets when serial port opens
time.sleep(2)

print("Connected!")
print()

# ============================================================
# OPEN CSV FILE
# ============================================================

csv_file = open(
    filepath,
    "w",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(csv_file)

writer.writerow([
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
])

csv_file.flush()

window_id = 0

# ============================================================
# READ ONE RECORDING
# ============================================================

def record_window(class_choice):

    global window_id

    print()
    print("--------------------------------------")
    print("Starting recording...")
    print("--------------------------------------")

    # Send class number to ESP32
    ser.write(class_choice.encode())

    recording = False
    rows_this_window = 0

    window_id += 1

    while True:

        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if not line:
            continue

        print(line)

        # ESP32 has started sending CSV data
        if line.startswith(
            "timestamp_ms,ax,ay,az,gx,gy,gz,ir,class_id,class_name"
        ):
            recording = True
            continue

        # End of one recording
        if line == "DATA_END":

            csv_file.flush()

            print()
            print("--------------------------------------")
            print("WINDOW SAVED")
            print("Window ID:", window_id)
            print("Rows saved:", rows_this_window)
            print("--------------------------------------")

            return

        # Save sensor data
        if recording:

            parts = line.split(",")

            if len(parts) == 10:

                try:
                    timestamp_ms = int(parts[0])

                    ax = float(parts[1])
                    ay = float(parts[2])
                    az = float(parts[3])

                    gx = float(parts[4])
                    gy = float(parts[5])
                    gz = float(parts[6])

                    ir = int(parts[7])
                    class_id = int(parts[8])
                    class_name = parts[9]

                    writer.writerow([
                        window_id,
                        timestamp_ms,
                        ax,
                        ay,
                        az,
                        gx,
                        gy,
                        gz,
                        ir,
                        class_id,
                        class_name
                    ])

                    rows_this_window += 1

                except ValueError:
                    pass


# ============================================================
# MAIN MENU
# ============================================================

try:

    while True:

        print()
        print("======================================")
        print(" SELECT DATASET CLASS")
        print("======================================")
        print("1 = NORMAL")
        print("2 = STATIONARY")
        print("3 = IMPACT")
        print("4 = ABNORMAL_VIBRATION")
        print("Q = QUIT")
        print()

        choice = input("Enter class number: ").strip().upper()

        if choice == "Q":
            break

        if choice not in ["1", "2", "3", "4"]:
            print("Invalid choice. Enter 1, 2, 3, 4 or Q.")
            continue

        record_window(choice)

except KeyboardInterrupt:

    print()
    print("Stopping logger...")

finally:

    csv_file.flush()
    csv_file.close()

    ser.close()

    print()
    print("======================================")
    print(" LOGGER STOPPED")
    print("======================================")
    print("Dataset saved to:")
    print(filepath)
    print()
