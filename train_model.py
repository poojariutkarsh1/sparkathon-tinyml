import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

DATA_FILE = "data/training_dataset.npz"
MODEL_FILE = "data/warehouse_cnn.keras"

CLASS_NAMES = [
    "NORMAL",
    "STATIONARY",
    "ABNORMAL_VIBRATION"
]

print("======================================")
print(" WAREHOUSE TinyML MODEL TRAINING")
print("======================================")
print()

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

data = np.load(DATA_FILE, allow_pickle=True)

X = data["X"]
y = data["y"]
sources = data["window_sources"]

print("Dataset loaded.")
print("X shape:", X.shape)
print("y shape:", y.shape)
print()

# --------------------------------------------------
# 2. Class distribution
# --------------------------------------------------

print("Class distribution:")

for class_id, class_name in enumerate(CLASS_NAMES):
    count = np.sum(y == class_id)
    print(f"{class_name:20s}: {count}")

print()

# --------------------------------------------------
# 3. Hold out one complete recording session
# --------------------------------------------------

TEST_SESSION = "raw_dataset_20261004_191314.csv"

test_mask = np.array([
    TEST_SESSION in str(source)
    for source in sources
])

X_test = X[test_mask]
y_test = y[test_mask]

X_remaining = X[~test_mask]
y_remaining = y[~test_mask]

print("Session-based split:")
print("--------------------------------------")
print("Training/validation windows:", len(X_remaining))
print("Held-out test windows:", len(X_test))
print()

# --------------------------------------------------
# 4. Train / validation split
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.20,
    random_state=42,
    stratify=y_remaining
)

print("Final split:")
print("--------------------------------------")
print("Training windows:", len(X_train))
print("Validation windows:", len(X_val))
print("Test windows:", len(X_test))
print()

# --------------------------------------------------
# 5. Calculate normalization from TRAINING ONLY
# --------------------------------------------------

train_flat = X_train.reshape(
    -1,
    X_train.shape[-1]
)

mean = train_flat.mean(axis=0)
std = train_flat.std(axis=0)

std[std < 1e-6] = 1.0

channels = [
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz"
]

print("Normalization values:")
print("--------------------------------------")

for i, channel in enumerate(channels):
    print(
        f"{channel}: "
        f"mean={mean[i]:.6f}, "
        f"std={std[i]:.6f}"
    )

print()

# --------------------------------------------------
# 6. Normalize datasets
# --------------------------------------------------

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

print(
    "Normalized input shape:",
    X_train.shape
)

print()

# --------------------------------------------------
# 7. Build small 1D CNN
# --------------------------------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(100, 6)
    ),

    tf.keras.layers.Conv1D(
        filters=16,
        kernel_size=5,
        activation="relu"
    ),

    tf.keras.layers.MaxPooling1D(
        pool_size=2
    ),

    tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=5,
        activation="relu"
    ),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        3,
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model:")
print("--------------------------------------")

model.summary()

# --------------------------------------------------
# 8. Save normalization values
# --------------------------------------------------

np.savez(
    "data/normalization.npz",
    mean=mean,
    std=std
)

print()
print("Normalization values saved:")
print("data/normalization.npz")

# --------------------------------------------------
# 9. Early stopping
# --------------------------------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

# --------------------------------------------------
# 10. Start training
# --------------------------------------------------

print()
print("======================================")
print(" STARTING TRAINING")
print("======================================")
print()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=8,
    callbacks=[early_stopping],
    verbose=1
)

# --------------------------------------------------
# 11. Save trained model
# --------------------------------------------------

model.save(MODEL_FILE)

print()
print("======================================")
print(" MODEL SAVED")
print("======================================")
print()
print("File:", MODEL_FILE)

# --------------------------------------------------
# 12. Validation evaluation
# --------------------------------------------------

print()
print("======================================")
print(" VALIDATION RESULTS")
print("======================================")

val_loss, val_accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

print(
    f"Validation loss: "
    f"{val_loss:.4f}"
)

print(
    f"Validation accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

# --------------------------------------------------
# 13. Held-out test evaluation
# --------------------------------------------------

print()
print("======================================")
print(" HELD-OUT TEST RESULTS")
print("======================================")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"Test loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

# --------------------------------------------------
# 14. Generate predictions
# --------------------------------------------------

predictions = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    predictions,
    axis=1
)

# --------------------------------------------------
# 15. Confusion matrix
# --------------------------------------------------

print()
print("======================================")
print(" CONFUSION MATRIX")
print("======================================")
print()

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2]
)

print(cm)

# --------------------------------------------------
# 16. Classification report
# --------------------------------------------------

print()
print("======================================")
print(" CLASSIFICATION REPORT")
print("======================================")
print()

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

# --------------------------------------------------
# 17. Individual test predictions
# --------------------------------------------------

print()
print("======================================")
print(" INDIVIDUAL TEST PREDICTIONS")
print("======================================")

test_sources = sources[test_mask]

for i in range(len(X_test)):

    actual_class = CLASS_NAMES[y_test[i]]

    predicted_class = CLASS_NAMES[y_pred[i]]

    confidence = (
        predictions[i][y_pred[i]]
        * 100
    )

    print(
        f"{test_sources[i]} | "
        f"Actual: {actual_class:20s} | "
        f"Predicted: {predicted_class:20s} | "
        f"Confidence: {confidence:.2f}%"
    )

# --------------------------------------------------
# 18. Finished
# --------------------------------------------------

print()
print("======================================")
print(" TRAINING COMPLETE")
print("======================================")