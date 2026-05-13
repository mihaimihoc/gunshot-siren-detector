import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras

FEATURES_PATH = "features"
MODEL_PATH    = "model/sound_classifier.keras"
EPOCHS        = 60
BATCH_SIZE    = 32
LABELS        = ["Gunshot", "Siren", "Background"]

os.makedirs("model", exist_ok=True)

X = np.load(os.path.join(FEATURES_PATH, "X.npy"))
y = np.load(os.path.join(FEATURES_PATH, "y.npy"))

print(f"Loaded: X={X.shape}, y={y.shape}")
print(f"  Gunshot   : {np.sum(y==0)}")
print(f"  Siren     : {np.sum(y==1)}")
print(f"  Background: {np.sum(y==2)}\n")

# Train / val / test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n")

# Weights to handle class imbalance during training. 
# Prevents model from ignoring minority classes
weights = compute_class_weight("balanced", classes=np.array([0,1,2]), y=y_train)
class_weight = {0: weights[0], 1: weights[1], 2: weights[2]}
print(f"Class weights: {class_weight}\n")

# model definition
model = keras.Sequential([
    keras.layers.Input(shape=(1024,)), # YAMNet embedding size - 1024-number fingerprint

    keras.layers.Dense(512, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),

    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.35),

    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(3, activation="softmax")
], name="SoundClassifier_v2")

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1), # stops training doesn't improve for 10 epochs, and restores best model weights
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1), # reduces learning rate by half if validation loss doesn't improve for 5 epochs
    keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1) #automatically saves the best model during training based on validation loss
]

# ── TRAIN ────────────────────────────────────────
print("\nTraining...\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=callbacks
)

# evaluate after training completes
print("\nTest Set Results:\n")
y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=LABELS))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
fig_cm, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png", dpi=150)

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history["accuracy"],     label="Train")
ax1.plot(history.history["val_accuracy"], label="Val")
ax1.set_title("Accuracy"); ax1.legend(); ax1.set_xlabel("Epoch")

ax2.plot(history.history["loss"],     label="Train")
ax2.plot(history.history["val_loss"], label="Val")
ax2.set_title("Loss"); ax2.legend(); ax2.set_xlabel("Epoch")

plt.tight_layout()
plt.savefig("model/training_curves.png", dpi=150)
print("\nSaved: confusion_matrix.png + training_curves.png")
print(f"Model saved: {MODEL_PATH}")