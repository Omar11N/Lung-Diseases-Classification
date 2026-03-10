
# # 🫁 Chest X-Ray Classification with CNN
# 
# **Multi-class classification of chest X-ray images into:**
# - Normal
# - COVID-19
# - Pneumonia
# - Lung Opacity
# 
# **Architecture:** Custom CNN with L1/L2 regularization and Dropout  
# **Final Test Accuracy:** ~86.7%  
# **Dataset:** Chest X-Ray dataset (16k+ images, balanced classes)
# 
# ---
# > 📌 See `ROADMAP.md` for planned extensions: Attention Gates, Grad-CAM, and Cancer Segmentation.


# ## 1. Setup & Imports


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")


# ## 2. Configuration


# ── Paths ──────────────────────────────────────────────────────────────────
DATASET_PATH  = "/kaggle/input/Data-eq/"   # Update if running locally
OUTPUT_PATH   = "/kaggle/working/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── Hyper-parameters ───────────────────────────────────────────────────────
IMAGE_SIZE    = 299
BATCH_SIZE    = 32
EPOCHS_WARM   = 5       # Short warmup run
EPOCHS_FULL   = 75      # Full training run
VAL_SPLIT     = 0.2
TEST_SPLIT    = 0.5     # fraction of val set used as test
RANDOM_SEED   = 42

# ── Classes ────────────────────────────────────────────────────────────────
CLASS_NAMES   = ['Normal', 'Covid', 'Pneumonia', 'Lung_Opacity']
NUM_CLASSES   = len(CLASS_NAMES)

print(f"Classes: {CLASS_NAMES}")
print(f"Image size: {IMAGE_SIZE}×{IMAGE_SIZE}, Batch size: {BATCH_SIZE}")


# ## 3. Data Loading


def load_dataset(dataset_path, class_names, image_size):
    """Load images from disk into numpy arrays.
    
    Expects directory structure:
        dataset_path/
            Train/
                Normal/
                Covid/
                Pneumonia/
                Lung_Opacity/
    """
    images, labels = [], []
    class_counts = {}

    for label_idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, 'Train', class_name)
        
        if not os.path.exists(class_path):
            print(f"⚠️  Warning: path not found — {class_path}")
            continue

        files = os.listdir(class_path)
        class_counts[class_name] = len(files)
        
        for image_name in files:
            img_path = os.path.join(class_path, image_name)
            try:
                with Image.open(img_path) as img:
                    # Ensure grayscale & correct size
                    img = img.convert('L').resize((image_size, image_size))
                    images.append(np.array(img, dtype=np.float32) / 255.0)
                    labels.append(label_idx)
            except Exception as e:
                print(f"⚠️  Skipping {img_path}: {e}")

    print("\n📊 Class distribution:")
    for cls, cnt in class_counts.items():
        print(f"   {cls:15s}: {cnt} images")
    print(f"\n   Total loaded: {len(images)} images")

    return np.array(images), np.array(labels)


X_raw, y_raw = load_dataset(DATASET_PATH, CLASS_NAMES, IMAGE_SIZE)


# ## 4. Exploratory Data Analysis


# ── Sample images ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Sample Images per Class", fontsize=14, fontweight='bold')

for col, class_name in enumerate(CLASS_NAMES):
    class_indices = np.where(y_raw == col)[0]
    for row in range(2):
        idx = class_indices[row]
        axes[row, col].imshow(X_raw[idx], cmap='gray')
        axes[row, col].set_title(class_name if row == 0 else '', fontsize=10)
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'sample_images.png'), dpi=150)
plt.show()

# ── Class distribution bar chart ───────────────────────────────────────────
unique, counts = np.unique(y_raw, return_counts=True)
plt.figure(figsize=(7, 4))
plt.bar([CLASS_NAMES[i] for i in unique], counts, color=['#4CAF50','#2196F3','#FF9800','#9C27B0'])
plt.title("Class Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'class_distribution.png'), dpi=150)
plt.show()


# ## 5. Preprocessing & Splits


# Reshape to (N, H, W, 1) — add channel dimension
X = X_raw.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
y = to_categorical(y_raw, NUM_CLASSES)

# ── Train / Val / Test split ───────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=y_raw
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=TEST_SPLIT, random_state=RANDOM_SEED,
    stratify=np.argmax(y_temp, axis=1)
)

print(f"Train : {X_train.shape[0]} samples")
print(f"Val   : {X_val.shape[0]} samples")
print(f"Test  : {X_test.shape[0]} samples")


# ## 6. Data Augmentation


def add_gaussian_noise(image):
    """Add Gaussian noise for robustness training."""
    sigma = 0.1 ** 0.5
    return np.clip(image + np.random.normal(0, sigma, image.shape), 0, 1)


# Augmentation pipeline
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=add_gaussian_noise
)

val_datagen = ImageDataGenerator()  # No augmentation on val/test

train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=RANDOM_SEED)
val_gen   = val_datagen.flow(X_val,   y_val,   batch_size=BATCH_SIZE, shuffle=False)

print("✅ Data generators ready")


# ## 7. Model Architecture
# 
# Custom CNN with four convolutional blocks, L1/L2 regularization on the first layer, and progressive Dropout.


def build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1), num_classes=NUM_CLASSES):
    """Build the baseline CNN classifier."""
    model = Sequential([
        # ── Block 1 ────────────────────────────────────────────────────────
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape,
               kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
               padding='same', name='conv1_1'),
        Conv2D(64, (3,3), activation='relu', padding='same', name='conv1_2'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # ── Block 2 ────────────────────────────────────────────────────────
        Conv2D(64, (3,3), activation='relu', padding='same', name='conv2_1'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # ── Block 3 ────────────────────────────────────────────────────────
        Conv2D(128, (3,3), activation='relu', padding='same', name='conv3_1'),
        MaxPooling2D((2,2)),
        Dropout(0.40),

        # ── Block 4 ────────────────────────────────────────────────────────
        Conv2D(128, (3,3), activation='relu', padding='same', name='conv4_1'),
        MaxPooling2D((2,2)),
        Dropout(0.50),

        # ── Classifier head ────────────────────────────────────────────────
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.40),
        Dense(num_classes, activation='softmax', name='predictions')
    ])
    return model


model = build_model()
model.summary()


# ## 8. Training
# 
# ### 8a. Warmup (5 epochs — sanity check)


# ── Compile ────────────────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── Callbacks ──────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        filepath=os.path.join(OUTPUT_PATH, 'best_model.h5'),
        monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

print("✅ Callbacks configured: EarlyStopping | ModelCheckpoint | ReduceLROnPlateau")


# Warmup — quick run to verify pipeline
history_warm = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS_WARM,
    validation_data=(X_val, y_val),
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nWarmup → Test loss: {loss:.4f} | Test accuracy: {acc:.4f}")


# ### 8b. Full Training (75 epochs with augmentation + callbacks)


history = model.fit(
    train_gen,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS_FULL,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# Final evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Final → Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")


# ## 9. Evaluation & Visualisation


def plot_history(history, title="Training History"):
    """Plot loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax1.plot(history.history['loss'],     label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()

    ax2.plot(history.history['accuracy'],     label='Train Acc')
    ax2.plot(history.history['val_accuracy'], label='Val Acc')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'training_curves.png'), dpi=150)
    plt.show()

plot_history(history)


# ── Confusion matrix ───────────────────────────────────────────────────────
y_pred  = np.argmax(model.predict(X_test), axis=1)
y_true  = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix.png'), dpi=150)
plt.show()

# ── Per-class report ───────────────────────────────────────────────────────
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


# ── Grad-CAM visualisation ─────────────────────────────────────────────────
# Highlights which regions influenced each prediction.

import tensorflow as tf

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate a Grad-CAM heatmap for a single image."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads     = tape.gradient(class_channel, conv_outputs)
    pooled    = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out  = conv_outputs[0]
    heatmap   = conv_out @ pooled[..., tf.newaxis]
    heatmap   = tf.squeeze(heatmap)
    heatmap   = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def display_gradcam(idx, model, X_test, y_test, class_names, last_conv='conv4_1'):
    img        = X_test[idx]
    img_input  = img[np.newaxis, ...]
    true_label = class_names[np.argmax(y_test[idx])]
    pred_label = class_names[np.argmax(model.predict(img_input, verbose=0))]

    heatmap = make_gradcam_heatmap(img_input, model, last_conv)
    heatmap_resized = np.array(Image.fromarray(heatmap).resize((IMAGE_SIZE, IMAGE_SIZE)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img.squeeze(), cmap='gray')
    axes[0].set_title(f"True: {true_label} | Pred: {pred_label}")
    axes[0].axis('off')

    axes[1].imshow(img.squeeze(), cmap='gray')
    axes[1].imshow(heatmap_resized, cmap='jet', alpha=0.4)
    axes[1].set_title("Grad-CAM Activation")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


# Show Grad-CAM for a few test samples
for i in range(4):
    display_gradcam(i * 10, model, X_test, y_test, CLASS_NAMES)


# ## 10. Save Model


# SavedModel format (recommended)
model.save(os.path.join(OUTPUT_PATH, 'chest_xray_model'), save_format='tf')

# Legacy .h5 for compatibility
model.save(os.path.join(OUTPUT_PATH, 'chest_xray_model.h5'))
model.save_weights(os.path.join(OUTPUT_PATH, 'chest_xray_weights.h5'))

print("✅ Model saved in SavedModel and .h5 formats")


# ## 11. Inference Utility
# 
# Reusable function to run inference on a single image path.


def predict_image(image_path, model, class_names, image_size=299):
    """Load an image and return the predicted class with confidence."""
    with Image.open(image_path) as img:
        img = img.convert('L').resize((image_size, image_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.reshape(1, image_size, image_size, 1)

    probs    = model.predict(arr, verbose=0)[0]
    pred_idx = np.argmax(probs)

    print(f"Prediction : {class_names[pred_idx]}  ({probs[pred_idx]*100:.1f}%)")
    print("Confidence breakdown:")
    for name, prob in zip(class_names, probs):
        bar = '█' * int(prob * 30)
        print(f"  {name:15s}: {prob*100:5.1f}%  {bar}")
    return class_names[pred_idx], probs


# Usage example:
# predict_image('/path/to/xray.png', model, CLASS_NAMES)
