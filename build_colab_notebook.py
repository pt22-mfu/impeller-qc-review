"""Build train_mobilenetv2.ipynb with clearly numbered cells for Google Colab."""
import json
from pathlib import Path


def create_notebook():
    cells = []

    def add_markdown(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Header
    add_markdown("""# Stage 3: Impeller Defect Inspection via Transfer Learning (MobileNetV2)
**Project**: `impeller-qc-review`
**Target Environment**: Google Colab (Recommended Runtime: T4 GPU or Standard CPU)

### Workflow Overview
1. **Inputs**: Reads `casting_data.zip` and `outputs/data/split_manifest.csv` using exact predefined splits (5,199 train, 1,044 validation, 1,041 test held out).
2. **Preprocessing**: Reads raw image bytes from ZIP, decodes as RGB, resizes to 160×160, and embeds resizing and normalization directly inside the Keras model graph.
3. **Architecture**: ImageNet-pretrained MobileNetV2 with a frozen backbone, Global Average Pooling, and a single sigmoid output unit.
4. **Training**: Up to 8 epochs with `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)` and fixed seed (`1501418`).
5. **Evaluation**: Evaluates at fixed 0.50 threshold, reporting Accuracy, Defective Precision, Recall, F1, and the Confusion Matrix in `[OK, Defective]` order alongside the Stage 2 Logistic Regression baseline.
6. **Exports**: Saves `defect_mobilenetv2.keras`, training history, validation metrics, and per-image validation probability scores (`validation_scores.csv`).""")

    # Cell 1
    add_markdown("""### Cell 1: Environment Setup, Fixed Seed & GPU Verification
Configure deterministic seeds across Python, NumPy, and TensorFlow, and detect available hardware accelerators (GPU/CPU).""")
    add_code("""# Cell 1: Environment Setup, Fixed Seed & GPU Verification
import csv
import io
import json
import os
import random
import time
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Fixed reproducibility seed
SEED = 1501418
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version:      {keras.__version__}")
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPU Detected:       {gpus[0].name} (Hardware acceleration enabled)")
else:
    print("No GPU detected. Running on CPU (training will succeed, but GPU is recommended for speed).")""")

    # Cell 2
    add_markdown("""### Cell 2: Input Verification & Directory Setup
Verify that `casting_data.zip` and `outputs/data/split_manifest.csv` are uploaded and accessible, and create `outputs/mobilenetv2`.""")
    add_code("""# Cell 2: Input Verification & Directory Setup
ARCHIVE_PATH = Path("casting_data.zip")
MANIFEST_PATH = Path("outputs/data/split_manifest.csv")
BASELINE_METRICS_PATH = Path("outputs/baseline/validation_metrics.json")
OUTPUT_DIR = Path("outputs/mobilenetv2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assert ARCHIVE_PATH.is_file(), (
    f"Missing archive: {ARCHIVE_PATH}! "
    "Please upload casting_data.zip to your Colab root working directory."
)
assert MANIFEST_PATH.is_file(), (
    f"Missing manifest: {MANIFEST_PATH}! "
    "Please upload split_manifest.csv to outputs/data/ in Colab."
)

print("Inputs verified:")
print(f"  Archive:  {ARCHIVE_PATH} ({ARCHIVE_PATH.stat().st_size / (1024**2):.1f} MB)")
print(f"  Manifest: {MANIFEST_PATH}")
print(f"  Outputs:  {OUTPUT_DIR}")""")

    # Cell 3
    add_markdown("""### Cell 3: Parse Split Manifest & Guard Test Set
Load and verify the split manifest. Enforce exact counts (5,199 train images, 1,044 validation images) and verify that the 1,041 test images remain strictly held out and unopened.""")
    add_code("""# Cell 3: Parse Split Manifest & Guard Test Set
train_rows = []
val_rows = []
test_rows = []

with MANIFEST_PATH.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        s = row["split"]
        if s == "train":
            train_rows.append(row)
        elif s == "validation":
            val_rows.append(row)
        elif s == "test":
            test_rows.append(row)
        else:
            raise ValueError(f"Unknown split: {s}")

print("Manifest Split Summary:")
print(f"  - Training samples:   {len(train_rows):,} (Expected: 5,199)")
print(f"  - Validation samples: {len(val_rows):,} (Expected: 1,044)")
print(f"  - Test samples:       {len(test_rows):,} (HELD OUT - strictly unopened)")

assert len(train_rows) == 5199, f"Expected 5,199 train samples, got {len(train_rows)}"
assert len(val_rows) == 1044, f"Expected 1,044 val samples, got {len(val_rows)}"
assert len(test_rows) == 1041, f"Expected 1,041 test samples, got {len(test_rows)}"

train_targets = [int(r["target"]) for r in train_rows]
val_targets = [int(r["target"]) for r in val_rows]
print(f"  - Train class balance: OK={train_targets.count(0):,} ({train_targets.count(0)/len(train_targets):.1%}), Defective={train_targets.count(1):,} ({train_targets.count(1)/len(train_targets):.1%})")
print(f"  - Val class balance:   OK={val_targets.count(0):,} ({val_targets.count(0)/len(val_targets):.1%}), Defective={val_targets.count(1):,} ({val_targets.count(1)/len(val_targets):.1%})")
print("Split integrity confirmed.")""")

    # Cell 4
    add_markdown("""### Cell 4: In-Memory Feature Stream from ZIP (RGB 160×160)
Read image bytes directly from `casting_data.zip`, decode each JPEG as RGB, resize to 160×160 uint8 into memory (~380 MB total), and build high-performance `tf.data.Dataset` pipelines with shuffling and prefetching.""")
    add_code("""# Cell 4: In-Memory Feature Stream from ZIP
TARGET_SIZE = (160, 160)
BATCH_SIZE = 32

def load_split_arrays(archive_path, rows, target_size=TARGET_SIZE):
    n = len(rows)
    # Store images as uint8 [0, 255] to minimize RAM footprint (~380 MB for 5199 images)
    X = np.empty((n, target_size[0], target_size[1], 3), dtype=np.uint8)
    y = np.empty(n, dtype=np.float32)
    t0 = time.time()
    with ZipFile(archive_path) as archive:
        for i, r in enumerate(rows):
            raw_bytes = archive.read(r["image_path"])
            with Image.open(io.BytesIO(raw_bytes)) as img:
                rgb = img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
                X[i] = np.asarray(rgb, dtype=np.uint8)
            y[i] = float(r["target"])
    elapsed = time.time() - t0
    print(f"Loaded {n:,} images in {elapsed:.2f}s | Memory: {X.nbytes / (1024**2):.1f} MB")
    return X, y

print("Loading training split...")
X_train, y_train = load_split_arrays(ARCHIVE_PATH, train_rows)
print("Loading validation split...")
X_val, y_val = load_split_arrays(ARCHIVE_PATH, val_rows)

# Build tf.data.Dataset pipelines
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(buffer_size=len(X_train), seed=SEED)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print(f"Datasets ready: {len(train_ds)} train batches, {len(val_ds)} val batches (Batch size = {BATCH_SIZE})")""")

    # Cell 5
    add_markdown("""### Cell 5: Model Definition with Embedded Preprocessing & Frozen MobileNetV2
Construct the Keras model with **embedded preprocessing** (`Resizing` + `Rescaling` inside the model graph). This guarantees that subsequent inference calls directly accept raw images without requiring an external preprocessing script.
The backbone is an ImageNet-pretrained `MobileNetV2` with `trainable = False`, connected to a `GlobalAveragePooling2D` layer and a single binary sigmoid output.""")
    add_code("""# Cell 5: Model Definition with Embedded Preprocessing & Frozen MobileNetV2
def build_mobilenetv2_qc_model(input_shape=(None, None, 3)):
    # Accepts arbitrary image dimensions (e.g. raw 300x300 uint8/float32 RGB images)
    inputs = keras.Input(shape=input_shape, dtype="float32", name="raw_image_input")
    
    # 1. Embedded resizing: Ensures any input image is scaled to 160x160
    x = layers.Resizing(160, 160, interpolation="bilinear", name="embedded_resizing_160")(inputs)
    
    # 2. Embedded normalization: MobileNetV2 requires [-1.0, 1.0] -> (pixel / 127.5) - 1.0
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="embedded_mobilenetv2_scaling")(x)
    
    # 3. ImageNet pretrained MobileNetV2 backbone (frozen)
    backbone = keras.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = False
    
    # 4. Feature extraction
    x = backbone(x, training=False)
    
    # 5. Global Average Pooling & Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.2, seed=SEED, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="defective_probability")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name="Impeller_MobileNetV2_QC")

model = build_mobilenetv2_qc_model()
model.summary()

trainable_count = sum(np.prod(p.shape) for p in model.trainable_weights)
non_trainable_count = sum(np.prod(p.shape) for p in model.non_trainable_weights)
print(f"\\nTrainable parameters:     {trainable_count:,}")
print(f"Non-trainable parameters: {non_trainable_count:,}")""")

    # Cell 6
    add_markdown("""### Cell 6: Model Compilation & Training (Up to 8 Epochs with EarlyStopping)
Compile with `Adam(learning_rate=1e-3)` and `BinaryCrossentropy()`. Train for up to 8 epochs with `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`. Plot loss and accuracy curves.""")
    add_code("""# Cell 6: Model Compilation & Training
EPOCHS = 8

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1,
)

print(f"Starting MobileNetV2 training for up to {EPOCHS} epochs with EarlyStopping...")
t0_train = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    verbose=1,
)
train_duration = time.time() - t0_train
print(f"Training completed in {train_duration:.2f} seconds.")

# Plot Training & Validation Learning Curves
epochs_range = range(1, len(history.history["loss"]) + 1)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history["loss"], "o-", label="Train Loss")
plt.plot(epochs_range, history.history["val_loss"], "s-", label="Val Loss")
plt.title("Loss vs. Epochs")
plt.xlabel("Epoch")
plt.ylabel("Binary Crossentropy Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history["accuracy"], "o-", label="Train Accuracy")
plt.plot(epochs_range, history.history["val_accuracy"], "s-", label="Val Accuracy")
plt.title("Accuracy vs. Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
plt.show()""")

    # Cell 7
    add_markdown("""### Cell 7: Validation Evaluation at Fixed 0.50 Threshold
Evaluate the trained MobileNetV2 on the 1,044 validation images using a fixed 0.50 probability cutoff.
Compute and report Accuracy, Defective Precision, Recall, F1-Score, and the Confusion Matrix in `[OK, Defective]` order.""")
    add_code("""# Cell 7: Validation Evaluation at Fixed 0.50 Threshold
val_probs = model.predict(val_ds, verbose=0).ravel()
CUTOFF = 0.50
val_preds = (val_probs >= CUTOFF).astype(int)
y_val_int = y_val.astype(int)

acc = float(accuracy_score(y_val_int, val_preds))
prec = float(precision_score(y_val_int, val_preds, pos_label=1))
rec = float(recall_score(y_val_int, val_preds, pos_label=1))
f1 = float(f1_score(y_val_int, val_preds, pos_label=1))

# Confusion matrix: rows = Actual [0=OK, 1=Defective], cols = Predicted [0=OK, 1=Defective]
cm = confusion_matrix(y_val_int, val_preds, labels=[0, 1])
tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

print("=" * 60)
print(f"MobileNetV2 Validation Performance (Cutoff = {CUTOFF:.2f}):")
print("=" * 60)
print(f"  - Accuracy:            {acc:.4f} ({acc*100:.2f}%)")
print(f"  - Defective Precision: {prec:.4f} ({prec*100:.2f}%)")
print(f"  - Defective Recall:    {rec:.4f} ({rec*100:.2f}%)")
print(f"  - Defective F1-score:  {f1:.4f} ({f1*100:.2f}%)")
print("\\nConfusion Matrix (Class Order: [0] OK, [1] Defective):")
print("+" + "-" * 50 + "+")
print(f"| {'Actual \\\\ Predicted':<20} | {'Pred OK':>11} | {'Pred Defective':>14} | {'Total':>8} |")
print("+" + "-" * 50 + "+")
print(f"| {'Actual OK':<20} | {tn:>11,d} | {fp:>14,d} | {tn+fp:>8,d} |")
print(f"| {'Actual Defective':<20} | {fn:>11,d} | {tp:>14,d} | {fn+tp:>8,d} |")
print("+" + "-" * 50 + "+")
print(f"| {'Total':<20} | {tn+fn:>11,d} | {fp+tp:>14,d} | {len(y_val_int):>8,d} |")
print("+" + "-" * 50 + "+")
print(f"Detailed counts: TN={tn}, FP={fp}, FN={fn}, TP={tp}")""")

    # Cell 8
    add_markdown("""### Cell 8: Comparison Against Stage 2 Baseline (Logistic Regression)
Load baseline metrics from `outputs/baseline/validation_metrics.json` and generate a side-by-side comparison table showing metric improvements and error reductions.""")
    add_code("""# Cell 8: Comparison Against Stage 2 Baseline (Logistic Regression)
baseline_metrics = {
    "accuracy": 0.880268,
    "precision": 0.887987,
    "recall": 0.907131,
    "f1": 0.897457,
    "tn": 372,
    "fp": 69,
    "fn": 56,
    "tp": 547,
}

if BASELINE_METRICS_PATH.is_file():
    try:
        with BASELINE_METRICS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            ev = data.get("validation_evaluation", {})
            baseline_metrics["accuracy"] = ev.get("accuracy", baseline_metrics["accuracy"])
            baseline_metrics["precision"] = ev.get("precision", baseline_metrics["precision"])
            baseline_metrics["recall"] = ev.get("recall", baseline_metrics["recall"])
            baseline_metrics["f1"] = ev.get("f1_score", baseline_metrics["f1"])
            cm_data = ev.get("confusion_matrix", {})
            baseline_metrics["tn"] = cm_data.get("true_negative_ok", baseline_metrics["tn"])
            baseline_metrics["fp"] = cm_data.get("false_positive_defective", baseline_metrics["fp"])
            baseline_metrics["fn"] = cm_data.get("false_negative_ok", baseline_metrics["fn"])
            baseline_metrics["tp"] = cm_data.get("true_positive_defective", baseline_metrics["tp"])
        print(f"Loaded baseline metrics from {BASELINE_METRICS_PATH}")
    except Exception as e:
        print(f"Using default baseline constants ({e})")

comparison_df = pd.DataFrame([
    {
        "Metric": "Validation Accuracy",
        "Stage 2 Baseline (LogReg)": f"{baseline_metrics['accuracy']:.4f}",
        "Stage 3 MobileNetV2": f"{acc:.4f}",
        "Absolute Gain (Delta)": f"{acc - baseline_metrics['accuracy']:+.4f} ({(acc - baseline_metrics['accuracy'])*100:+.2f}%)",
    },
    {
        "Metric": "Defective Precision",
        "Stage 2 Baseline (LogReg)": f"{baseline_metrics['precision']:.4f}",
        "Stage 3 MobileNetV2": f"{prec:.4f}",
        "Absolute Gain (Delta)": f"{prec - baseline_metrics['precision']:+.4f} ({(prec - baseline_metrics['precision'])*100:+.2f}%)",
    },
    {
        "Metric": "Defective Recall",
        "Stage 2 Baseline (LogReg)": f"{baseline_metrics['recall']:.4f}",
        "Stage 3 MobileNetV2": f"{rec:.4f}",
        "Absolute Gain (Delta)": f"{rec - baseline_metrics['recall']:+.4f} ({(rec - baseline_metrics['recall'])*100:+.2f}%)",
    },
    {
        "Metric": "Defective F1-score",
        "Stage 2 Baseline (LogReg)": f"{baseline_metrics['f1']:.4f}",
        "Stage 3 MobileNetV2": f"{f1:.4f}",
        "Absolute Gain (Delta)": f"{f1 - baseline_metrics['f1']:+.4f} ({(f1 - baseline_metrics['f1'])*100:+.2f}%)",
    },
    {
        "Metric": "False Negatives (Missed Defects)",
        "Stage 2 Baseline (LogReg)": f"{baseline_metrics['fn']}",
        "Stage 3 MobileNetV2": f"{fn}",
        "Absolute Gain (Delta)": f"{fn - baseline_metrics['fn']:+d}",
    },
    {
        "Metric": "False Positives (False Alarms)",
        "Stage 2 Baseline (LogReg)": f"{baseline_metrics['fp']}",
        "Stage 3 MobileNetV2": f"{fp}",
        "Absolute Gain (Delta)": f"{fp - baseline_metrics['fp']:+d}",
    },
])

print("\\n" + "=" * 85)
print("STAGE 2 BASELINE vs. STAGE 3 MOBILENETV2 VALIDATION COMPARISON")
print("=" * 85)
try:
    from IPython.display import display
    display(comparison_df)
except ImportError:
    print(comparison_df.to_string(index=False))""")

    # Cell 9
    add_markdown("""### Cell 9: Export Artifacts (Model, Training History, Validation Metrics & Per-Image Scores)
Export the trained model and all metadata:
- `defect_mobilenetv2.keras` and `outputs/mobilenetv2/defect_mobilenetv2.keras`
- `outputs/mobilenetv2/training_history.json`: Epoch losses, accuracies, and timing
- `outputs/mobilenetv2/validation_metrics.json`: Final evaluated validation performance
- `outputs/mobilenetv2/validation_scores.csv`: Per-image probabilities for subsequent threshold optimization
Also verifies model reload and executes an inference pass on a raw 300×300 image to confirm that embedded preprocessing works standalone.""")
    add_code("""# Cell 9: Export Artifacts & Sanity Check Reload
MODEL_NAME = "defect_mobilenetv2.keras"
root_model_path = Path(MODEL_NAME)
output_model_path = OUTPUT_DIR / MODEL_NAME

# 1. Save .keras model
model.save(output_model_path)
model.save(root_model_path)
print(f"Model saved to:\\n  - {output_model_path}\\n  - {root_model_path}")

# 2. Export Training History
history_path = OUTPUT_DIR / "training_history.json"
history_serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
history_serializable["epochs_trained"] = len(history.history["loss"])
history_serializable["best_epoch"] = int(np.argmin(history.history["val_loss"]) + 1)
history_serializable["train_time_seconds"] = round(train_duration, 2)
with history_path.open("w", encoding="utf-8") as f:
    json.dump(history_serializable, f, indent=2)
print(f"Training history exported to: {history_path}")

# 3. Export Validation Metrics
metrics_path = OUTPUT_DIR / "validation_metrics.json"
val_metrics_payload = {
    "stage": "Stage 3 - MobileNetV2 Transfer Learning",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "architecture": {
        "backbone": "MobileNetV2 (ImageNet pretrained, frozen)",
        "input_resolution": [160, 160, 3],
        "embedded_preprocessing": ["Resizing(160, 160)", "Rescaling(1/127.5, offset=-1.0)"],
        "trainable_parameters": int(trainable_count),
        "non_trainable_parameters": int(non_trainable_count),
    },
    "training_parameters": {
        "max_epochs": EPOCHS,
        "epochs_trained": len(history.history["loss"]),
        "batch_size": BATCH_SIZE,
        "learning_rate": 1e-3,
        "optimizer": "Adam",
        "seed": SEED,
    },
    "validation_evaluation": {
        "probability_threshold": CUTOFF,
        "positive_class": "Defective (1)",
        "negative_class": "OK (0)",
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "confusion_matrix": {
            "class_order": ["OK", "Defective"],
            "matrix": [[tn, fp], [fn, tp]],
            "true_negative_ok": tn,
            "false_positive_defective": fp,
            "false_negative_ok": fn,
            "true_positive_defective": tp,
        },
    },
}
with metrics_path.open("w", encoding="utf-8") as f:
    json.dump(val_metrics_payload, f, indent=2)
print(f"Validation metrics exported to: {metrics_path}")

# 4. Export Per-Image Validation Scores for QC Threshold Selection
scores_path = OUTPUT_DIR / "validation_scores.csv"
scores_rows = []
for i, row in enumerate(val_rows):
    prob = float(val_probs[i])
    pred_bin = int(prob >= CUTOFF)
    scores_rows.append({
        "image_path": row["image_path"],
        "original_split": row.get("original_split", "validation"),
        "true_target": int(row["target"]),
        "true_label": row.get("label", "def_front" if int(row["target"]) == 1 else "ok_front"),
        "defective_probability": round(prob, 6),
        "predicted_target_0_50": pred_bin,
        "predicted_label_0_50": "def_front" if pred_bin == 1 else "ok_front",
        "is_correct": bool(pred_bin == int(row["target"])),
    })

pd.DataFrame(scores_rows).to_csv(scores_path, index=False)
print(f"Per-image validation scores exported to: {scores_path} ({len(scores_rows):,} rows)")

# 5. Sanity Check: Reload model and test inference on a raw unresized 300x300 image
print("\\nSanity checking model reload and embedded preprocessing...")
reloaded_model = keras.models.load_model(output_model_path)
sample_raw_image = np.random.randint(0, 256, (1, 300, 300, 3), dtype=np.uint8).astype(np.float32)
sample_pred = reloaded_model.predict(sample_raw_image, verbose=0)
print(f"Inference on raw 300x300 image succeeded! Output shape: {sample_pred.shape}, Prob: {float(sample_pred[0][0]):.4f}")""")

    # Cell 10
    add_markdown("""### Cell 10: Download Artifacts Bundle
Package all Stage 3 outputs into `mobilenetv2_outputs.zip` and download it to your local machine to keep your local workspace synchronized.""")
    add_code("""# Cell 10: Download Artifacts Bundle
bundle_zip = "mobilenetv2_outputs.zip"
files_to_bundle = [
    OUTPUT_DIR / "defect_mobilenetv2.keras",
    OUTPUT_DIR / "training_history.json",
    OUTPUT_DIR / "validation_metrics.json",
    OUTPUT_DIR / "validation_scores.csv",
    OUTPUT_DIR / "training_curves.png",
]

with ZipFile(bundle_zip, "w") as zf:
    for f in files_to_bundle:
        if f.is_file():
            zf.write(f, arcname=f"outputs/mobilenetv2/{f.name}")
    if Path(MODEL_NAME).is_file():
        zf.write(MODEL_NAME, arcname=MODEL_NAME)

print(f"Created archive: {bundle_zip} ({Path(bundle_zip).stat().st_size / (1024**2):.1f} MB)")

try:
    from google.colab import files
    files.download(bundle_zip)
    print("Download prompt triggered in Colab.")
except ImportError:
    print("google.colab not detected. Please manually download mobilenetv2_outputs.zip.")""")

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "train_mobilenetv2.ipynb",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    out_file = Path("train_mobilenetv2.ipynb")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)

    print(f"Generated {out_file} with {len(cells)} cells.")


if __name__ == "__main__":
    create_notebook()

