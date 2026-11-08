"""Build evaluate_mobilenetv2_test.ipynb for Google Colab."""
import json
from pathlib import Path


def create_test_notebook():
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
    add_markdown("""# Stage 5: MobileNetV2 Inference on Held-Out Test Set
**Project**: `impeller-qc-review`
**Environment**: Google Colab (CPU or GPU)

### Purpose
Run inference only using the pre-trained `defect_mobilenetv2.keras` model on the **1,041 held-out test images** (439 OK, 602 Defective).
Feed raw decoded RGB images directly to the model because the model graph already embeds `Resizing(160, 160)` and `Rescaling(1/127.5, offset=-1.0)`.
Export the test predictions to `mobilenetv2_test_scores.csv`.""")

    # Cell 1
    add_markdown("""### Cell 1: Environment Setup & File Verification
Verify that `defect_mobilenetv2.keras`, `casting_data.zip`, and `outputs/data/split_manifest.csv` are present.""")
    add_code("""# Cell 1: Environment Setup & File Verification
import csv
import io
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow import keras

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version:      {keras.__version__}")

MODEL_PATH = Path("defect_mobilenetv2.keras")
if not MODEL_PATH.is_file():
    MODEL_PATH = Path("outputs/mobilenetv2/defect_mobilenetv2.keras")

ARCHIVE_PATH = Path("casting_data.zip")
MANIFEST_PATH = Path("outputs/data/split_manifest.csv")

assert MODEL_PATH.is_file(), f"Missing model: {MODEL_PATH}! Please upload defect_mobilenetv2.keras."
assert ARCHIVE_PATH.is_file(), f"Missing archive: {ARCHIVE_PATH}! Please upload casting_data.zip."
assert MANIFEST_PATH.is_file(), f"Missing manifest: {MANIFEST_PATH}! Please upload split_manifest.csv to outputs/data/."

print(f"All required input files verified:")
print(f"  Model:    {MODEL_PATH}")
print(f"  Archive:  {ARCHIVE_PATH}")
print(f"  Manifest: {MANIFEST_PATH}")""")

    # Cell 2
    add_markdown("""### Cell 2: Parse Held-Out Test Split from Manifest
Extract exactly the 1,041 test rows (439 OK, 602 Defective) recorded in `outputs/data/split_manifest.csv`.""")
    add_code("""# Cell 2: Parse Held-Out Test Split
test_rows = []
with MANIFEST_PATH.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["split"] == "test":
            test_rows.append(row)

total_test = len(test_rows)
ok_test = sum(1 for r in test_rows if int(r["target"]) == 0)
def_test = sum(1 for r in test_rows if int(r["target"]) == 1)

print(f"Held-Out Test Manifest Rows: {total_test:,}")
print(f"  - Actual OK (0):        {ok_test:,} ({ok_test/total_test:.1%})")
print(f"  - Actual Defective (1): {def_test:,} ({def_test/total_test:.1%})")

assert total_test == 1041, f"Expected 1041 test samples, got {total_test}"
assert ok_test == 439, f"Expected 439 OK samples, got {ok_test}"
assert def_test == 602, f"Expected 602 Defective samples, got {def_test}"
print("Test split verified perfectly.")""")

    # Cell 3
    add_markdown("""### Cell 3: Load Model & Stream Raw RGB Test Images
Load `defect_mobilenetv2.keras`. Decode images as raw RGB arrays and feed them directly to the model (embedded resizing and normalization handle image scaling automatically).""")
    add_code("""# Cell 3: Load Model & Stream Raw RGB Test Images
print(f"Loading trained model: {MODEL_PATH}...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Stream raw RGB images from ZIP
print("Streaming raw RGB test images from casting_data.zip...")
t0 = time.time()
test_images = []
test_targets = []

with ZipFile(ARCHIVE_PATH) as archive:
    for row in test_rows:
        raw_bytes = archive.read(row["image_path"])
        with Image.open(io.BytesIO(raw_bytes)) as img:
            rgb = img.convert("RGB")
            test_images.append(np.asarray(rgb, dtype=np.uint8))
        test_targets.append(int(row["target"]))

X_test_raw = np.array(test_images, dtype=np.float32)
y_test = np.array(test_targets, dtype=np.int64)
elapsed = time.time() - t0
print(f"Decoded {len(X_test_raw):,} raw images in {elapsed:.2f}s | Shape: {X_test_raw.shape}")""")

    # Cell 4
    add_markdown("""### Cell 4: Run Test Inference (Zero Retraining)
Execute model inference to obtain defective-class probability scores for all 1,041 test images.""")
    add_code("""# Cell 4: Run Test Inference
t0_infer = time.time()
# Raw images passed directly into the model
probs = model.predict(X_test_raw, batch_size=32, verbose=1).ravel()
infer_duration = time.time() - t0_infer
print(f"Inference completed in {infer_duration:.2f}s ({len(probs)/infer_duration:.1f} images/sec).")""")

    # Cell 5
    add_markdown("""### Cell 5: Evaluate Test Performance at Cutoff 0.50
Compute Accuracy, Defective Precision, Recall, F1-Score, and the Confusion Matrix on the held-out test set.""")
    add_code("""# Cell 5: Evaluate Test Performance at Cutoff 0.50
CUTOFF = 0.50
preds = (probs >= CUTOFF).astype(int)

acc = float(accuracy_score(y_test, preds))
prec = float(precision_score(y_test, preds, pos_label=1))
rec = float(recall_score(y_test, preds, pos_label=1))
f1 = float(f1_score(y_test, preds, pos_label=1))

cm = confusion_matrix(y_test, preds, labels=[0, 1])
tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

print("=" * 65)
print("MobileNetV2 Held-Out Test Set Performance (Cutoff = 0.50):")
print("=" * 65)
print(f"  - Test Accuracy:             {acc:.4f} ({acc*100:.2f}%)")
print(f"  - Defective Precision:       {prec:.4f} ({prec*100:.2f}%)")
print(f"  - Defective Recall:          {rec:.4f} ({rec*100:.2f}%)")
print(f"  - Defective F1-score:        {f1:.4f} ({f1*100:.2f}%)")
print("\\nConfusion Matrix (Class Order: [0] OK, [1] Defective):")
print("+" + "-" * 50 + "+")
print(f"| {'Actual \\\\ Predicted':<20} | {'Pred OK':>11} | {'Pred Defective':>14} | {'Total':>8} |")
print("+" + "-" * 50 + "+")
print(f"| {'Actual OK':<20} | {tn:>11,d} | {fp:>14,d} | {tn+fp:>8,d} |")
print(f"| {'Actual Defective':<20} | {fn:>11,d} | {tp:>14,d} | {fn+tp:>8,d} |")
print("+" + "-" * 50 + "+")
print(f"| {'Total':<20} | {tn+fn:>11,d} | {fp+tp:>14,d} | {total_test:>8,d} |")
print("+" + "-" * 50 + "+")
print(f"Detailed counts: TN={tn}, FP={fp}, FN={fn}, TP={tp}")""")

    # Cell 6
    add_markdown("""### Cell 6: Export mobilenetv2_test_scores.csv & Download
Export the per-image test predictions to `mobilenetv2_test_scores.csv` containing `image_path`, `true_target`, and `defective_score`.""")
    add_code("""# Cell 6: Export mobilenetv2_test_scores.csv & Download
scores_rows = []
for i, row in enumerate(test_rows):
    score_val = round(float(probs[i]), 6)
    pred_val = int(preds[i])
    scores_rows.append({
        "image_path": row["image_path"],
        "true_target": int(row["target"]),
        "true_label": row["label"],
        "defective_score": score_val,
        "predicted_target_0_50": pred_val,
        "predicted_label_0_50": "def_front" if pred_val == 1 else "ok_front",
        "is_correct": bool(pred_val == int(row["target"])),
    })

scores_df = pd.DataFrame(scores_rows)

# Save to root and outputs directory
output_csv = Path("mobilenetv2_test_scores.csv")
scores_df.to_csv(output_csv, index=False)

output_mobilenet_dir = Path("outputs/mobilenetv2")
output_mobilenet_dir.mkdir(parents=True, exist_ok=True)
scores_df.to_csv(output_mobilenet_dir / "mobilenetv2_test_scores.csv", index=False)

print(f"Saved {len(scores_df):,} test predictions to: {output_csv}")

try:
    from google.colab import files
    files.download(str(output_csv))
    print("Download prompt triggered for mobilenetv2_test_scores.csv.")
except ImportError:
    print("google.colab not detected. Please download mobilenetv2_test_scores.csv manually.")""")

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "evaluate_mobilenetv2_test.ipynb",
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

    out_file = Path("evaluate_mobilenetv2_test.ipynb")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)

    print(f"Generated {out_file} with {len(cells)} cells.")


if __name__ == "__main__":
    create_test_notebook()

