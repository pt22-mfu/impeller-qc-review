"""Build evaluate_input_checker_test.ipynb for Google Colab."""
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
    add_markdown("""# Stage 6C: Input Checker Held-Out Test Evaluation
**Project**: `impeller-qc-review`
**Target Environment**: Google Colab (CPU or GPU)

### Purpose & Requirements
1. **Model & Policy**: Load the trained `input_checker_mobilenetv2.keras` and the locked policy threshold $\\tau$ from `outputs/input_checker/input_checker_policy.json`. Do not retrain or alter thresholds.
2. **Held-Out Test Set**:
   - 1,041 supported test images from `outputs/data/split_manifest.csv` (439 OK, 602 Defective).
   - 17 unsupported test images from `outputs/data/unsupported_manifest.csv`.
   - Total: 1,058 held-out images.
3. **Preprocessing**: Identical to Stage 6B (decoded RGB 160×160 normalized inside the model graph).
4. **Evaluation**:
   - Confusion matrix with labeled axes `[0: Unsupported, 1: Supported]`.
   - Supported images wrongly blocked (FN) and rate.
   - Breakdown of wrongly blocked supported images by **good (OK) vs defective impeller**.
   - Unsupported images wrongly accepted (FP) and rate.
   - Rejection counts and rates by unsupported category.
5. **Exports**:
   - Per-image test scores: `outputs/input_checker/input_checker_test_scores.csv`
   - Test metrics: `outputs/input_checker/test_metrics.json`
   - Download package: `outputs/input_checker/stage6c_test_outputs.zip`""")

    # Cell 1
    add_markdown("""### Cell 1: Environment Setup & Hardware Verification
Initialize environment, verify TensorFlow / Keras versions, and detect GPU / CPU accelerator.""")
    add_code("""# Cell 1: Environment Setup & Hardware Verification
import csv
import io
import json
import os
import shutil
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version:      {keras.__version__}")
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPU Detected:       {gpus[0].name} (Hardware acceleration enabled)")
else:
    print("No GPU detected. Running on CPU (inference on 1,058 images takes ~15-30s).")""")

    # Cell 2
    add_markdown("""### Cell 2: Input Verification & Directory Setup
Auto-locate and verify all required files:
- Model: `outputs/input_checker/input_checker_mobilenetv2.keras`
- Policy: `outputs/input_checker/input_checker_policy.json`
- Archives: `casting_data.zip`, `unsupported_data/` (or `unsupported_data.zip`)
- Manifests: `outputs/data/split_manifest.csv`, `outputs/data/unsupported_manifest.csv`""")
    add_code("""# Cell 2: Input Verification & Directory Setup
OUTPUT_DIR = Path("outputs/input_checker")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("outputs/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Model resolution
MODEL_PATH = OUTPUT_DIR / "input_checker_mobilenetv2.keras"
if not MODEL_PATH.is_file() and Path("input_checker_mobilenetv2.keras").is_file():
    shutil.copy("input_checker_mobilenetv2.keras", MODEL_PATH)
assert MODEL_PATH.is_file(), f"Missing model: {MODEL_PATH}!"

# 2. Policy resolution
POLICY_PATH = OUTPUT_DIR / "input_checker_policy.json"
if not POLICY_PATH.is_file() and Path("input_checker_policy.json").is_file():
    shutil.copy("input_checker_policy.json", POLICY_PATH)
assert POLICY_PATH.is_file(), f"Missing policy: {POLICY_PATH}!"

# 3. Archive & folder resolution
CASTING_ZIP = Path("casting_data.zip")
assert CASTING_ZIP.is_file(), f"Missing archive: {CASTING_ZIP}!"

UNSUPPORTED_DIR = Path("unsupported_data")
UNSUPPORTED_ZIP = Path("unsupported_data.zip")
if not UNSUPPORTED_DIR.exists() or len(list(UNSUPPORTED_DIR.rglob("*.*"))) == 0:
    if UNSUPPORTED_ZIP.is_file():
        print(f"Extracting {UNSUPPORTED_ZIP}...")
        with ZipFile(UNSUPPORTED_ZIP, "r") as zf:
            zf.extractall(".")
    else:
        raise FileNotFoundError("Missing unsupported_data/ folder or unsupported_data.zip!")

# 4. Manifest resolution
SPLIT_MANIFEST = DATA_DIR / "split_manifest.csv"
if not SPLIT_MANIFEST.is_file() and Path("split_manifest.csv").is_file():
    shutil.copy("split_manifest.csv", SPLIT_MANIFEST)
assert SPLIT_MANIFEST.is_file(), f"Missing manifest: {SPLIT_MANIFEST}!"

UNSUPPORTED_MANIFEST = DATA_DIR / "unsupported_manifest.csv"
if not UNSUPPORTED_MANIFEST.is_file() and Path("unsupported_manifest.csv").is_file():
    shutil.copy("unsupported_manifest.csv", UNSUPPORTED_MANIFEST)
assert UNSUPPORTED_MANIFEST.is_file(), f"Missing manifest: {UNSUPPORTED_MANIFEST}!"

# Load policy threshold
with POLICY_PATH.open("r", encoding="utf-8") as f:
    policy_data = json.load(f)
LOCKED_THRESHOLD = float(policy_data["threshold"])

print("All required Stage 6C input files verified:")
print(f"  - Model:     {MODEL_PATH} ({MODEL_PATH.stat().st_size / (1024**2):.2f} MB)")
print(f"  - Policy:    {POLICY_PATH} (Locked Threshold tau = {LOCKED_THRESHOLD:.4f})")
print(f"  - Archives:  {CASTING_ZIP}, {UNSUPPORTED_DIR} ({len(list(UNSUPPORTED_DIR.rglob('*.*')))} files)")
print(f"  - Manifests: {SPLIT_MANIFEST}, {UNSUPPORTED_MANIFEST}")""")

    # Cell 3
    add_markdown("""### Cell 3: Parse Held-Out Test Splits from Manifests
Extract only the test rows:
- **Supported Test**: Exactly 1,041 images (439 OK, 602 Defective).
- **Unsupported Test**: Exactly 17 images across 6 negative categories.
- Total Test Samples: **1,058 images**.""")
    add_code("""# Cell 3: Parse Held-Out Test Splits
supp_test_rows = []
with SPLIT_MANIFEST.open("r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["split"] == "test":
            supp_test_rows.append(row)

unsupp_test_rows = []
with UNSUPPORTED_MANIFEST.open("r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["split"] == "test":
            unsupp_test_rows.append(row)

n_supp = len(supp_test_rows)
n_unsupp = len(unsupp_test_rows)
total_test = n_supp + n_unsupp

assert n_supp == 1041, f"Expected 1,041 supported test samples, got {n_supp}"
assert n_unsupp == 17, f"Expected 17 unsupported test samples, got {n_unsupp}"

supp_ok = sum(1 for r in supp_test_rows if r["target"] == "0")
supp_def = sum(1 for r in supp_test_rows if r["target"] == "1")

print(f"Held-Out Test Set Verified ({total_test:,} samples total):")
print(f"  - Supported Impellers:   {n_supp:,} (100.0%)")
print(f"      - Good / OK:         {supp_ok:,} ({supp_ok / n_supp:.1%})")
print(f"      - Defective:         {supp_def:,} ({supp_def / n_supp:.1%})")
print(f"  - Unsupported Negatives: {n_unsupp:,} (across {len(set(r['category'] for r in unsupp_test_rows))} categories)")""")

    # Cell 4
    add_markdown("""### Cell 4: Load Test Images into Memory
Read 1,058 test images into memory as uint8 160×160 RGB arrays (~78 MB RAM total):
- 1,041 supported images from `casting_data.zip`.
- 17 unsupported images from `unsupported_data/`.""")
    add_code("""# Cell 4: Load Test Images into Memory
TARGET_SIZE = (160, 160)
X_test = np.empty((total_test, TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.uint8)
test_meta = []

t0 = time.time()
# 1. Supported test images from ZIP
with ZipFile(CASTING_ZIP, "r") as zf:
    for idx, r in enumerate(supp_test_rows):
        raw_bytes = zf.read(r["image_path"])
        with Image.open(io.BytesIO(raw_bytes)) as img:
            rgb = img.convert("RGB").resize(TARGET_SIZE, Image.Resampling.BILINEAR)
            X_test[idx] = np.asarray(rgb, dtype=np.uint8)
        test_meta.append({
            "image_path": r["image_path"],
            "true_target": 1,
            "is_supported": 1,
            "impeller_defect_label": "Defective" if r["target"] == "1" else "OK",
            "category": "supported_impeller"
        })

# 2. Unsupported test images from disk
offset = n_supp
for j, r in enumerate(unsupp_test_rows):
    idx = offset + j
    img_path = Path(r["image_path"])
    with Image.open(img_path) as img:
        rgb = img.convert("RGB").resize(TARGET_SIZE, Image.Resampling.BILINEAR)
        X_test[idx] = np.asarray(rgb, dtype=np.uint8)
    test_meta.append({
        "image_path": r["image_path"],
        "true_target": 0,
        "is_supported": 0,
        "impeller_defect_label": "N/A",
        "category": r["category"]
    })

elapsed = time.time() - t0
print(f"Loaded {total_test:,} images in {elapsed:.2f}s | Memory: {X_test.nbytes / (1024**2):.1f} MB")""")

    # Cell 5
    add_markdown("""### Cell 5: Load Saved Model & Run Test Inference
Load `input_checker_mobilenetv2.keras` and run inference on `X_test`.""")
    add_code("""# Cell 5: Load Saved Model & Run Test Inference
print(f"Loading {MODEL_PATH}...")
model = keras.models.load_model(MODEL_PATH)

print(f"Running inference on {total_test:,} held-out test images...")
t0_inf = time.time()
test_probs = model.predict(X_test, batch_size=32, verbose=1).ravel()
inf_duration = time.time() - t0_inf
print(f"Inference complete in {inf_duration:.2f}s ({inf_duration / total_test * 1000:.1f} ms/image).")""")

    # Cell 6
    add_markdown("""### Cell 6: Apply Locked Policy & Save Per-Image Test Scores
Apply the locked threshold $\\tau$ from `input_checker_policy.json`:
$$\\text{Decision} = \\begin{cases} \\text{Accepted (1)}, & \\text{if } p \\ge \\tau \\\\ \\text{Blocked (0)}, & \\text{if } p < \\tau \\end{cases}$$
Export per-image scores to `outputs/input_checker/input_checker_test_scores.csv`.""")
    add_code("""# Cell 6: Apply Locked Policy & Save Per-Image Test Scores
tau = LOCKED_THRESHOLD

for idx, prob in enumerate(test_probs):
    p = float(prob)
    test_meta[idx]["supported_probability"] = p
    pred = 1 if p >= tau else 0
    test_meta[idx]["predicted_label"] = pred
    test_meta[idx]["routing_decision"] = "Accepted" if pred == 1 else "Blocked"

test_scores_path = OUTPUT_DIR / "input_checker_test_scores.csv"
with test_scores_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_path", "true_target", "is_supported", "impeller_defect_label",
            "category", "supported_probability", "predicted_label", "routing_decision"
        ]
    )
    writer.writeheader()
    writer.writerows(test_meta)

print(f"Saved {len(test_meta):,} per-image test scores to {test_scores_path}")

supp_scores = [r["supported_probability"] for r in test_meta if r["is_supported"] == 1]
unsupp_scores = [r["supported_probability"] for r in test_meta if r["is_supported"] == 0]

print(f"\\nHeld-Out Test Score Summary:")
print(f"  - Supported (N={len(supp_scores):,}):   Min={min(supp_scores):.6f}, Median={np.median(supp_scores):.6f}, Max={max(supp_scores):.6f}")
print(f"  - Unsupported (N={len(unsupp_scores):,}): Min={min(unsupp_scores):.6f}, Median={np.median(unsupp_scores):.6f}, Max={max(unsupp_scores):.6f}")""")

    # Cell 7
    add_markdown("""### Cell 7: Performance Evaluation & Detailed Breakdown
Generate:
1. Confusion matrix with labeled axes `[0: Unsupported, 1: Supported]`.
2. Supported images wrongly blocked ($FN$) and rate.
3. Supported wrongly blocked breakdown: **good (OK) vs. defective**.
4. Unsupported images wrongly accepted ($FP$) and rate.
5. Rejection counts and rates by unsupported category.""")
    add_code("""# Cell 7: Performance Evaluation & Detailed Breakdown
y_test_true = np.array([r["true_target"] for r in test_meta], dtype=int)
y_test_pred = np.array([r["predicted_label"] for r in test_meta], dtype=int)

# Confusion matrix: rows = Actual [0=Unsupported, 1=Supported], cols = Predicted [0=Unsupported, 1=Supported]
cm = confusion_matrix(y_test_true, y_test_pred, labels=[0, 1])
tn, fp = int(cm[0, 0]), int(cm[0, 1])
fn, tp = int(cm[1, 0]), int(cm[1, 1])

supp_results = [r for r in test_meta if r["is_supported"] == 1]
unsupp_results = [r for r in test_meta if r["is_supported"] == 0]

# Supported breakdown by good vs defective
ok_results = [r for r in supp_results if r["impeller_defect_label"] == "OK"]
def_results = [r for r in supp_results if r["impeller_defect_label"] == "Defective"]

ok_blocked = sum(1 for r in ok_results if r["predicted_label"] == 0)
ok_acc = sum(1 for r in ok_results if r["predicted_label"] == 1)
def_blocked = sum(1 for r in def_results if r["predicted_label"] == 0)
def_acc = sum(1 for r in def_results if r["predicted_label"] == 1)

print("=" * 80)
print(f"STAGE 6C: HELD-OUT TEST EVALUATION (Threshold tau = {tau:.4f})")
print("=" * 80)

print("\n1. CONFUSION MATRIX (Labels: [0: Unsupported, 1: Supported]):")
print(f"{'':22} | {'Pred Unsupported (0)':<22} | {'Pred Supported (1)':<20} | {'Total'}")
print("-" * 80)
print(f"{'Actual Unsupported (0)':<22} | {tn:<22d} | {fp:<20d} | {tn+fp}")
print(f"{'Actual Supported (1)':<22}   | {fn:<22d} | {tp:<20d} | {fn+tp}")
print("-" * 80)
print(f"{'Total':<22} | {tn+fn:<22d} | {fp+tp:<20d} | {total_test}")

print("\n2. ERROR COUNTS & OVERALL METRICS:")
print(f"  - Supported Images Wrongly Blocked (FN):   {fn:,} / {n_supp:,} ({fn / n_supp:.2%})")
print(f"  - Unsupported Images Wrongly Accepted (FP): {fp:,} / {n_unsupp:,} ({fp / n_unsupp:.2%})")
print(f"  - Supported Acceptance Rate:                {tp:,} / {n_supp:,} ({tp / n_supp:.2%})")
print(f"  - Unsupported Rejection Rate:               {tn:,} / {n_unsupp:,} ({tn / n_unsupp:.2%})")
print(f"  - Overall Held-Out Test Accuracy:           {(tp + tn):,} / {total_test:,} ({(tp + tn) / total_test:.2%})")

print("\n3. SUPPORTED IMAGES WRONGLY BLOCKED BREAKDOWN (GOOD VS. DEFECTIVE):")
print(f"  - Good / OK Impellers:       {ok_blocked:,} wrongly blocked / {len(ok_results):,} ({ok_blocked / len(ok_results):.2%}) | {ok_acc:,} accepted ({ok_acc / len(ok_results):.2%})")
print(f"  - Defective Impellers:       {def_blocked:,} wrongly blocked / {len(def_results):,} ({def_blocked / len(def_results):.2%}) | {def_acc:,} accepted ({def_acc / len(def_results):.2%})")
print(f"  - Total Supported Impellers: {fn:,} wrongly blocked / {n_supp:,} ({fn / n_supp:.2%}) | {tp:,} accepted ({tp / n_supp:.2%})")

# Unsupported breakdown by category
cat_summary = {}
for r in unsupp_results:
    cat = r["category"]
    if cat not in cat_summary:
        cat_summary[cat] = {"total": 0, "rejected": 0, "wrongly_accepted": 0, "scores": []}
    cat_summary[cat]["total"] += 1
    cat_summary[cat]["scores"].append(r["supported_probability"])
    if r["predicted_label"] == 0:
        cat_summary[cat]["rejected"] += 1
    else:
        cat_summary[cat]["wrongly_accepted"] += 1

print("\n4. UNSUPPORTED REJECTION BREAKDOWN BY CATEGORY:")
print(f"{'Category':<26} | {'Total':<6} | {'Rejected':<10} | {'Wrongly Acc':<12} | {'Rejection Rate':<14} | {'Mean Score':<10} | {'Max Score'}")
print("-" * 96)
for cat, stats in sorted(cat_summary.items()):
    rej_pct = stats["rejected"] / stats["total"] * 100.0
    m_score = float(np.mean(stats["scores"]))
    max_s = float(np.max(stats["scores"]))
    print(f"{cat:<26} | {stats['total']:<6} | {stats['rejected']:<10} | {stats['wrongly_accepted']:<12} | {rej_pct:>12.1f}%  | {m_score:<10.4f} | {max_s:.4f}")
print("=" * 96)""")

    # Cell 8
    add_markdown("""### Cell 8: Save Test Metrics & Package for 1-Click Download
Export:
- `outputs/input_checker/test_metrics.json`
- `outputs/input_checker/stage6c_test_outputs.zip`""")
    add_code("""# Cell 8: Save Test Metrics & Package for 1-Click Download
METRICS_PATH = OUTPUT_DIR / "test_metrics.json"

test_metrics = {
    "evaluation_split": "held_out_test",
    "threshold": float(tau),
    "total_test_samples": total_test,
    "overall_accuracy": float((tp + tn) / total_test),
    "supported_evaluation": {
        "total_supported": n_supp,
        "supported_accepted": tp,
        "supported_wrongly_blocked": fn,
        "supported_acceptance_rate": float(tp / n_supp),
        "breakdown_by_impeller_defect_status": {
            "ok_good_impellers": {
                "total": len(ok_results),
                "accepted": ok_acc,
                "wrongly_blocked": ok_blocked,
                "wrongly_blocked_rate": float(ok_blocked / len(ok_results))
            },
            "defective_impellers": {
                "total": len(def_results),
                "accepted": def_acc,
                "wrongly_blocked": def_blocked,
                "wrongly_blocked_rate": float(def_blocked / len(def_results))
            }
        }
    },
    "unsupported_evaluation": {
        "total_unsupported": n_unsupp,
        "unsupported_rejected": tn,
        "unsupported_wrongly_accepted": fp,
        "unsupported_rejection_rate": float(tn / n_unsupp),
        "category_breakdown": {
            cat: {
                "total": stats["total"],
                "rejected": stats["rejected"],
                "wrongly_accepted": stats["wrongly_accepted"],
                "rejection_rate": float(stats["rejected"] / stats["total"]),
                "mean_score": float(np.mean(stats["scores"])),
                "max_score": float(np.max(stats["scores"]))
            }
            for cat, stats in sorted(cat_summary.items())
        }
    },
    "confusion_matrix": {
        "labels": ["Unsupported (0)", "Supported (1)"],
        "matrix": [
            [tn, fp],
            [fn, tp]
        ],
        "true_unsupported_pred_unsupported_TN": tn,
        "true_unsupported_pred_supported_FP": fp,
        "true_supported_pred_unsupported_FN": fn,
        "true_supported_pred_supported_TP": tp
    }
}

with METRICS_PATH.open("w", encoding="utf-8") as f:
    json.dump(test_metrics, f, indent=2)
print(f"Saved test metrics to {METRICS_PATH}")

# Package deliverables into stage6c_test_outputs.zip
ZIP_PATH = OUTPUT_DIR / "stage6c_test_outputs.zip"
with ZipFile(ZIP_PATH, "w") as zf:
    zf.write(test_scores_path, "input_checker_test_scores.csv")
    zf.write(METRICS_PATH, "test_metrics.json")

print(f"\\nPackaged test deliverables into {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024:.1f} KB)")
print("You can download stage6c_test_outputs.zip and extract it into outputs/input_checker/.")""")

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    out_path = Path("evaluate_input_checker_test.ipynb")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Successfully generated {out_path} ({out_path.stat().st_size / 1024:.1f} KB, {len(cells)} cells).")

if __name__ == "__main__":
    create_test_notebook()

