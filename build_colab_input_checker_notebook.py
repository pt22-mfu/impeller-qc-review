"""Build train_input_checker.ipynb with clearly numbered cells for Google Colab."""
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
    add_markdown("""# Stage 6B: Impeller Input Checker Training & Policy Optimization
**Project**: `impeller-qc-review`
**Target Environment**: Google Colab (Recommended Runtime: T4 GPU or Standard CPU)

### Workflow Overview
1. **Inputs**:
   - Supported images from `casting_data.zip` via `outputs/data/split_manifest.csv` (5,199 train, 1,044 validation, 1,041 test held out).
   - Unsupported negative images from `unsupported_data/` via `outputs/data/unsupported_manifest.csv` (60 train, 13 validation, 17 test held out).
2. **Preprocessing**:
   - Decodes images as RGB, resizes to 160×160, and embeds normalization directly inside the Keras model graph.
3. **Architecture**:
   - Frozen ImageNet-pretrained `MobileNetV2` backbone with a lightweight classification head (`GlobalAveragePooling2D` + `Dropout(0.2)` + `Dense(1, sigmoid)`).
4. **Balanced Training**:
   - Fits on **train rows only** (5,199 supported + 60 unsupported = 5,259 total) with balanced class weighting ($w_0 = 43.825, w_1 = 0.506$) for up to 8 epochs with early stopping.
5. **Threshold Selection (Validation Only)**:
   - Evaluates on 1,057 validation images (1,044 supported + 13 unsupported).
   - Constraint: Accept at least 95.0% of supported validation images ($\ge 992$ / 1,044).
   - Objective: Maximize the number of rejected unsupported validation images.
   - Tie-breaker: Prefer higher supported acceptance.
6. **Exports**:
   - Model: `outputs/input_checker/input_checker_mobilenetv2.keras`
   - Policy: `outputs/input_checker/input_checker_policy.json`
   - Validation Metrics: `outputs/input_checker/validation_metrics.json`
   - History: `outputs/input_checker/training_history.json`
   - Per-sample Validation Scores: `outputs/input_checker/input_checker_val_scores.csv`
   - Download Package: `outputs/input_checker/input_checker_outputs.zip`""")

    # Cell 1
    add_markdown("""### Cell 1: Environment Setup, Fixed Seed & GPU Verification
Configure deterministic seeds across Python, NumPy, and TensorFlow, and detect available hardware accelerators (GPU/CPU).""")
    add_code("""# Cell 1: Environment Setup, Fixed Seed & GPU Verification
import csv
import io
import json
import os
import random
import shutil
import time
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
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
Auto-detect and extract inputs:
- `casting_data.zip`
- `unsupported_data.zip` (auto-extracted to `unsupported_data/` if needed)
- `outputs/data/split_manifest.csv`
- `outputs/data/unsupported_manifest.csv`
Creates `outputs/input_checker/`.""")
    add_code("""# Cell 2: Input Verification & Directory Setup
CASTING_ZIP = Path("casting_data.zip")
UNSUPPORTED_ZIP = Path("unsupported_data.zip")
UNSUPPORTED_DIR = Path("unsupported_data")

DATA_DIR = Path("outputs/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path("outputs/input_checker")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Check supported casting_data.zip
assert CASTING_ZIP.is_file(), (
    f"Missing archive: {CASTING_ZIP}! Please upload casting_data.zip to Colab."
)

# 2. Extract unsupported_data.zip if directory not already present
if not UNSUPPORTED_DIR.exists() or len(list(UNSUPPORTED_DIR.rglob("*.*"))) == 0:
    if UNSUPPORTED_ZIP.is_file():
        print(f"Extracting {UNSUPPORTED_ZIP}...")
        with ZipFile(UNSUPPORTED_ZIP, "r") as zf:
            zf.extractall(".")
        print(f"Extracted unsupported_data/ ({len(list(UNSUPPORTED_DIR.rglob('*.*')))} files).")
    else:
        raise FileNotFoundError(
            "Neither unsupported_data/ directory nor unsupported_data.zip was found! "
            "Please upload unsupported_data.zip to Colab."
        )

# 3. Handle split_manifest.csv placement
SPLIT_MANIFEST = DATA_DIR / "split_manifest.csv"
if not SPLIT_MANIFEST.is_file() and Path("split_manifest.csv").is_file():
    shutil.copy("split_manifest.csv", SPLIT_MANIFEST)

assert SPLIT_MANIFEST.is_file(), (
    f"Missing manifest: {SPLIT_MANIFEST}! Please upload split_manifest.csv to outputs/data/ or root."
)

# 4. Handle unsupported_manifest.csv placement
UNSUPPORTED_MANIFEST = DATA_DIR / "unsupported_manifest.csv"
if not UNSUPPORTED_MANIFEST.is_file() and Path("unsupported_manifest.csv").is_file():
    shutil.copy("unsupported_manifest.csv", UNSUPPORTED_MANIFEST)

assert UNSUPPORTED_MANIFEST.is_file(), (
    f"Missing manifest: {UNSUPPORTED_MANIFEST}! Please upload unsupported_manifest.csv to outputs/data/ or root."
)

print("All inputs verified successfully:")
print(f"  - Casting Archive:      {CASTING_ZIP} ({CASTING_ZIP.stat().st_size / (1024**2):.1f} MB)")
print(f"  - Unsupported Folder:   {UNSUPPORTED_DIR} ({len(list(UNSUPPORTED_DIR.rglob('*.*')))} files)")
print(f"  - Supported Manifest:   {SPLIT_MANIFEST}")
print(f"  - Unsupported Manifest: {UNSUPPORTED_MANIFEST}")
print(f"  - Output Directory:     {OUTPUT_DIR}")""")

    # Cell 3
    add_markdown("""### Cell 3: Parse Split Manifests & Enforce Split Isolation
Load both manifests and verify exact counts:
- **Supported**: 5,199 train, 1,044 validation (1,041 test strictly held out).
- **Unsupported**: 60 train, 13 validation (17 test strictly held out).
Confirm that no part groups cross split boundaries.""")
    add_code("""# Cell 3: Parse Split Manifests & Enforce Split Isolation
# 1. Parse supported manifest
supp_train, supp_val, supp_test = [], [], []
supp_groups = {"train": set(), "validation": set(), "test": set()}

with SPLIT_MANIFEST.open("r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        s = row["split"]
        if s == "train":
            supp_train.append(row)
            supp_groups["train"].add(row["group_id"])
        elif s == "validation":
            supp_val.append(row)
            supp_groups["validation"].add(row["group_id"])
        elif s == "test":
            supp_test.append(row)
            supp_groups["test"].add(row["group_id"])

assert len(supp_train) == 5199, f"Expected 5,199 supported train, got {len(supp_train)}"
assert len(supp_val) == 1044, f"Expected 1,044 supported val, got {len(supp_val)}"
assert len(supp_test) == 1041, f"Expected 1,041 supported test, got {len(supp_test)}"

# Verify supported group isolation
assert not supp_groups["train"].intersection(supp_groups["validation"]), "Leak: supp train & val share groups!"
assert not supp_groups["train"].intersection(supp_groups["test"]), "Leak: supp train & test share groups!"
assert not supp_groups["validation"].intersection(supp_groups["test"]), "Leak: supp val & test share groups!"

# 2. Parse unsupported manifest
unsupp_train, unsupp_val, unsupp_test = [], [], []
unsupp_groups = {"train": set(), "validation": set(), "test": set()}

with UNSUPPORTED_MANIFEST.open("r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        s = row["split"]
        if s == "train":
            unsupp_train.append(row)
            unsupp_groups["train"].add(row["group_id"])
        elif s == "validation":
            unsupp_val.append(row)
            unsupp_groups["validation"].add(row["group_id"])
        elif s == "test":
            unsupp_test.append(row)
            unsupp_groups["test"].add(row["group_id"])

assert len(unsupp_train) == 60, f"Expected 60 unsupported train, got {len(unsupp_train)}"
assert len(unsupp_val) == 13, f"Expected 13 unsupported val, got {len(unsupp_val)}"
assert len(unsupp_test) == 17, f"Expected 17 unsupported test, got {len(unsupp_test)}"

# Verify unsupported group isolation
assert not unsupp_groups["train"].intersection(unsupp_groups["validation"]), "Leak: unsupp train & val share groups!"
assert not unsupp_groups["train"].intersection(unsupp_groups["test"]), "Leak: unsupp train & test share groups!"
assert not unsupp_groups["validation"].intersection(unsupp_groups["test"]), "Leak: unsupp val & test share groups!"

print("Manifest split counts & group isolation confirmed:")
print(f"  - Supported:   {len(supp_train):,} train, {len(supp_val):,} validation, {len(supp_test):,} test (HELD OUT)")
print(f"  - Unsupported: {len(unsupp_train):,} train, {len(unsupp_val):,} validation, {len(unsupp_test):,} test (HELD OUT)")
print(f"  - Combined Training Set:   {len(supp_train) + len(unsupp_train):,} samples (5,199 supported + 60 unsupported)")
print(f"  - Combined Validation Set: {len(supp_val) + len(unsupp_val):,} samples (1,044 supported + 13 unsupported)")""")

    # Cell 4
    add_markdown("""### Cell 4: Load Images into Memory & Compute Balanced Class Weights
Read images into memory as uint8 160×160 RGB arrays:
- Supported images are read from `casting_data.zip` (label = 1.0).
- Unsupported images are read from `unsupported_data/` (label = 0.0).
Compute balanced class weights:
$$w_0 = \\frac{N_{\\text{train}}}{2 \\times N_0} = \\frac{5259}{2 \\times 60} = 43.825$$
$$w_1 = \\frac{N_{\\text{train}}}{2 \\times N_1} = \\frac{5259}{2 \\times 5199} \\approx 0.50577$$""")
    add_code("""# Cell 4: Load Images into Memory & Compute Balanced Class Weights
TARGET_SIZE = (160, 160)

def load_combined_split(archive_path, supp_rows, unsupp_rows, target_size=TARGET_SIZE):
    total_n = len(supp_rows) + len(unsupp_rows)
    X = np.empty((total_n, target_size[0], target_size[1], 3), dtype=np.uint8)
    y = np.empty(total_n, dtype=np.float32)
    meta = []
    
    t0 = time.time()
    
    # 1. Load supported images from ZIP (label = 1.0)
    with ZipFile(archive_path, "r") as zf:
        for idx, r in enumerate(supp_rows):
            raw_bytes = zf.read(r["image_path"])
            with Image.open(io.BytesIO(raw_bytes)) as img:
                rgb = img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
                X[idx] = np.asarray(rgb, dtype=np.uint8)
            y[idx] = 1.0
            meta.append({
                "image_path": r["image_path"],
                "true_target": 1,
                "is_supported": 1,
                "category": "supported_impeller"
            })
            
    # 2. Load unsupported images from disk (label = 0.0)
    offset = len(supp_rows)
    for j, r in enumerate(unsupp_rows):
        idx = offset + j
        img_path = Path(r["image_path"])
        with Image.open(img_path) as img:
            rgb = img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
            X[idx] = np.asarray(rgb, dtype=np.uint8)
        y[idx] = 0.0
        meta.append({
            "image_path": r["image_path"],
            "true_target": 0,
            "is_supported": 0,
            "category": r["category"]
        })
        
    elapsed = time.time() - t0
    print(f"Loaded {total_n:,} images in {elapsed:.2f}s | Memory: {X.nbytes / (1024**2):.1f} MB")
    return X, y, meta

print("Loading training data (5,199 supported + 60 unsupported)...")
X_train, y_train, train_meta = load_combined_split(CASTING_ZIP, supp_train, unsupp_train)

print("Loading validation data (1,044 supported + 13 unsupported)...")
X_val, y_val, val_meta = load_combined_split(CASTING_ZIP, supp_val, unsupp_val)

# Compute exact balanced class weights
n_train = len(y_train)
n_unsupp = int(np.sum(y_train == 0.0))
n_supp = int(np.sum(y_train == 1.0))

w0 = float(n_train / (2.0 * n_unsupp))
w1 = float(n_train / (2.0 * n_supp))
class_weights = {0: w0, 1: w1}

print("\\nClass balance and weighting:")
print(f"  - Class 0 (Unsupported): {n_unsupp:,} images -> weight = {w0:.4f}")
print(f"  - Class 1 (Supported):   {n_supp:,} images -> weight = {w1:.4f}")
print(f"  - Effective weight ratio: {w0 / w1:.2f}x")""")

    # Cell 5
    add_markdown("""### Cell 5: Model Definition with Embedded Preprocessing & Frozen MobileNetV2
Construct the Keras model with **embedded preprocessing** (`Resizing` + `Rescaling` inside the model graph).
The backbone is an ImageNet-pretrained `MobileNetV2` with `trainable = False`, connected to `GlobalAveragePooling2D`, `Dropout(0.2)`, and a single sigmoid output (`supported_probability`).""")
    add_code("""# Cell 5: Model Definition with Embedded Preprocessing & Frozen MobileNetV2
def build_input_checker_model(input_shape=(None, None, 3)):
    inputs = keras.Input(shape=input_shape, dtype="float32", name="raw_image_input")
    
    # 1. Embedded resizing: Ensures input is scaled to 160x160
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
    
    # 5. Lightweight Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.2, seed=SEED, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="supported_probability")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name="Impeller_Input_Checker_MobileNetV2")

model = build_input_checker_model()
model.summary()

trainable_count = sum(np.prod(p.shape) for p in model.trainable_weights)
non_trainable_count = sum(np.prod(p.shape) for p in model.non_trainable_weights)
print(f"\\nTrainable parameters:     {trainable_count:,}")
print(f"Non-trainable parameters: {non_trainable_count:,}")""")

    # Cell 6
    add_markdown("""### Cell 6: Model Compilation & Balanced Training
Compile with `Adam(learning_rate=1e-3)` and `BinaryCrossentropy()`.
Train for up to 8 epochs with `class_weight=class_weights` and `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`.
Plot loss and accuracy curves.""")
    add_code("""# Cell 6: Model Compilation & Balanced Training
EPOCHS = 8
BATCH_SIZE = 32

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

print(f"Starting Input Checker training for up to {EPOCHS} epochs with class weighting...")
t0_train = time.time()
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1,
)
train_duration = time.time() - t0_train
print(f"Training completed in {train_duration:.2f} seconds.")

# Plot and save training curves
epochs_range = range(1, len(history.history["loss"]) + 1)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history["loss"], "o-", label="Train Loss (Weighted)")
plt.plot(epochs_range, history.history["val_loss"], "s-", label="Val Loss")
plt.title("Input Checker Loss vs. Epochs")
plt.xlabel("Epoch")
plt.ylabel("Binary Crossentropy")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history["accuracy"], "o-", label="Train Accuracy")
plt.plot(epochs_range, history.history["val_accuracy"], "s-", label="Val Accuracy")
plt.title("Input Checker Accuracy vs. Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
plt.show()

# Export training history
with (OUTPUT_DIR / "training_history.json").open("w", encoding="utf-8") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
print(f"Saved training history to {OUTPUT_DIR / 'training_history.json'}")""")

    # Cell 7
    add_markdown("""### Cell 7: Validation Inference & Score Export
Generate supported-class probability scores for all 1,057 validation images (1,044 supported + 13 unsupported) and export `outputs/input_checker/input_checker_val_scores.csv`.""")
    add_code("""# Cell 7: Validation Inference & Score Export
print("Computing validation predictions...")
val_probs = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0).ravel()

# Attach scores to metadata
for idx, prob in enumerate(val_probs):
    val_meta[idx]["supported_probability"] = float(prob)

val_scores_path = OUTPUT_DIR / "input_checker_val_scores.csv"
with val_scores_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["image_path", "true_target", "is_supported", "category", "supported_probability"]
    )
    writer.writeheader()
    writer.writerows(val_meta)

print(f"Saved {len(val_meta):,} validation scores to {val_scores_path}")

supp_scores = [r["supported_probability"] for r in val_meta if r["is_supported"] == 1]
unsupp_scores = [r["supported_probability"] for r in val_meta if r["is_supported"] == 0]

print(f"\\nValidation Score Distribution:")
print(f"  Supported (N=1,044):   Min={min(supp_scores):.4f}, Mean={np.mean(supp_scores):.4f}, Max={max(supp_scores):.4f}")
print(f"  Unsupported (N=13):    Min={min(unsupp_scores):.4f}, Mean={np.mean(unsupp_scores):.4f}, Max={max(unsupp_scores):.4f}")""")

    # Cell 8
    add_markdown("""### Cell 8: Threshold Optimization (Validation Only)
Select the input checker threshold $\\tau$ using **validation data only**:
1. **Constraint**: Accept at least 95.0% of supported validation images ($\\ge 992$ / 1,044).
2. **Objective**: Maximize the number of rejected unsupported validation images (out of 13).
3. **Tie-breaker**: Prefer higher supported acceptance count / rate.

Decision rule:
- $\\text{supported\\_probability} \\ge \\tau \\implies$ **Accepted** (Predicted Supported = 1)
- $\\text{supported\\_probability} < \\tau \\implies$ **Blocked** (Predicted Unsupported = 0)""")
    add_code("""# Cell 8: Threshold Optimization (Validation Only)
supp_scores_arr = np.array(supp_scores)
unsupp_scores_arr = np.array(unsupp_scores)
N_SUPP = len(supp_scores_arr)       # 1,044
N_UNSUPP = len(unsupp_scores_arr)   # 13

# Test candidate thresholds across [0.001, 0.999] in fine steps plus empirical score values
candidate_thresholds = np.unique(np.concatenate([
    np.linspace(0.001, 0.999, 1997),
    np.clip(supp_scores_arr, 1e-4, 1.0 - 1e-4),
    np.clip(unsupp_scores_arr, 1e-4, 1.0 - 1e-4)
]))
candidate_thresholds.sort()

best_tau = None
best_unsupp_rejected = -1
best_supp_accepted = -1
sweep_results = []

for tau in candidate_thresholds:
    supp_acc_count = int(np.sum(supp_scores_arr >= tau))
    supp_acc_rate = supp_acc_count / N_SUPP
    unsupp_rej_count = int(np.sum(unsupp_scores_arr < tau))
    unsupp_rej_rate = unsupp_rej_count / N_UNSUPP
    
    sweep_results.append({
        "threshold": float(tau),
        "supp_accepted": supp_acc_count,
        "supp_acceptance_rate": supp_acc_rate,
        "unsupp_rejected": unsupp_rej_count,
        "unsupp_rejection_rate": unsupp_rej_rate,
        "meets_constraint": supp_acc_rate >= 0.95
    })
    
    # Feasibility constraint: >= 95% supported acceptance
    if supp_acc_rate >= 0.95:
        # Primary objective: maximize unsupported rejection
        # Tie-breaker: maximize supported acceptance
        if (unsupp_rej_count > best_unsupp_rejected) or (
            unsupp_rej_count == best_unsupp_rejected and supp_acc_count > best_supp_accepted
        ):
            best_unsupp_rejected = unsupp_rej_count
            best_supp_accepted = supp_acc_count
            best_tau = float(tau)

print("=" * 70)
print(f"OPTIMAL THRESHOLD FOUND: tau = {best_tau:.4f}")
print("=" * 70)
print(f"Supported Acceptance:  {best_supp_accepted:,} / {N_SUPP:,} ({best_supp_accepted / N_SUPP:.2%}) [Constraint >= 95.0% MET]")
print(f"Unsupported Rejection: {best_unsupp_rejected:,} / {N_UNSUPP:,} ({best_unsupp_rejected / N_UNSUPP:.2%}) [Maximized]")

# Plot threshold search curve
sweep_df = pd.DataFrame(sweep_results)
plt.figure(figsize=(10, 5))
plt.plot(sweep_df["threshold"], sweep_df["supp_acceptance_rate"] * 100, label="Supported Acceptance Rate (%)", color="navy")
plt.plot(sweep_df["threshold"], sweep_df["unsupp_rejection_rate"] * 100, label="Unsupported Rejection Rate (%)", color="crimson")
plt.axvline(best_tau, color="darkgreen", linestyle="--", label=f"Chosen Threshold tau={best_tau:.4f}")
plt.axhline(95.0, color="gray", linestyle=":", label="95% Supported Acceptance Floor")
plt.title("Input Checker Threshold Optimization Curve (Validation Set)")
plt.xlabel("Threshold (Supported Probability Cutoff)")
plt.ylabel("Rate (%)")
plt.legend(loc="best")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "threshold_search_curve.png", dpi=150)
plt.show()""")

    # Cell 9
    add_markdown("""### Cell 9: Detailed Validation Evaluation & Category Breakdown
Compute confusion matrix, counts for wrongly blocked supported images, wrongly accepted unsupported images, and negative rejection breakdown across all 6 categories.""")
    add_code("""# Cell 9: Detailed Validation Evaluation & Category Breakdown
y_val_true = np.array([r["true_target"] for r in val_meta], dtype=int)
y_val_pred = (val_probs >= best_tau).astype(int)

# Confusion matrix: rows = Actual [0=Unsupported, 1=Supported], cols = Predicted [0=Unsupported, 1=Supported]
cm = confusion_matrix(y_val_true, y_val_pred, labels=[0, 1])
tn, fp = int(cm[0, 0]), int(cm[0, 1])
fn, tp = int(cm[1, 0]), int(cm[1, 1])

supp_wrongly_blocked = fn
unsupp_wrongly_accepted = fp

print("=" * 70)
print(f"VALIDATION PERFORMANCE AT THRESHOLD tau = {best_tau:.4f}")
print("=" * 70)
print(f"Confusion Matrix (labels = [0: Unsupported, 1: Supported]):")
print(f"                     Pred Unsupported (0)    Pred Supported (1)    Total")
print(f"  Actual Unsupp (0): {tn:20d}  {fp:20d}  {tn+fp:7d}")
print(f"  Actual Supp   (1): {fn:20d}  {tp:20d}  {fn+tp:7d}")
print(f"  Total:             {tn+fn:20d}  {fp+tp:20d}  {len(y_val_true):7d}")
print("-" * 70)
print(f"Supported Images Summary:")
print(f"  - Correctly Accepted (TP): {tp:,} / {tp+fn:,} ({tp / (tp+fn):.2%})")
print(f"  - Wrongly Blocked    (FN): {fn:,} / {tp+fn:,} ({fn / (tp+fn):.2%})")
print(f"\\nUnsupported Images Summary:")
print(f"  - Correctly Rejected (TN): {tn:,} / {tn+fp:,} ({tn / (tn+fp):.2%})")
print(f"  - Wrongly Accepted   (FP): {fp:,} / {tn+fp:,} ({fp / (tn+fp):.2%})")
print("=" * 70)

# Category breakdown for unsupported validation images
cat_summary = {}
for r in val_meta:
    if r["is_supported"] == 0:
        cat = r["category"]
        if cat not in cat_summary:
            cat_summary[cat] = {"total": 0, "rejected": 0, "wrongly_accepted": 0, "scores": []}
        cat_summary[cat]["total"] += 1
        cat_summary[cat]["scores"].append(r["supported_probability"])
        if r["supported_probability"] < best_tau:
            cat_summary[cat]["rejected"] += 1
        else:
            cat_summary[cat]["wrongly_accepted"] += 1

print("\\nUnsupported Validation Rejection Breakdown by Category:")
print(f"{'Category':<26} | {'Total':<6} | {'Rejected':<10} | {'Wrongly Acc':<12} | {'Rejection Rate':<14} | {'Mean Score'}")
print("-" * 85)
for cat, stats in sorted(cat_summary.items()):
    rej_rate = stats["rejected"] / stats["total"] * 100.0
    mean_score = float(np.mean(stats["scores"]))
    print(f"{cat:<26} | {stats['total']:<6} | {stats['rejected']:<10} | {stats['wrongly_accepted']:<12} | {rej_rate:>12.1f}%  | {mean_score:.4f}")
print("=" * 85)""")

    # Cell 10
    add_markdown("""### Cell 10: Export Artifacts & Zip for 1-Click Download
Serialize:
- `outputs/input_checker/input_checker_mobilenetv2.keras`
- `outputs/input_checker/input_checker_policy.json`
- `outputs/input_checker/validation_metrics.json`
- `outputs/input_checker/input_checker_outputs.zip`""")
    add_code("""# Cell 10: Export Artifacts & Zip for 1-Click Download
MODEL_PATH = OUTPUT_DIR / "input_checker_mobilenetv2.keras"
POLICY_PATH = OUTPUT_DIR / "input_checker_policy.json"
METRICS_PATH = OUTPUT_DIR / "validation_metrics.json"

# 1. Save Keras model
print(f"Saving model to {MODEL_PATH}...")
model.save(MODEL_PATH)
print(f"Model saved successfully ({MODEL_PATH.stat().st_size / (1024**2):.2f} MB)")

# 2. Save Policy JSON
policy_data = {
    "model_name": "input_checker_mobilenetv2.keras",
    "target_definition": {
        "1": "Supported (valid top-view cast impeller)",
        "0": "Unsupported (out-of-scope part, wrong-view impeller, unrelated object)"
    },
    "routing_rule": {
        "accepted": f"supported_probability >= {best_tau:.4f}",
        "blocked": f"supported_probability < {best_tau:.4f}"
    },
    "threshold": float(best_tau),
    "optimization_constraint": "Supported Acceptance Rate >= 95.0% on validation set",
    "optimization_objective": "Maximize Unsupported Rejection on validation set",
    "validation_summary": {
        "supported_total": tp + fn,
        "supported_accepted": tp,
        "supported_wrongly_blocked": fn,
        "supported_acceptance_rate": float(tp / (tp + fn)),
        "unsupported_total": tn + fp,
        "unsupported_rejected": tn,
        "unsupported_wrongly_accepted": fp,
        "unsupported_rejection_rate": float(tn / (tn + fp)),
        "confusion_matrix_labels": ["Unsupported (0)", "Supported (1)"],
        "confusion_matrix": [
            [tn, fp],
            [fn, tp]
        ]
    },
    "unsupported_category_breakdown": {
        cat: {
            "total": stats["total"],
            "rejected": stats["rejected"],
            "wrongly_accepted": stats["wrongly_accepted"],
            "rejection_rate": float(stats["rejected"] / stats["total"]),
            "mean_score": float(np.mean(stats["scores"]))
        }
        for cat, stats in sorted(cat_summary.items())
    }
}

with POLICY_PATH.open("w", encoding="utf-8") as f:
    json.dump(policy_data, f, indent=2)
print(f"Saved policy to {POLICY_PATH}")

# 3. Save Validation Metrics JSON
metrics_data = {
    "threshold": float(best_tau),
    "accuracy": float((tp + tn) / len(y_val_true)),
    "supported_precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
    "supported_recall": float(tp / (tp + fn)),
    "unsupported_precision": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
    "unsupported_recall": float(tn / (tn + fp)),
    "confusion_matrix": {
        "true_unsupported_pred_unsupported_TN": tn,
        "true_unsupported_pred_supported_FP": fp,
        "true_supported_pred_unsupported_FN": fn,
        "true_supported_pred_supported_TP": tp
    }
}

with METRICS_PATH.open("w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=2)
print(f"Saved metrics to {METRICS_PATH}")

# 4. Package all output files into a single zip for 1-click download
ZIP_EXPORT_PATH = OUTPUT_DIR / "input_checker_outputs.zip"
with ZipFile(ZIP_EXPORT_PATH, "w") as zf:
    for item in OUTPUT_DIR.glob("*"):
        if item.name != "input_checker_outputs.zip" and item.is_file():
            zf.write(item, item.name)

print(f"\\nAll outputs packaged into {ZIP_EXPORT_PATH} ({ZIP_EXPORT_PATH.stat().st_size / (1024**2):.2f} MB)")
print("You can download input_checker_outputs.zip directly to your local computer and extract into outputs/input_checker/.")""")

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

    out_path = Path("train_input_checker.ipynb")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Successfully generated {out_path} ({out_path.stat().st_size / 1024:.1f} KB, {len(cells)} cells).")

if __name__ == "__main__":
    create_notebook()

