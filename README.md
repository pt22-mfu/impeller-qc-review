# Quality Control Inspector for Pump Parts

AI-assisted visual inspection of cast pump impellers. The first milestone prepares the shared dataset split for the baseline and MobileNetV2.

## Run the dataset audit

Use Python 3.11 or newer. Download the casting archive from the original [Kaggle dataset page](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) or use the supplied `casting_data.zip`. Put the ZIP in this project folder and keep it compressed.

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_data.py casting_data.zip
```

In Colab, upload the ZIP and project files, install the requirements in the notebook, and run `python prepare_data.py casting_data.zip` from the project directory.

## Stage 2: Logistic Regression Baseline

Run the baseline training and validation pipeline:

```powershell
.\.venv\Scripts\python.exe train_baseline.py
```

### Baseline Outputs

| File | Purpose |
|---|---|
| `outputs/baseline/model.joblib` | Fitted `Pipeline(StandardScaler, LogisticRegression)` on 32×32 grayscale features |
| `outputs/baseline/validation_metrics.json` | Validation accuracy, precision, recall, F1, and confusion matrix |

Validation results (5,199 train images, 1,044 validation images, fixed 0.50 threshold):
- Accuracy: **88.03%**
- Defective Precision: **88.80%**
- Defective Recall: **90.71%**
- Defective F1-score: **89.75%**
- Confusion Matrix (Actual rows x Predicted cols: `[0] OK, [1] Defective`): `[[372, 69], [56, 547]]`

The test split (1,041 images) remains strictly held out.

## Stage 3: MobileNetV2 Transfer Learning (Colab)

Run the deep learning transfer learning workflow in Google Colab using `train_mobilenetv2.ipynb`:

1. Upload `train_mobilenetv2.ipynb` to Google Colab.
2. Set runtime type to **Python 3 with T4 GPU** (or standard CPU).
3. Upload `casting_data.zip` to the Colab root directory and `outputs/data/split_manifest.csv` to `outputs/data/`.
4. Run all cells in `train_mobilenetv2.ipynb`.

### Stage 3 Outputs

| File | Status | Purpose |
|---|---|---|
| `train_mobilenetv2.ipynb` | Prepared | Colab training notebook with 10 numbered execution steps |
| `outputs/mobilenetv2/defect_mobilenetv2.keras` | Completed | Trained Keras model with embedded 160×160 resizing and normalization |
| `outputs/mobilenetv2/training_history.json` | Completed | Epoch losses, accuracies, and training duration |
| `outputs/mobilenetv2/validation_metrics.json` | Completed | Validation accuracy (98.56%), precision (99.49%), recall (98.01%), F1 (98.75%) |
| `outputs/mobilenetv2/validation_scores.csv` | Completed | Per-image validation probabilities for threshold optimization |

## Stage 4: QC Routing Policy & Threshold Selection

Run the QC threshold optimization script on the validation predictions:

```powershell
.\.venv\Scripts\python.exe select_qc_policy.py
```

### Stage 4 Outputs

| File | Purpose |
|---|---|
| `select_qc_policy.py` | Script to optimize Pass and Hold thresholds under defect escape and precision constraints |
| `outputs/mobilenetv2/qc_policy.json` | Selected thresholds, 3-way routing counts, workload analysis, and sensitivity sweep |

Validation routing summary (1,044 validation images, `pass_threshold = 0.50`, `hold_threshold = 0.76`):
- **Pass Candidate** (`score < 0.50`): 450 images (438 OK, 12 Defective; 1.99% defect escape rate $\le 5\%$)
- **Manual Review** (`0.50 <= score < 0.76`): 22 images (3 OK, 19 Defective)
- **Hold** (`score >= 0.76`): 572 images (572 Defective, 0 OK; 100.00% precision $\ge 98\%$)
- **Operator checks**: 594 inspections (22 reviews + 572 hold confirmations; 43.1% reduction vs. inspecting all 1,044 images)

The test split (1,041 images) is evaluated in Stage 5.

## Stage 5: Final Held-Out Test Evaluation & Comparison

Evaluate both models on the 1,041 held-out test images (439 OK, 602 Defective) without modifying any parameters or thresholds:

1. **Local Baseline Evaluation**:
   ```powershell
   .\.venv\Scripts\python.exe run_baseline_test.py
   ```
   Generates `outputs/baseline/baseline_test_scores.csv` (83.77% accuracy, 86.45% F1 at 0.50 cutoff).

2. **MobileNetV2 Inference in Colab**:
   - Upload `evaluate_mobilenetv2_test.ipynb`, `defect_mobilenetv2.keras`, `casting_data.zip`, and `outputs/data/split_manifest.csv` to Colab.
   - Run inference to export `mobilenetv2_test_scores.csv` and download it to `outputs/mobilenetv2/`.

3. **Final Comparison & Locked QC Routing Benchmark**:
   ```powershell
   .\.venv\Scripts\python.exe compare_test_models.py
   ```
   Compares both models at cutoff 0.50, applies the locked QC policy (`Pass = 0.50`, `Hold = 0.76`), and saves `outputs/test_benchmark.json`.

### Stage 5 Outputs

| File | Status | Purpose |
|---|---|---|
| `run_baseline_test.py` | Completed | Local script evaluating baseline pipeline on test images |
| `outputs/baseline/baseline_test_scores.csv` | Completed | 1,041 test predictions and probabilities from Baseline model |
| `evaluate_mobilenetv2_test.ipynb` | Completed | Colab inference notebook for MobileNetV2 on test images |
| `outputs/mobilenetv2/mobilenetv2_test_scores.csv` | Completed | 1,041 test predictions and probabilities from MobileNetV2 |
| `compare_test_models.py` | Completed | Verifies scores against manifest, benchmarks models, applies locked QC policy |
| `outputs/test_benchmark.json` | Completed | Final test benchmark and comparative metrics |

Held-out test results summary (1,041 test parts: 439 OK, 602 Defective):
- **Model Comparison (0.50 cutoff)**:
  - Baseline: Accuracy **83.77%**, Precision **83.57%**, Recall **89.53%**, F1 **86.45%** (FN=63, FP=106)
  - MobileNetV2: Accuracy **99.14%**, Precision **99.33%**, Recall **99.17%**, F1 **99.25%** (FN=5, FP=4)
- **Locked QC Policy Routing (`Pass = 0.50`, `Hold = 0.76`)**:
  - Pass Candidate: **440 parts** (435 OK, 5 Defective; defect escape rate = **0.83%** $\le 5\%$)
  - Manual Review: **20 parts** (3 OK, 17 Defective)
  - Hold: **581 parts** (580 Defective, 1 OK; Hold precision = **99.83%** $\ge 98\%$)
  - Operator workload: **601 inspections** vs. 1,041 total parts (**42.3% workload reduction**)

## Stage 6A: Input-Checker Dataset Preparation

The input-checker protects the defect classifier from processing unsupported objects or out-of-scope parts.

- **Supported parts** (positive class): Top-view impellers from `casting_data.zip` (7,284 images across train, validation, and test splits in `outputs/data/split_manifest.csv`).
- **Unsupported parts** (negative class): Out-of-scope industrial objects, similar-looking components, and misoriented impellers placed in `unsupported_data/`.

Run the unsupported manifest builder:

```powershell
.\.venv\Scripts\python.exe prepare_unsupported_manifest.py
```

### Stage 6A Manifest Schema

Manifest location: `outputs/data/unsupported_manifest.csv` (Extended metadata: `outputs/data/unsupported_metadata.json`)

| Column | Description |
|---|---|
| `image_path` | Relative path to image file in `unsupported_data/` |
| `category` | Fine-grained category (`bearing`, `pump_housing`, `round_metal_component`, `misoriented_impeller`, `unrelated_industrial`, `unrelated_general`) |
| `source` | Origin dataset (`Wikimedia Commons`) |
| `license` | Verifiable license (`CC BY`, `CC BY-SA`, `CC0`, `Public domain`, etc.) |
| `group_id` | Source-image group ID (clusters multi-view sets and variants into the same split) |
| `split` | Assigned split (`train`, `validation`, `test`) preserving group isolation |
| `is_supported` | `0` (unsupported negative class) |
| `width`, `height` | Pixel dimensions |
| `sha256` | Byte-level SHA-256 integrity hash |
| `source_url` | Original Wikimedia Commons file-page URL |
| `creator` | Author / artist credit |

### Stage 6A Dataset Summary (90 Images, 79 Groups)

| Category | Train | Validation | Test | Total | Status |
|---|---|---|---|---|---|
| `bearing` | 14 | 3 | 3 | **20** | Filled (ball/roller/thrust bearings on plain backgrounds) |
| `pump_housing` | 12 | 2 | 2 | **16** | Filled (cast pump casings, volutes, end covers) |
| `round_metal_component` | 13 | 3 | 4 | **20** | Filled (flanges, gears, pulleys, sprockets, washers) |
| `misoriented_impeller` | 7 | 1 | 2 | **10** | Filled (side-view / non-top view compressor & pump impellers) |
| `unrelated_industrial` | 8 | 2 | 2 | **12** | Filled (wrenches, calipers, spanners, pliers) |
| `unrelated_general` | 6 | 2 | 4 | **12** | Filled (bananas, fruits on plain backgrounds) |
| **Total** | **60** (66.7%) | **13** (14.4%) | **17** (18.9%) | **90** | Leakage test: 0 of 79 groups cross splits |

## Stage 6B: Input-Checker Model Training & Validation Policy Selection

Train the input-checker gatekeeper and select its acceptance threshold using validation data only:

1. **Google Colab Training (`train_input_checker.ipynb`)**:
   - Uses frozen ImageNet-pretrained `MobileNetV2` with embedded 160×160 normalization.
   - Fits on train rows only (5,199 supported + 60 unsupported = 5,259 images) with balanced class weights ($w_0 = 43.825$, $w_1 = 0.506$).
   - Optimizes threshold on validation data (1,044 supported + 13 unsupported = 1,057 images) under constraint: Supported Acceptance $\ge 95.0\%$, maximizing unsupported rejection.
   - Optimal threshold found: $\tau = \mathbf{0.0049}$.

2. **Validation Policy Outputs**:

| File | Purpose |
|---|---|
| `outputs/input_checker/input_checker_mobilenetv2.keras` | Serialized gatekeeper model with embedded preprocessing |
| `outputs/input_checker/input_checker_policy.json` | Locked policy ($\tau = 0.0049$), routing rules, and validation confusion matrix |
| `outputs/input_checker/validation_metrics.json` | Summary validation metrics (100% supported acceptance, 100% negative rejection) |
| `outputs/input_checker/input_checker_val_scores.csv` | 1,057 validation probability scores |

## Stage 6C: Input-Checker Held-Out Test Evaluation

Evaluate the locked input checker ($\tau = 0.0049$) on the strictly held-out test cohort (1,041 supported + 17 unsupported = 1,058 images) without retraining or threshold adjustment:

```powershell
.\.venv\Scripts\python.exe run_input_checker_test.py
```

### Stage 6C Outputs

| File | Status | Purpose |
|---|---|---|
| `evaluate_input_checker_test.ipynb` | Completed | Colab inference notebook on test cohort |
| `run_input_checker_test.py` | Completed | Local verification and benchmark evaluation script |
| `outputs/input_checker/input_checker_test_scores.csv` | Completed | 1,058 test predictions, probabilities, and routing decisions |
| `outputs/input_checker/test_metrics.json` | Completed | Final test confusion matrix, error counts, and category breakdown |

Held-out test results summary (1,058 test images):
- **Confusion Matrix** (Labels: `[0: Unsupported, 1: Supported]`): `[[15, 2], [0, 1041]]`
- **Supported images wrongly blocked ($FN$)**: **0 / 1,041 (0.00%)**
  - Good / OK impellers: **0 / 439 blocked (0.00%)** (100.00% accepted)
  - Defective impellers: **0 / 602 blocked (0.00%)** (100.00% accepted)
- **Unsupported images wrongly accepted ($FP$)**: **2 / 17 (11.76%)**
  - `bearing_011_spinbal_jpg.jpg`: $p = 0.1583$
  - `misoriented_impeller_001_24_blade_compressor_impeller_nara.jpg`: $p = 0.0051$
- **Unsupported rejection rate ($TN$)**: **15 / 17 (88.24%)**
  - `pump_housing`: 2/2 rejected (100.0%)
  - `round_metal_component`: 4/4 rejected (100.0%)
  - `unrelated_general`: 4/4 rejected (100.0%)
  - `unrelated_industrial`: 2/2 rejected (100.0%)
  - `bearing`: 2/3 rejected (66.7%)
  - `misoriented_impeller`: 1/2 rejected (50.0%)
- **Overall Held-Out Test Accuracy**: **1,056 / 1,058 (99.81%)**

## Stage 7: Defect Detection System for Pump Impellers (Streamlit Application)

Launch the interactive web application:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
*(Alternatively: `.\.venv\Scripts\python.exe app.py`)*

Then open **`http://localhost:8501`** in your browser.

### Layout, Styling & Interactive 3-Step Workflow

- **Desktop 1:2 Layout & Styling**: Uses a white page background (`#ffffff`), light-grey panels (`#f1f5f9`), teal accents (`#0d9488`), light-teal button fills, pure black (`#000000`) text throughout, and `Arial` typography (34 px title, 22 px section headings, 16 px body & buttons). On mobile, stacks cleanly with zero horizontal overflow.
- **Left Third (`Dataset` & Photo Preview)**:
  - Blind held-out test selector covering **all 1,041 casting test images** (`5,199` train / `1,044` val / `1,041` test manifest) and **all 17 unsupported test images**, including error cases, with ground-truth labels hidden until explicitly revealed after testing.
  - Pre-built held-out ZIP & CSV downloads (`data/held_out_test/casting_held_out_test.zip`, `data/held_out_test/unsupported_held_out_test.zip`), exact split manifests (`split_manifest.csv`, `unsupported_manifest.csv`), original image bytes (SHA-256 preserved), source attributions, and full original Kaggle dataset reference.
  - Displays a short project description before a photo is selected; replaces it with the uncropped full photo preview when selected (and restores the description when cleared).
  - Secondary **"Quick examples"** section (clearly separate from the full test sets) containing the 4 preset demo shortcuts.
- **Right Two-Thirds (`Upload a photo`, Step 1 → Step 2 → Step 3)**:
  1. **Step 1 — Photo Eligibility Check** (runs ONLY the gatekeeper model on upload or dataset/quick selection):
     - Accepted: displays *"This photo can be checked for defects."* and the **"Check for defects"** button (no defect scores or Gemini button shown yet).
     - Rejected: displays *"Please upload a clear top-view photo of a pump impeller."* and hides both defect and Gemini buttons.
  2. **Step 2 — Defect Inspection** (defect model runs ONLY when **"Check for defects"** is clicked):
     - Displays one result card with `Pass Candidate`, `Manual Review`, or `Hold`, **"What to do next"** directly below, **"OK score"** and **"Defect score"** clearly labeled as model scores, and thresholds/diagnostics inside a collapsed section.
  3. **Step 3 — Gemini Explanation** (calls Gemini ONLY when **"Explain this result with Gemini"** is clicked):
     - Reads `GEMINI_API_KEY`, `GEMINI_MODEL`, and `GEMINI_FALLBACK_MODEL` from Streamlit Secrets (`st.secrets`), with environment variables taking precedence. Uses `gemini-3.5-flash-lite` primary → `gemini-3.1-flash-lite` fallback → deterministic `Local summary` guardrail if unreachable. Explains the actual verdict in plain English without changing it or inventing defect details.
- Changing or clearing the active photo resets Step 2 defect results, Step 3 Gemini explanation, and revealed ground truth, while preserving them across unrelated Streamlit reruns for the same photo.

### Verification Suite

Run the unit & live browser verification suite (requires `playwright` + Chrome):

```powershell
.\.venv\Scripts\python.exe test_redesign_pipeline.py
```

### Stage 7 Deliverables

| File | Status | Purpose |
|---|---|---|
| `app.py` | Completed | Redesigned Streamlit web app with 1:2 layout, blind held-out testing, 3-step gating, and Gemini explanation |
| `pipeline.py` | Completed | Stage-separated two-model pipeline (`check_eligibility`, `check_defects`, `inspect`) |
| `prepare_held_out_bundles.py` | Completed | Packages byte-exact held-out test sets (1,041 casting + 17 unsupported) with ground-truth CSVs & attribution |
| `data/held_out_test/` | Completed | Pre-built `casting_held_out_test.zip` (9.9 MB) and `unsupported_held_out_test.zip` (3.3 MB) |
| `sample_assets/` | Completed | Curated 4 preset images for the secondary Quick examples section and `ATTRIBUTION.md` |
| `requirements.txt` | Completed | Pinned Linux CPU-compatible dependencies for Streamlit Cloud |
| `.streamlit/config.toml` | Completed | White background, light-grey panel, teal accent, and black text theme |
| `.streamlit/secrets.toml` | Completed | Local git-ignored secrets configuration for Gemini API key and model IDs |
| `test_redesign_pipeline.py` | Completed | Full unit + Playwright desktop/mobile E2E verification suite |

## Production Deployment: Streamlit Community Cloud

This repository is optimized for one-click deployment on [Streamlit Community Cloud](https://share.streamlit.io/) without requiring the 90+ MB raw dataset archives.

### 1. App Configuration
- **Repository**: `pt22-mfu/impeller-qc-review`
- **Branch**: `main`
- **Main file path**: `app.py`

### 2. Advanced Settings
- **Python Version**: `3.11` or `3.12`
- **Secrets** (configured in Streamlit Cloud Dashboard under App Settings > Secrets):
  ```toml
  GEMINI_API_KEY = "AIzaSy..."
  GEMINI_MODEL = "gemini-3.5-flash-lite"
  GEMINI_FALLBACK_MODEL = "gemini-3.1-flash-lite"
  ```

## Audit outputs

| File | Purpose |
|---|---|
| `outputs/data/audit.json` | Dataset counts, duplicate findings and split class balance |
| `outputs/data/removed_images.csv` | Removed identical images and their retained representatives |
| `outputs/data/split_manifest.csv` | Archive paths, labels, original splits, image dimensions, hashes, brightness, contrast, similarity groups and new split assignments |

The classifier target is `1` for defective (`def_front`) and `0` for acceptable (`ok_front`). Keep `test` held out until model and threshold choices are complete. The script pools the supplied folders, removes exact pixel duplicates and keeps images linked by a small perceptual-hash distance in the same split. Similarity groups are a safeguard against detectable leakage. Physical-part independence remains unverified because the archive does not provide part IDs.

The dataset provider lists CC BY-NC-ND 4.0. The project checkpoint contains metadata and code. The original archive is available from the dataset provider.
