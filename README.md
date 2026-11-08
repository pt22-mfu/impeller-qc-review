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

## Stage 7: Pump Impeller Quality Inspector Demo & Gemini QA Integration

Launch the interactive web application to demonstrate the sequential two-stage QC pipeline (**Upload $\to$ Input Checker $\to$ Defect Detector**) with Gemini QA summary synthesis:

```powershell
.\.venv\Scripts\python.exe app.py
```

Then open **`http://127.0.0.1:8080`** in your browser.

### UI & Pipeline Specifications

- **Header**: "Pump Impeller Quality Inspector" with main instruction "Upload a top-view photo of a pump impeller".
- **Styling**: White cards, light grey background (`#f1f5f9`), teal primary actions (`#0d9488`), pure black text (`#000000`), 32px desktop title, 20px section headings, 16px body text.
- **Layout**: Left column for upload and compact image preview; right column for prominent result and next action. Responsive stack on narrow screens.
- **Four Standard Labels**:
  1. `Unsupported Image`: Triggered when Stage 1 gatekeeper blocks image ($p < 0.0049$). Next action: *"Upload a top-view photo of a pump impeller"*.
  2. `Pass Candidate`: Triggered when defect score $p < 0.50$. Next action: *"Release part to next production step"*.
  3. `Manual Review`: Triggered when defect score $0.50 \le p < 0.76$. Next action: *"Conduct visual inspection at review station"*.
  4. `Hold`: Triggered when defect score $p \ge 0.76$. Next action: *"Hold part for quality supervisor disposition"*.
- **Inspection Details**: Collapsed `<details>` section containing raw probability scores and policy thresholds, removing defect type and severity claims.
- **Gemini QA Configuration & Dual-Model Fallback**:
  - **Secrets & Precedence**: Reads `GEMINI_API_KEY`, `GEMINI_MODEL`, and `GEMINI_FALLBACK_MODEL` from `.streamlit/secrets.toml`, with environment variables taking highest precedence. `.streamlit/secrets.toml` is strictly git-ignored.
  - **Model Hierarchy**: Defaults to `gemini-3.5-flash-lite` (primary) with `gemini-3.1-flash-lite` as the single fallback.
  - **Automatic Fallback**: Tries the fallback model once upon encountering model unavailability (HTTP 404), model-specific quota limits (HTTP 429), or temporary service errors (HTTP 5xx / timeout).
  - **Local Summary Guardrail**: If neither model can answer (or API key is absent), displays a clearly labeled `Local summary` based strictly on the deterministic QC decision and prescribed next action.
  - **Data Privacy**: The server transmits only numerical inspection probabilities, thresholds, and policy decision metadata (raw images are never sent).
  - Automatically clears old summaries when a new image or sample is selected.

### Verification Suite

Run the dedicated Gemini configuration and fallback test suite:

```powershell
.\.venv\Scripts\python.exe test_gemini_config_and_fallback.py
```

Run the full end-to-end interactive demo verification suite:

```powershell
# In terminal 1 (or background):
.\.venv\Scripts\python.exe app.py --port 8086

# In terminal 2:
.\.venv\Scripts\python.exe verify_polished_demo.py
```

### Stage 7 Deliverables

| File | Status | Purpose |
|---|---|---|
| `app.py` | Completed | Web dashboard with upload, sample buttons, collapsed details, Gemini config & Local summary |
| `pipeline.py` | Completed | Two-stage inference pipeline class connecting gatekeeper and defect models |
| `sample_assets/` | Completed | Curated 4 sample images and attribution file for zero-archive lightweight deployment |
| `requirements.txt` | Completed | Pinned Linux CPU-compatible dependencies for cloud deployment (`tensorflow-cpu`, `Pillow`, etc.) |
| `runtime.txt` | Completed | Pinned Python 3.11.9 runtime for Render deployment |
| `.streamlit/secrets.toml` | Completed | Local git-ignored secrets configuration for Gemini API key and model IDs |
| `test_gemini_config_and_fallback.py` | Completed | Unit test suite for config hierarchy, mock fallback, Local summary, and live requests |
| `verify_polished_demo.py` | Completed | End-to-end integration test verifying UI, 4 samples, upload, fallback, and live Gemini QA |
| `smoke_test_demo.py` | Completed | Pipeline unit test confirming all 4 QC outcomes |
| `test_demo_client.py` | Completed | Automated HTTP client verifying API endpoints and live routing |

## Production Cloud Deployment (Render)

This repository is optimized for standalone, lightweight container deployment on [Render](https://render.com) (or any containerized Linux environment) without needing the 90+ MB raw dataset archives.

### 1. Render Web Service Configuration
- **Environment**: `Python`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py`
- **Auto-Deploy**: `Yes` (deploys on every push to `main`)

### 2. Render Environment Variables
| Variable Name | Required | Default / Recommended Value | Purpose |
| :--- | :---: | :--- | :--- |
| `PYTHON_VERSION` | Yes | `3.11.9` | Ensures exact Python runtime matching pre-compiled wheels |
| `PORT` | Auto | (Set dynamically by Render, e.g. `10000`) | Server automatically reads `$PORT` and binds to `0.0.0.0` |
| `GEMINI_API_KEY` | Optional | `AIzaSy...` | Enables Google Gemini QA summaries. If omitted, Local summary activates automatically. |

### 3. First GitHub Push Commands
```bash
# 1. Stage all tracked project files (secrets and dataset archives are automatically ignored)
git add .

# 2. Commit the production deployment state
git commit -m "feat: complete impeller QC inspector demo with two-stage pipeline and Gemini QA"

# 3. Rename branch to main
git branch -M main

# 4. Link to your GitHub repository (replace with your repo URL)
git remote add origin https://github.com/<your-username>/impeller-qc-review.git

# 5. Push to GitHub
git push -u origin main
```

## Audit outputs

| File | Purpose |
|---|---|
| `outputs/data/audit.json` | Dataset counts, duplicate findings and split class balance |
| `outputs/data/removed_images.csv` | Removed identical images and their retained representatives |
| `outputs/data/split_manifest.csv` | Archive paths, labels, original splits, image dimensions, hashes, brightness, contrast, similarity groups and new split assignments |

The classifier target is `1` for defective (`def_front`) and `0` for acceptable (`ok_front`). Keep `test` held out until model and threshold choices are complete. The script pools the supplied folders, removes exact pixel duplicates and keeps images linked by a small perceptual-hash distance in the same split. Similarity groups are a safeguard against detectable leakage. Physical-part independence remains unverified because the archive does not provide part IDs.

The dataset provider lists CC BY-NC-ND 4.0. The project checkpoint contains metadata and code. The original archive is available from the dataset provider.
