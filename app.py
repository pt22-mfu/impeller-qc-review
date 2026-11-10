"""Defect Detection System for Pump Impellers - Streamlit Application.

Sequential Interactive Inspection Workflow:
1. Upload a photo or choose from held-out datasets / quick examples -> runs ONLY Step 1 Eligibility Check.
2. Click "Check for defects" -> runs Step 2 Defect Model, shows verdict, "What to do next", "OK score", "Defect score", and collapsed thresholds.
3. Click "Explain this result with Gemini" -> calls Gemini (with dual-model fallback & Local summary guardrail) on demand.

Run command:
    streamlit run app.py
"""
import csv
import hashlib
import io
import json
import os
import socket
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
import streamlit as st
from pipeline import ImpellerQCPipeline

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Defect Detection System for Pump Impellers",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
  :root {
    --bg-page: #ffffff;
    --bg-panel: #f1f5f9;
    --bg-panel-alt: #f8fafc;
    --border-panel: #cbd5e1;
    --text-black: #000000;
    --teal-accent: #0d9488;
    --teal-btn-bg: #b2f5ea;
    --teal-btn-hover: #81e6d9;
    --teal-btn-sec: #e6fffa;
    --teal-btn-sec-hover: #ccfbf1;
  }

  /* Enforce white background, pure black text, Arial typography, and zero horizontal overflow */
  html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--bg-page) !important;
    color: var(--text-black) !important;
    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 16px !important;
    overflow-x: hidden !important;
  }

  /* Force Arial and pure black text across Markdown & UI widgets, preserving Material Symbols icons */
  .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span:not([data-testid="stIconMaterial"]),
  label, [data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"],
  [data-testid="stExpander"] summary p, .stSelectbox div, .stRadio div {
    font-family: Arial, Helvetica, sans-serif !important;
    color: var(--text-black) !important;
  }

  [data-testid="stIconMaterial"], .material-symbols-rounded {
    font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
  }

  [data-testid="stHeaderActionElements"] {
    display: none !important;
  }

  .stMarkdown p, .stMarkdown li, label, [data-testid="stWidgetLabel"] p {
    font-size: 16px !important;
    line-height: 1.5 !important;
  }

  /* Page Title (34px Arial) and Top Rule matching sketch */
  .app-title-bar {
    border-bottom: 2px solid var(--teal-accent);
    padding: 4px 0 12px 0;
    margin-bottom: 20px;
  }

  h1.main-page-title {
    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 34px !important;
    font-weight: 700 !important;
    color: var(--text-black) !important;
    margin: 0 !important;
    line-height: 1.2 !important;
  }

  /* Section Headings (22px Arial) */
  h2.section-heading {
    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    color: var(--text-black) !important;
    margin: 0 0 10px 0 !important;
    padding-bottom: 6px;
    border-bottom: 2px solid var(--teal-accent);
  }

  h3.subsection-heading {
    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    color: var(--text-black) !important;
    margin: 12px 0 6px 0 !important;
  }

  /* Light-grey Panels with Teal Accents */
  .panel-box {
    background-color: var(--bg-panel);
    border: 1px solid var(--border-panel);
    border-left: 4px solid var(--teal-accent);
    border-radius: 8px;
    padding: 16px 18px;
    margin-bottom: 16px;
    color: var(--text-black) !important;
    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 16px !important;
  }

  .description-box {
    background-color: var(--bg-panel);
    border: 1px solid var(--border-panel);
    border-radius: 8px;
    padding: 16px 18px;
    margin: 10px 0 16px 0;
    color: var(--text-black) !important;
    font-size: 16px !important;
    line-height: 1.55;
  }

  .step-box {
    background-color: var(--bg-panel);
    border: 1px solid var(--border-panel);
    border-radius: 8px;
    padding: 16px 18px;
    margin: 12px 0;
    color: var(--text-black) !important;
  }

  /* Result Card */
  .result-card {
    background-color: var(--bg-panel);
    border: 1px solid var(--border-panel);
    border-left: 5px solid var(--teal-accent);
    border-radius: 8px;
    padding: 18px 20px;
    margin: 12px 0 14px 0;
    color: var(--text-black) !important;
  }

  /* Verdict Badges with Light Fills and Pure Black Text */
  .verdict-badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 6px;
    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #000000 !important;
    margin-bottom: 12px;
  }

  .verdict-pass {
    background-color: #ccf2e1;
    border: 2px solid #0d9488;
  }

  .verdict-review {
    background-color: #fde68a;
    border: 2px solid #b45309;
  }

  .verdict-hold {
    background-color: #fecaca;
    border: 2px solid #dc2626;
  }

  .verdict-rejected {
    background-color: #e2e8f0;
    border: 2px solid #475569;
  }

  .next-action-block {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-left: 4px solid var(--teal-accent);
    border-radius: 6px;
    padding: 12px 14px;
    margin: 8px 0 14px 0;
    color: #000000 !important;
  }

  .scores-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-top: 8px;
  }

  .score-tile {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 10px 14px;
    color: #000000 !important;
  }

  .score-tile-label {
    font-size: 14px !important;
    font-weight: 700 !important;
    color: #000000 !important;
    margin-bottom: 2px;
  }

  .score-tile-value {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #000000 !important;
  }

  /* Gemini Explanation Card */
  .gemini-box {
    background-color: var(--bg-panel);
    border: 1px solid var(--teal-accent);
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 10px;
    color: #000000 !important;
  }

  .gemini-source-tag {
    display: inline-block;
    background-color: var(--teal-btn-bg);
    border: 1px solid var(--teal-accent);
    color: #000000 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    padding: 2px 10px;
    border-radius: 4px;
    margin-bottom: 8px;
  }

  /* Light-Teal Buttons with Pure Black (#000000) Text */
  div.stButton > button, div.stDownloadButton > button {
    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #000000 !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    min-height: 44px !important;
    transition: background-color 0.15s ease;
  }

  div.stButton > button p, div.stButton > button span:not([data-testid="stIconMaterial"]),
  div.stDownloadButton > button p, div.stDownloadButton > button span:not([data-testid="stIconMaterial"]) {
    color: #000000 !important;
    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 16px !important;
  }

  div.stButton > button[kind="primary"] {
    background-color: var(--teal-btn-bg) !important;
    border: 2px solid var(--teal-accent) !important;
  }

  div.stButton > button[kind="primary"]:hover {
    background-color: var(--teal-btn-hover) !important;
  }

  div.stButton > button[kind="secondary"],
  div.stDownloadButton > button {
    background-color: var(--teal-btn-sec) !important;
    border: 1.5px solid var(--teal-accent) !important;
  }

  div.stButton > button[kind="secondary"]:hover,
  div.stDownloadButton > button:hover {
    background-color: var(--teal-btn-sec-hover) !important;
  }

  /* File Uploader Panel (preserve Material Symbols icon font) */
  [data-testid="stFileUploader"] section {
    background-color: var(--bg-panel) !important;
    border: 2px dashed var(--teal-accent) !important;
    color: #000000 !important;
  }

  [data-testid="stFileUploader"] section p,
  [data-testid="stFileUploader"] section small,
  [data-testid="stFileUploader"] section span:not([data-testid="stIconMaterial"]) {
    color: #000000 !important;
    font-family: Arial, Helvetica, sans-serif !important;
  }

  [data-testid="stFileUploader"] section button {
    background-color: var(--teal-btn-bg) !important;
    border: 1.5px solid var(--teal-accent) !important;
    color: #000000 !important;
    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
  }

  /* Expander Light-Grey Styling */
  [data-testid="stExpander"] {
    background-color: var(--bg-panel) !important;
    border: 1px solid var(--border-panel) !important;
    border-radius: 8px !important;
  }

  /* Full Uncropped Photo Preview on Left */
  [data-testid="stImage"] img {
    width: 100% !important;
    height: auto !important;
    max-height: 380px !important;
    object-fit: contain !important;
    border: 1px solid var(--border-panel);
    border-radius: 6px;
    background-color: #ffffff;
  }

  /* Details Table */
  .details-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 15px !important;
    color: #000000 !important;
    background-color: #ffffff;
  }

  .details-table th, .details-table td {
    padding: 8px 10px;
    text-align: left;
    border: 1px solid #cbd5e1;
    color: #000000 !important;
  }

  .details-table th {
    background-color: #e2e8f0;
    font-weight: 700;
  }

  /* Mobile Responsive Rules: clean stack without horizontal overflow */
  @media (max-width: 768px) {
    .block-container {
      padding-left: 14px !important;
      padding-right: 14px !important;
      padding-top: 12px !important;
    }
    h1.main-page-title {
      font-size: 26px !important;
    }
    .scores-row {
      flex-direction: column;
    }
    .details-table {
      display: block;
      overflow-x: auto;
    }
  }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Pipeline Resource Loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading gatekeeper and defect models...")
def load_pipeline():
    """Load and cache the two-stage inspection pipeline."""
    return ImpellerQCPipeline()

pipeline = load_pipeline()

# ---------------------------------------------------------------------------
# Held-Out Test Manifests & Archive Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(".")
SPLIT_MANIFEST_PATH = BASE_DIR / "outputs" / "data" / "split_manifest.csv"
UNSUPP_MANIFEST_PATH = BASE_DIR / "outputs" / "data" / "unsupported_manifest.csv"
CASTING_HELD_OUT_ZIP = BASE_DIR / "data" / "held_out_test" / "casting_held_out_test.zip"
UNSUPP_HELD_OUT_ZIP = BASE_DIR / "data" / "held_out_test" / "unsupported_held_out_test.zip"
ORIG_CASTING_ZIP = BASE_DIR / "casting_data.zip"
ORIG_UNSUPP_ZIP = BASE_DIR / "unsupported_data.zip"

QUICK_EXAMPLES = {
    "unsupported": {
        "id": "quick_unsupported",
        "label": "Unsupported part",
        "path": BASE_DIR / "sample_assets" / "sample_unsupported.jpg",
        "filename": "sample_unsupported.jpg",
    },
    "pass": {
        "id": "quick_pass",
        "label": "Pass Candidate",
        "path": BASE_DIR / "sample_assets" / "sample_pass.jpeg",
        "filename": "sample_pass.jpeg",
    },
    "review": {
        "id": "quick_review",
        "label": "Manual Review",
        "path": BASE_DIR / "sample_assets" / "sample_review.jpeg",
        "filename": "sample_review.jpeg",
    },
    "hold": {
        "id": "quick_hold",
        "label": "Hold",
        "path": BASE_DIR / "sample_assets" / "sample_hold.jpeg",
        "filename": "sample_hold.jpeg",
    },
}


@st.cache_data
def load_test_catalog():
    """Load frozen manifests and prepare blind reviewer test catalogs (interleaved by SHA-256)."""
    casting_test = []
    split_counts = {"train": 0, "validation": 0, "test": 0}
    if SPLIT_MANIFEST_PATH.is_file():
        with SPLIT_MANIFEST_PATH.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                s = r["split"]
                split_counts[s] = split_counts.get(s, 0) + 1
                if s == "test":
                    casting_test.append(r)

    # Interleave def_front and ok_front deterministically without revealing class order
    casting_test.sort(key=lambda r: r["sha256"])
    casting_entries = []
    for idx, r in enumerate(casting_test, start=1):
        rel_subpath = r["image_path"].replace("casting_data/test/", "")
        zip_entry = f"images/{rel_subpath}"
        blind_label = f"Casting Test #{idx:04d} [id:{r['sha256'][:8]}]"
        casting_entries.append({
            "kind": "casting",
            "index": idx,
            "blind_label": blind_label,
            "zip_entry": zip_entry,
            "original_path": r["image_path"],
            "original_filename": Path(r["image_path"]).name,
            "actual_label": "Defective (def_front)" if r["target"] == "1" else "OK (ok_front)",
            "target": r["target"],
            "sha256": r["sha256"],
            "group_id": r["group_id"],
            "source": "Real-Life Industrial Dataset of Casting Product (Kaggle)",
            "creator": "Raviraj Joshi",
            "license": "CC BY-NC-ND 4.0",
            "source_url": "https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product",
        })

    unsupp_test = []
    unsupp_counts = {"train": 0, "validation": 0, "test": 0}
    if UNSUPP_MANIFEST_PATH.is_file():
        with UNSUPP_MANIFEST_PATH.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                s = r["split"]
                unsupp_counts[s] = unsupp_counts.get(s, 0) + 1
                if s == "test":
                    unsupp_test.append(r)

    unsupp_test.sort(key=lambda r: r["sha256"])
    unsupp_entries = []
    for idx, r in enumerate(unsupp_test, start=1):
        fname = Path(r["image_path"]).name
        zip_entry = f"images/{fname}"
        blind_label = f"Unsupported Test #{idx:02d} [id:{r['sha256'][:8]}]"
        unsupp_entries.append({
            "kind": "unsupported",
            "index": idx,
            "blind_label": blind_label,
            "zip_entry": zip_entry,
            "original_path": r["image_path"],
            "original_filename": fname,
            "actual_label": f"Unsupported ({r['category']})",
            "category": r["category"],
            "sha256": r["sha256"],
            "group_id": r["group_id"],
            "source": r["source"],
            "creator": r["creator"],
            "license": r["license"],
            "source_url": r["source_url"],
        })

    return {
        "casting": casting_entries,
        "unsupported": unsupp_entries,
        "split_counts": split_counts,
        "unsupp_counts": unsupp_counts,
    }


def read_test_image_bytes(entry):
    """Read original image bytes cleanly from held-out ZIP bundles or original local archives."""
    if entry["kind"] == "casting":
        if CASTING_HELD_OUT_ZIP.is_file():
            with ZipFile(CASTING_HELD_OUT_ZIP, "r") as zf:
                return zf.read(entry["zip_entry"])
        if ORIG_CASTING_ZIP.is_file():
            with ZipFile(ORIG_CASTING_ZIP, "r") as zf:
                return zf.read(entry["original_path"])
    else:
        if UNSUPP_HELD_OUT_ZIP.is_file():
            with ZipFile(UNSUPP_HELD_OUT_ZIP, "r") as zf:
                return zf.read(entry["zip_entry"])
        disk_path = BASE_DIR / entry["original_path"]
        if disk_path.is_file():
            return disk_path.read_bytes()
        if ORIG_UNSUPP_ZIP.is_file():
            with ZipFile(ORIG_UNSUPP_ZIP, "r") as zf:
                return zf.read(entry["original_path"])
    raise FileNotFoundError(f"Image bytes unavailable for {entry['blind_label']}")


@st.cache_data
def get_ground_truth_csv_bytes(kind):
    """Extract standalone held-out ground-truth CSV bytes from the held-out archive."""
    if kind == "casting" and CASTING_HELD_OUT_ZIP.is_file():
        with ZipFile(CASTING_HELD_OUT_ZIP, "r") as zf:
            return zf.read("casting_test_ground_truth.csv")
    if kind == "unsupported" and UNSUPP_HELD_OUT_ZIP.is_file():
        with ZipFile(UNSUPP_HELD_OUT_ZIP, "r") as zf:
            return zf.read("unsupported_test_ground_truth.csv")
    return b""


# ---------------------------------------------------------------------------
# Gemini Configuration & Call Logic
# ---------------------------------------------------------------------------
class GeminiCallError(Exception):
    pass


class ModelUnavailableError(GeminiCallError):
    pass


class QuotaLimitError(GeminiCallError):
    pass


class TemporaryServiceError(GeminiCallError):
    pass


def get_gemini_config():
    """Load Gemini configuration with environment variables taking precedence over Streamlit secrets."""
    secrets_dict = {}
    try:
        if hasattr(st, "secrets") and st.secrets is not None:
            secrets_dict = dict(st.secrets)
    except Exception:
        pass

    secrets_path = BASE_DIR / ".streamlit" / "secrets.toml"
    if not secrets_dict and secrets_path.is_file():
        try:
            with open(secrets_path, "rb") as f:
                secrets_dict = tomllib.load(f)
        except Exception as e:
            sys.stderr.write(f"Warning: Failed to parse {secrets_path}: {e}\n")

    def lookup_secret(keys):
        for k in keys:
            if k in secrets_dict and isinstance(secrets_dict[k], str) and secrets_dict[k].strip():
                return secrets_dict[k].strip()
            gemini_sec = secrets_dict.get("gemini", {})
            if isinstance(gemini_sec, dict) and k in gemini_sec and isinstance(gemini_sec[k], str) and gemini_sec[k].strip():
                return gemini_sec[k].strip()
        return None

    api_key = os.environ.get("GEMINI_API_KEY")
    key_src = "environment" if api_key else None
    if not api_key:
        api_key = lookup_secret(["GEMINI_API_KEY", "gemini_api_key", "api_key", "key"])
        if api_key:
            key_src = "streamlit_secrets"

    primary_model = os.environ.get("GEMINI_MODEL") or os.environ.get("GEMINI_PRIMARY_MODEL")
    if not primary_model:
        primary_model = lookup_secret(["GEMINI_MODEL", "model", "model_id", "primary_model"])
    if not primary_model:
        primary_model = "gemini-3.5-flash-lite"

    fallback_model = os.environ.get("GEMINI_FALLBACK_MODEL")
    if not fallback_model:
        fallback_model = lookup_secret(["GEMINI_FALLBACK_MODEL", "fallback_model", "fallback_model_id"])
    if not fallback_model:
        fallback_model = "gemini-3.1-flash-lite"

    return {
        "api_key": api_key,
        "primary_model": primary_model,
        "fallback_model": fallback_model,
        "key_configured": bool(api_key),
        "key_source": key_src or "none",
    }


def generate_local_summary(data):
    """Plain-English rule-based fallback summary if Gemini cannot answer."""
    action_label = data.get("action_label", "Pass Candidate")
    detector = data.get("defect_detector", {})
    p_def = detector.get("defective_probability")
    p_ok = detector.get("ok_probability")
    tau_pass = detector.get("pass_threshold", 0.50)
    tau_hold = detector.get("hold_threshold", 0.76)

    def_str = f"{p_def:.4f}" if p_def is not None else "N/A"
    ok_str = f"{p_ok:.4f}" if p_ok is not None else "N/A"

    if action_label == "Pass Candidate":
        return (
            f"The defect model produced an OK score of {ok_str} and a Defect score of {def_str}. "
            f"Because the Defect score is below the {tau_pass:.2f} pass cutoff, the system assigns Pass Candidate. "
            f"Pass candidate. Follow your normal release procedure."
        )
    if action_label == "Manual Review":
        return (
            f"The defect model produced an OK score of {ok_str} and a Defect score of {def_str}. "
            f"Because the Defect score falls in the review zone between {tau_pass:.2f} and {tau_hold:.2f}, the system assigns Manual Review. "
            f"Conduct visual inspection at review station."
        )
    if action_label == "Hold":
        return (
            f"The defect model produced an OK score of {ok_str} and a Defect score of {def_str}. "
            f"Because the Defect score is at or above the {tau_hold:.2f} hold cutoff, the system assigns Hold. "
            f"Hold for supervisor review."
        )
    return f"Disposition: {action_label}. Next action: {data.get('next_action', '')}"


def build_gemini_prompt(data):
    """Construct a plain-English Gemini explanation prompt based strictly on actual results."""
    action_label = data.get("action_label", "Unknown")
    next_action = data.get("next_action", "")
    p_supp = data.get("gatekeeper", {}).get("supported_probability", 0.0)
    det = data.get("defect_detector", {})
    p_def = det.get("defective_probability", 0.0)
    p_ok = det.get("ok_probability", 1.0 - p_def)
    tau_pass = det.get("pass_threshold", 0.50)
    tau_hold = det.get("hold_threshold", 0.76)

    return (
        f"You are explaining an automated visual inspection result to a quality reviewer in plain English.\n"
        f"Actual model results:\n"
        f"- Photo eligibility score (Stage 1 Gatekeeper): {p_supp:.4f} (accepted)\n"
        f"- Defect Inspection Verdict: {action_label}\n"
        f"- Prescribed action (What to do next): {next_action}\n"
        f"- Model OK score: {p_ok:.4f}\n"
        f"- Model Defect score: {p_def:.4f} (Pass cutoff: < {tau_pass:.2f}, Hold cutoff: >= {tau_hold:.2f})\n\n"
        f"Instructions:\n"
        f"1. Write 2 short, complete sentences (under 55 words total) in plain English stating the verdict '{action_label}', OK score ({p_ok:.4f}), and Defect score ({p_def:.4f}) relative to the cutoffs.\n"
        f"2. End with the prescribed action: '{next_action}'.\n"
        f"3. Do NOT change the decision, do NOT invent any specific physical defect details (no mention of cracks, porosity, holes, etc.), "
        f"and do NOT claim that quality standards were verified or physical defects were confirmed."
    )


def _call_gemini_api(model_name, api_key, prompt, timeout=20, simulate=None):
    """Call Google Gemini generateContent endpoint for a specific model."""
    if simulate:
        sim = str(simulate).lower()
        if sim in ["404", "model_unavailable", "not_found"]:
            raise ModelUnavailableError(f"Simulated 404 for {model_name}")
        if sim in ["429", "quota", "rate_limit"]:
            raise QuotaLimitError(f"Simulated 429 for {model_name}")
        if sim in ["500", "503", "service_error", "temporary_error", "timeout"]:
            raise TemporaryServiceError(f"Simulated service error for {model_name}")
        if sim in ["success", "mock_success"]:
            return f"Simulated Gemini explanation ({model_name})."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 350},
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                resp_json = json.loads(resp.read().decode("utf-8"))
                candidates = resp_json.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"].strip()
                raise ValueError("Empty response payload from Gemini API.")
            raise TemporaryServiceError(f"Unexpected HTTP status {resp.status}")
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="ignore")
        if e.code == 404:
            raise ModelUnavailableError(f"Model unavailable (HTTP 404): {err_msg[:120]}")
        if e.code == 429:
            raise QuotaLimitError(f"Quota exceeded (HTTP 429): {err_msg[:120]}")
        if e.code in [500, 502, 503, 504]:
            raise TemporaryServiceError(f"Temporary service error (HTTP {e.code}): {err_msg[:120]}")
        if e.code == 400 and ("not found" in err_msg.lower() or "not supported" in err_msg.lower()):
            raise ModelUnavailableError(f"Model unavailable (HTTP 400): {err_msg[:120]}")
        raise RuntimeError(f"Gemini API HTTP {e.code}: {err_msg[:120]}")
    except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
        raise TemporaryServiceError(f"Network/timeout error: {e}")


def generate_gemini_qa_summary(data, api_key=None, config_override=None, simulation=None):
    """Explain the actual inspection decision with Gemini primary -> fallback -> Local summary."""
    config = config_override or get_gemini_config()
    key = api_key or config.get("api_key")
    primary_model = config.get("primary_model", "gemini-3.5-flash-lite")
    fallback_model = config.get("fallback_model", "gemini-3.1-flash-lite")

    if not key:
        return {
            "success": True,
            "source": "local",
            "is_local": True,
            "label": "Local summary",
            "summary": generate_local_summary(data),
            "model_used": None,
            "fallback_used": False,
            "reason": "GEMINI_API_KEY is not configured in environment or Streamlit Secrets.",
        }

    prompt = build_gemini_prompt(data)
    primary_sim = simulation.get("primary") if isinstance(simulation, dict) else (
        "404" if simulation in ["primary_404", "primary_unavailable"] else (
        "429" if simulation in ["primary_429", "both_fail"] else None))
    fallback_sim = simulation.get("fallback") if isinstance(simulation, dict) else (
        "503" if simulation == "both_fail" else None)

    primary_error = None
    try:
        text = _call_gemini_api(primary_model, key, prompt, simulate=primary_sim)
        return {
            "success": True,
            "source": "gemini",
            "is_local": False,
            "label": f"Gemini ({primary_model})",
            "summary": text,
            "model_used": primary_model,
            "fallback_used": False,
            "reason": None,
        }
    except Exception as e:
        primary_error = str(e)

    try:
        text = _call_gemini_api(fallback_model, key, prompt, simulate=fallback_sim)
        return {
            "success": True,
            "source": "gemini",
            "is_local": False,
            "label": f"Gemini ({fallback_model})",
            "summary": text,
            "model_used": fallback_model,
            "fallback_used": True,
            "reason": f"Fallback used after primary ({primary_model}) error.",
        }
    except Exception as e:
        fallback_error = str(e)

    return {
        "success": True,
        "source": "local",
        "is_local": True,
        "label": "Local summary",
        "summary": generate_local_summary(data),
        "model_used": None,
        "fallback_used": True,
        "reason": f"Gemini unreachable ({primary_error}; {fallback_error}).",
    }


# ---------------------------------------------------------------------------
# Session State Management
# ---------------------------------------------------------------------------
STATE_DEFAULTS = {
    "active_photo_id": None,
    "active_bytes": None,
    "active_caption": None,
    "active_internal_name": None,
    "active_source_kind": None,  # "upload", "dataset_casting", "dataset_unsupported", "quick"
    "active_dataset_entry": None,
    "eligibility_result": None,
    "defect_result": None,
    "gemini_result": None,
    "label_revealed": False,
    "uploader_key": 0,
    "casting_select_key": 0,
    "unsupp_select_key": 0,
    "gemini_call_count": 0,
}
for _k, _v in STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def set_active_photo(
    img_bytes,
    photo_id,
    caption,
    source_kind,
    internal_filename="",
    dataset_entry=None,
):
    """Switch active photo, clear Step 2 / Step 3 / revealed label, and run ONLY Step 1 Eligibility."""
    st.session_state["active_photo_id"] = photo_id
    st.session_state["active_bytes"] = img_bytes
    st.session_state["active_caption"] = caption
    st.session_state["active_internal_name"] = internal_filename
    st.session_state["active_source_kind"] = source_kind
    st.session_state["active_dataset_entry"] = dataset_entry

    # Strictly reset prior Step 2 defect result, Step 3 Gemini explanation, and revealed ground truth
    st.session_state["defect_result"] = None
    st.session_state["gemini_result"] = None
    st.session_state["label_revealed"] = False

    # Reset competing input widgets so their stale state cannot re-trigger on rerun
    if source_kind != "upload":
        st.session_state["uploader_key"] += 1
    if source_kind != "dataset_casting":
        st.session_state["casting_select_key"] += 1
    if source_kind != "dataset_unsupported":
        st.session_state["unsupp_select_key"] += 1

    # Stage 1 ONLY: Run photo eligibility check
    st.session_state["eligibility_result"] = pipeline.check_eligibility(
        img_bytes, filename=internal_filename
    )


def clear_active_photo():
    """Clear active photo, reset all results/explanations, and restore description."""
    st.session_state["active_photo_id"] = None
    st.session_state["active_bytes"] = None
    st.session_state["active_caption"] = None
    st.session_state["active_internal_name"] = None
    st.session_state["active_source_kind"] = None
    st.session_state["active_dataset_entry"] = None
    st.session_state["eligibility_result"] = None
    st.session_state["defect_result"] = None
    st.session_state["gemini_result"] = None
    st.session_state["label_revealed"] = False
    st.session_state["uploader_key"] += 1
    st.session_state["casting_select_key"] += 1
    st.session_state["unsupp_select_key"] += 1


# ---------------------------------------------------------------------------
# UI Layout Rendering
# ---------------------------------------------------------------------------
def main():
    catalog = load_test_catalog()

    # Header bar across top matching sketch
    st.markdown(
        """
        <div class="app-title-bar">
          <h1 class="main-page-title">Defect Detection System for Pump Impellers</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Desktop 1:2 Column Ratio (Left third = Dataset & Description/Preview; Right two-thirds = Upload & Results)
    col_left, col_right = st.columns([1, 2], gap="large")

    # =========================================================================
    # LEFT COLUMN (1/3): Dataset, Project Description / Full Photo Preview, Quick Examples
    # =========================================================================
    with col_left:
        st.markdown('<h2 class="section-heading">Dataset</h2>', unsafe_allow_html=True)

        # Independent Testing Blind Selector (1,041 Casting Test + 17 Unsupported Test)
        subset_choice = st.radio(
            "Held-out test split (ground-truth labels hidden):",
            options=[
                f"Casting test set ({len(catalog['casting']):,})",
                f"Unsupported test set ({len(catalog['unsupported'])})",
            ],
            horizontal=True,
            key="dataset_subset_radio",
        )

        if "Casting test set" in subset_choice:
            entries = catalog["casting"]
            map_by_label = {e["blind_label"]: e for e in entries}
            chosen_label = st.selectbox(
                "Select a held-out casting test image:",
                options=[e["blind_label"] for e in entries],
                index=None,
                placeholder="Choose or type ID / SHA prefix...",
                key=f"casting_select_{st.session_state['casting_select_key']}",
            )
            if chosen_label is not None:
                entry = map_by_label[chosen_label]
                expected_id = f"casting:{entry['sha256']}"
                if st.session_state["active_photo_id"] != expected_id:
                    img_bytes = read_test_image_bytes(entry)
                    set_active_photo(
                        img_bytes=img_bytes,
                        photo_id=expected_id,
                        caption=entry["blind_label"],
                        source_kind="dataset_casting",
                        internal_filename=entry["original_filename"],
                        dataset_entry=entry,
                    )
                    st.rerun()
        else:
            entries = catalog["unsupported"]
            map_by_label = {e["blind_label"]: e for e in entries}
            chosen_label = st.selectbox(
                "Select a held-out unsupported test image:",
                options=[e["blind_label"] for e in entries],
                index=None,
                placeholder="Choose from 17 unsupported test images...",
                key=f"unsupp_select_{st.session_state['unsupp_select_key']}",
            )
            if chosen_label is not None:
                entry = map_by_label[chosen_label]
                expected_id = f"unsupp:{entry['sha256']}"
                if st.session_state["active_photo_id"] != expected_id:
                    img_bytes = read_test_image_bytes(entry)
                    set_active_photo(
                        img_bytes=img_bytes,
                        photo_id=expected_id,
                        caption=entry["blind_label"],
                        source_kind="dataset_unsupported",
                        internal_filename=entry["original_filename"],
                        dataset_entry=entry,
                    )
                    st.rerun()

        # Held-out Downloads, Exact Split Manifests & Full Original Dataset Access
        with st.expander("Test downloads, split manifests & full dataset", expanded=False):
            sc = catalog["split_counts"]
            uc = catalog["unsupp_counts"]
            st.markdown(
                f"**Exact Project Split Manifest:** `{sc['train']:,}` train, `{sc['validation']:,}` validation, "
                f"`{sc['test']:,}` test (`{sum(sc.values()):,}` casting images). "
                f"Unsupported set: `{uc['train']}` train, `{uc['validation']}` val, `{uc['test']}` test."
            )

            if CASTING_HELD_OUT_ZIP.is_file():
                st.download_button(
                    label="Download casting held-out test set (ZIP + CSV)",
                    data=lambda: CASTING_HELD_OUT_ZIP.read_bytes(),
                    file_name="casting_held_out_test.zip",
                    mime="application/zip",
                    on_click="ignore",
                    use_container_width=True,
                )
                st.download_button(
                    label="Download casting test labels CSV",
                    data=lambda: get_ground_truth_csv_bytes("casting"),
                    file_name="casting_test_ground_truth.csv",
                    mime="text/csv",
                    on_click="ignore",
                    use_container_width=True,
                )

            if UNSUPP_HELD_OUT_ZIP.is_file():
                st.download_button(
                    label="Download unsupported held-out test set (ZIP + CSV)",
                    data=lambda: UNSUPP_HELD_OUT_ZIP.read_bytes(),
                    file_name="unsupported_held_out_test.zip",
                    mime="application/zip",
                    on_click="ignore",
                    use_container_width=True,
                )
                st.download_button(
                    label="Download unsupported test labels CSV",
                    data=lambda: get_ground_truth_csv_bytes("unsupported"),
                    file_name="unsupported_test_ground_truth.csv",
                    mime="text/csv",
                    on_click="ignore",
                    use_container_width=True,
                )

            if SPLIT_MANIFEST_PATH.is_file():
                st.download_button(
                    label="Download project split manifest (7,284 rows CSV)",
                    data=lambda: SPLIT_MANIFEST_PATH.read_bytes(),
                    file_name="split_manifest.csv",
                    mime="text/csv",
                    on_click="ignore",
                    use_container_width=True,
                )

            if UNSUPP_MANIFEST_PATH.is_file():
                st.download_button(
                    label="Download unsupported manifest (90 rows CSV)",
                    data=lambda: UNSUPP_MANIFEST_PATH.read_bytes(),
                    file_name="unsupported_manifest.csv",
                    mime="text/csv",
                    on_click="ignore",
                    use_container_width=True,
                )

            st.markdown(
                "**Full Original Casting Dataset:** "
                "[Kaggle Casting Product Image Data](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) "
                "by Raviraj Joshi (**CC BY-NC-ND 4.0**). "
                "Unsupported photos come from Wikimedia Commons under individual Creative Commons / Public Domain licenses "
                "(per-image attribution included inside `unsupported_held_out_test.zip`)."
            )
            if ORIG_CASTING_ZIP.is_file():
                st.download_button(
                    label="Download full original casting_data.zip (69.4 MB)",
                    data=lambda: ORIG_CASTING_ZIP.read_bytes(),
                    file_name="casting_data.zip",
                    mime="application/zip",
                    on_click="ignore",
                    use_container_width=True,
                )

        # Project Description (before selection) OR Full Photo Preview (when photo selected)
        if st.session_state["active_bytes"] is None:
            st.markdown(
                """
                <div class="description-box">
                  <strong>About this inspector</strong><br>
                  This two-stage visual quality control tool checks whether an uploaded image is a top-view
                  cast pump impeller and then scores casting defect likelihood using locked validation cutoffs.
                  Select any blind held-out image above or upload your own photo under <strong>Upload a photo</strong> to begin.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            try:
                pil_preview = Image.open(io.BytesIO(st.session_state["active_bytes"]))
                st.image(
                    pil_preview,
                    caption=st.session_state["active_caption"] or "Selected photo",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Could not render preview image: {e}")

            if st.button("Clear photo", type="secondary", use_container_width=True):
                clear_active_photo()
                st.rerun()

        # Secondary Quick Examples section — clearly separate from held-out test sets
        st.markdown("<hr style='margin: 14px 0 8px 0; border-top: 1px solid #cbd5e1;'>", unsafe_allow_html=True)
        with st.expander("Quick examples (4 preset demo images)", expanded=False):
            st.markdown(
                "Preset shortcuts for quick exploration, kept separate from the full 1,041 + 17 held-out test sets above."
            )
            for q_key, q_info in QUICK_EXAMPLES.items():
                if st.button(q_info["label"], key=f"quick_btn_{q_key}", type="secondary", use_container_width=True):
                    if q_info["path"].is_file():
                        set_active_photo(
                            img_bytes=q_info["path"].read_bytes(),
                            photo_id=f"quick:{q_info['id']}",
                            caption=f"Quick example: {q_info['filename']}",
                            source_kind="quick",
                            internal_filename=q_info["filename"],
                        )
                        st.rerun()

    # =========================================================================
    # RIGHT COLUMN (2/3): Upload a photo, Step 1 Result, Step 2 Result, Step 3 Gemini
    # =========================================================================
    with col_right:
        st.markdown('<h2 class="section-heading">Upload a photo</h2>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload a photo",
            type=["jpg", "jpeg", "png"],
            key=f"photo_uploader_{st.session_state['uploader_key']}",
            label_visibility="collapsed",
        )

        if uploaded is not None:
            up_bytes = uploaded.getvalue()
            up_sha = hashlib.sha256(up_bytes).hexdigest()
            expected_up_id = f"upload:{uploaded.name}:{up_sha}"
            if st.session_state["active_photo_id"] != expected_up_id:
                set_active_photo(
                    img_bytes=up_bytes,
                    photo_id=expected_up_id,
                    caption=f"Uploaded: {uploaded.name}",
                    source_kind="upload",
                    internal_filename=uploaded.name,
                    dataset_entry=None,
                )
                st.rerun()
        elif st.session_state["active_source_kind"] == "upload":
            # User clicked the uploader clear (x) button -> clear photo and restore description
            clear_active_photo()
            st.rerun()

        # Check active inspection workflow
        elig = st.session_state["eligibility_result"]

        if st.session_state["active_bytes"] is None or elig is None:
            st.markdown(
                """
                <div class="panel-box">
                  <strong>Step 1: Photo Eligibility Check</strong><br>
                  Upload a photo above or select a held-out test image from <strong>Dataset</strong> to begin.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # -----------------------------------------------------------------
            # STEP 1 RESULT: Eligibility Check ONLY
            # -----------------------------------------------------------------
            st.markdown('<h3 class="subsection-heading">Step 1 Result</h3>', unsafe_allow_html=True)

            if not elig["accepted"]:
                # Rejected at Step 1 -> Show exact rejection prompt, hide defect & Gemini buttons
                st.markdown(
                    """
                    <div class="step-box">
                      <span class="verdict-badge verdict-rejected">Unsupported Image</span>
                      <p style="margin: 0; font-size: 16px; color: #000000;">
                        Please upload a clear top-view photo of a pump impeller.
                      </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                testing_complete = True
            else:
                # Accepted at Step 1 -> Show single line + "Check for defects" button
                st.markdown(
                    """
                    <div class="step-box">
                      <p style="margin: 0; font-size: 16px; font-weight: 600; color: #000000;">
                        This photo can be checked for defects.
                      </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("Check for defects", type="primary", key="btn_check_defects"):
                    defect_res = pipeline.check_defects(
                        st.session_state["active_bytes"],
                        filename=st.session_state["active_internal_name"],
                        gatekeeper_result=elig,
                    )
                    st.session_state["defect_result"] = defect_res
                    st.session_state["gemini_result"] = None
                    st.rerun()

                testing_complete = st.session_state["defect_result"] is not None

            # -----------------------------------------------------------------
            # STEP 2 RESULT: Defect Inspection (only shown after button click)
            # -----------------------------------------------------------------
            defect_res = st.session_state["defect_result"]
            if elig["accepted"] and defect_res is not None:
                st.markdown('<h3 class="subsection-heading">Step 2 Result</h3>', unsafe_allow_html=True)

                action_label = defect_res["action_label"]
                next_action = defect_res["next_action"]
                det = defect_res["defect_detector"]
                p_ok = det["ok_probability"]
                p_def = det["defective_probability"]

                if action_label == "Pass Candidate":
                    badge_cls = "verdict-pass"
                elif action_label == "Manual Review":
                    badge_cls = "verdict-review"
                else:
                    badge_cls = "verdict-hold"

                st.markdown(
                    f"""
                    <div class="result-card">
                      <div>
                        <span class="verdict-badge {badge_cls}">{action_label}</span>
                      </div>
                      <div class="next-action-block">
                        <strong>What to do next</strong><br>
                        {next_action}
                      </div>
                      <div style="font-size: 14px; font-weight: 700; color: #000000; margin-top: 4px;">
                        Model scores
                      </div>
                      <div class="scores-row">
                        <div class="score-tile">
                          <div class="score-tile-label">OK score</div>
                          <div class="score-tile-value">{p_ok:.4f}</div>
                        </div>
                        <div class="score-tile">
                          <div class="score-tile-label">Defect score</div>
                          <div class="score-tile-value">{p_def:.4f}</div>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Collapsed Thresholds and Diagnostic Details
                with st.expander("Thresholds and diagnostic details", expanded=False):
                    p_supp = elig.get("supported_probability", 0.0)
                    gate_tau = elig.get("threshold", 0.0049)
                    tau_pass = det.get("pass_threshold", 0.50)
                    tau_hold = det.get("hold_threshold", 0.76)

                    st.markdown(
                        f"""
                        <table class="details-table">
                          <tr>
                            <th>Stage</th>
                            <th>Model</th>
                            <th>Model Scores</th>
                            <th>Decision Thresholds</th>
                            <th>Outcome</th>
                          </tr>
                          <tr>
                            <td>Stage 1: Eligibility</td>
                            <td>MobileNetV2 Input Checker</td>
                            <td>Supported score: {p_supp:.4f}</td>
                            <td>Accept &ge; {gate_tau:.4f}</td>
                            <td>Accepted</td>
                          </tr>
                          <tr>
                            <td>Stage 2: Defect Inspection</td>
                            <td>MobileNetV2 Defect Classifier</td>
                            <td>OK score: {p_ok:.4f}<br>Defect score: {p_def:.4f}</td>
                            <td>Pass Candidate &lt; {tau_pass:.2f}<br>Manual Review [{tau_pass:.2f}, {tau_hold:.2f})<br>Hold &ge; {tau_hold:.2f}</td>
                            <td>{action_label}</td>
                          </tr>
                        </table>
                        """,
                        unsafe_allow_html=True,
                    )

                # -------------------------------------------------------------
                # STEP 3: Gemini Explanation (called ONLY when button is clicked)
                # -------------------------------------------------------------
                if st.button("Explain this result with Gemini", type="primary", key="btn_explain_gemini"):
                    with st.spinner("Generating plain-English explanation..."):
                        st.session_state["gemini_call_count"] += 1
                        st.session_state["gemini_result"] = generate_gemini_qa_summary(defect_res)
                    st.rerun()

                g_res = st.session_state["gemini_result"]
                if g_res is not None:
                    reason_html = (
                        f"<p style='margin: 8px 0 0 0; font-size: 14px; color: #000000;'><em>{g_res['reason']}</em></p>"
                        if g_res.get("reason")
                        else ""
                    )
                    st.markdown(
                        f"""
                        <div class="gemini-box">
                          <span class="gemini-source-tag">{g_res['label']}</span>
                          <div style="font-size: 16px; color: #000000; line-height: 1.5;">
                            {g_res['summary']}
                          </div>
                          {reason_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # -----------------------------------------------------------------
            # Ground-Truth Reveal for Independent Held-Out Dataset Testing
            # (Only offered AFTER testing is complete for dataset images)
            # -----------------------------------------------------------------
            entry = st.session_state.get("active_dataset_entry")
            if entry is not None and testing_complete:
                st.markdown("<hr style='margin: 16px 0 10px 0; border-top: 1px solid #cbd5e1;'>", unsafe_allow_html=True)
                if not st.session_state["label_revealed"]:
                    if st.button("Reveal actual ground-truth label", type="secondary", key="btn_reveal_label"):
                        st.session_state["label_revealed"] = True
                        st.rerun()
                else:
                    src_url = entry.get("source_url", "")
                    url_line = f"<br><strong>Source URL:</strong> <a href='{src_url}' target='_blank'>{src_url}</a>" if src_url else ""
                    st.markdown(
                        f"""
                        <div class="panel-box">
                          <strong>Ground-Truth Record (Revealed after testing)</strong><br>
                          <strong>Actual label:</strong> {entry['actual_label']}<br>
                          <strong>Original dataset path:</strong> <code>{entry['original_path']}</code><br>
                          <strong>Group ID:</strong> <code>{entry['group_id']}</code> | <strong>SHA-256:</strong> <code>{entry['sha256'][:16]}...</code><br>
                          <strong>Attribution:</strong> {entry['creator']} — {entry['license']} ({entry['source']}){url_line}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    from streamlit.runtime import exists as st_runtime_exists

    if not st_runtime_exists():
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
    else:
        main()
