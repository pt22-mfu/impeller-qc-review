"""Pump Impeller Quality Inspector - Streamlit Web Application.

Sequential Two-Stage Machine Vision Pipeline:
1. Input Checker Gatekeeper (tau = 0.0049)
2. Defect Inspection Classifier (Pass < 0.50, Hold >= 0.76)
With Gemini QA Summary integration & dual-model fallback.

Run command:
    streamlit run app.py
"""
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

from PIL import Image
import streamlit as st
from pipeline import ImpellerQCPipeline

# ---------------------------------------------------------------------------
# Page Configuration & Theming
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pump Impeller Quality Inspector",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

CUSTOM_CSS = """
<style>
  :root {
    --bg-page: #f1f5f9;
    --bg-card: #ffffff;
    --border-card: #e2e8f0;
    --text-black: #000000;
    --text-muted: #475569;
    --teal-primary: #0d9488;
    --teal-hover: #0f766e;
    --teal-light: #ccfbf1;
    --color-pass: #16a34a;
    --bg-pass: #dcfce7;
    --color-review: #d97706;
    --bg-review: #fef3c7;
    --color-hold: #dc2626;
    --bg-hold: #fee2e2;
    --color-blocked: #64748b;
    --bg-blocked: #f1f5f9;
  }

  /* Global typography and background */
  .stApp {
    background-color: var(--bg-page);
    color: var(--text-black);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 16px;
    line-height: 1.5;
  }

  /* Header Typography */
  .app-header {
    margin-bottom: 24px;
    padding-top: 8px;
  }

  h1.page-title {
    font-size: 32px;
    font-weight: 700;
    color: var(--text-black);
    letter-spacing: -0.02em;
    margin: 0 0 6px 0;
  }

  p.main-instruction {
    font-size: 16px;
    color: var(--text-muted);
    font-weight: 500;
    margin: 0;
  }

  /* White Card styling */
  .qc-card {
    background-color: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }

  .section-heading {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-black);
    margin: 0 0 16px 0;
  }

  /* Streamlit Button Tweaks */
  div.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    transition: all 0.15s ease-in-out;
  }

  div.stButton > button[kind="primary"] {
    background-color: var(--teal-primary) !important;
    color: #ffffff !important;
    border: none !important;
  }

  div.stButton > button[kind="primary"]:hover {
    background-color: var(--teal-hover) !important;
  }

  /* Status Badges */
  .status-badge {
    display: inline-flex;
    align-items: center;
    padding: 8px 16px;
    border-radius: 9999px;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 16px;
  }

  .badge-pass {
    background-color: var(--bg-pass);
    color: var(--color-pass);
    border: 1px solid #86efac;
  }

  .badge-review {
    background-color: var(--bg-review);
    color: var(--color-review);
    border: 1px solid #fde68a;
  }

  .badge-hold {
    background-color: var(--bg-hold);
    color: var(--color-hold);
    border: 1px solid #fca5a5;
  }

  .badge-blocked {
    background-color: var(--bg-blocked);
    color: var(--color-blocked);
    border: 1px solid #cbd5e1;
  }

  /* Action Banner */
  .action-callout {
    padding: 16px 20px;
    border-radius: 8px;
    margin: 16px 0;
  }

  .action-callout-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700;
    margin-bottom: 4px;
  }

  .action-callout-text {
    font-size: 18px;
    font-weight: 700;
  }

  .callout-pass {
    background-color: #ecfdf5;
    border-left: 4px solid var(--color-pass);
    color: #15803d;
  }

  .callout-review {
    background-color: #fffbeb;
    border-left: 4px solid var(--color-review);
    color: #b45309;
  }

  .callout-hold {
    background-color: #fef2f2;
    border-left: 4px solid var(--color-hold);
    color: #b91c1c;
  }

  .callout-blocked {
    background-color: #f8fafc;
    border-left: 4px solid #64748b;
    color: #334155;
  }

  /* Details Table */
  .details-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    margin-top: 8px;
  }

  .details-table th, .details-table td {
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid #e2e8f0;
  }

  .details-table th {
    color: #64748b;
    font-weight: 600;
  }

  .details-table td {
    color: #000000;
  }

  /* QA Summary Box */
  .qa-summary-card {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 18px;
    margin-top: 16px;
  }

  .qa-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 10px;
  }

  .badge-gemini {
    background-color: var(--teal-light);
    color: var(--teal-hover);
    border: 1px solid #99f6e4;
  }

  .badge-local {
    background-color: #e2e8f0;
    color: #334155;
    border: 1px solid #cbd5e1;
  }

  .qa-text {
    font-size: 15px;
    color: #000000;
    line-height: 1.5;
  }

  .empty-preview {
    border: 2px dashed #cbd5e1;
    border-radius: 8px;
    padding: 24px;
    text-align: center;
    color: var(--text-muted);
    font-size: 14px;
  }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Pipeline Resource Loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Inspection Models...")
def load_pipeline():
    """Load and cache the two-stage inspection pipeline."""
    return ImpellerQCPipeline()

pipeline = load_pipeline()

# ---------------------------------------------------------------------------
# Sample Assets Definition
# ---------------------------------------------------------------------------
SAMPLES = {
    "unsupported": {
        "id": "sample_unsupported",
        "btn_label": "🚫 Unsupported part",
        "path": Path("sample_assets") / "sample_unsupported.jpg",
        "filename": "sample_unsupported.jpg",
        "expected": "Unsupported Image"
    },
    "pass": {
        "id": "sample_pass",
        "btn_label": "✅ Pass Candidate",
        "path": Path("sample_assets") / "sample_pass.jpeg",
        "filename": "sample_pass.jpeg",
        "expected": "Pass Candidate"
    },
    "review": {
        "id": "sample_review",
        "btn_label": "⚠️ Manual Review",
        "path": Path("sample_assets") / "sample_review.jpeg",
        "filename": "sample_review.jpeg",
        "expected": "Manual Review"
    },
    "hold": {
        "id": "sample_hold",
        "btn_label": "🛑 Hold",
        "path": Path("sample_assets") / "sample_hold.jpeg",
        "filename": "sample_hold.jpeg",
        "expected": "Hold"
    }
}

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
    """Load Gemini configuration with environment variables taking precedence over Streamlit secrets.
    
    Order of precedence:
    1. Environment variables:
       - GEMINI_API_KEY
       - GEMINI_MODEL (default: gemini-3.5-flash-lite)
       - GEMINI_FALLBACK_MODEL (default: gemini-3.1-flash-lite)
    2. st.secrets / .streamlit/secrets.toml:
       - GEMINI_API_KEY / gemini_api_key / api_key / gemini.api_key
       - GEMINI_MODEL / model / model_id / primary_model
       - GEMINI_FALLBACK_MODEL / fallback_model / fallback_model_id
    """
    secrets_dict = {}
    # Check st.secrets
    try:
        if hasattr(st, "secrets") and st.secrets is not None:
            secrets_dict = dict(st.secrets)
    except Exception:
        pass

    # Fallback to direct secrets.toml read if st.secrets is empty
    secrets_path = Path(".streamlit") / "secrets.toml"
    if not secrets_dict and secrets_path.exists():
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

    # 1. API Key
    api_key = os.environ.get("GEMINI_API_KEY")
    key_src = "environment" if api_key else None
    if not api_key:
        api_key = lookup_secret(["GEMINI_API_KEY", "gemini_api_key", "api_key", "key"])
        if api_key:
            key_src = "streamlit_secrets"

    # 2. Primary Model ID
    primary_model = os.environ.get("GEMINI_MODEL") or os.environ.get("GEMINI_PRIMARY_MODEL")
    if not primary_model:
        primary_model = lookup_secret(["GEMINI_MODEL", "model", "model_id", "primary_model"])
    if not primary_model:
        primary_model = "gemini-3.5-flash-lite"

    # 3. Fallback Model ID
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
        "key_source": key_src or "none"
    }

def generate_local_summary(data):
    """Generate a deterministic, rule-governed Local QA summary based strictly on the existing QC decision."""
    action_label = data.get("action_label", "Pass Candidate")
    detector = data.get("defect_detector", {})
    p_def = detector.get("defective_probability")
    next_action = data.get("next_action", "")
    tau_pass = detector.get("pass_threshold", 0.50)
    tau_hold = detector.get("hold_threshold", 0.76)

    score_str = f"{p_def:.4f}" if p_def is not None else "N/A"

    if action_label == "Pass Candidate":
        return (
            f"Local summary: Defect score {score_str} is below {tau_pass:.2f} threshold. "
            f"Pass candidate. Follow your normal release procedure."
        )
    elif action_label == "Manual Review":
        return (
            f"Local summary: Defect score {score_str} is in the review range [{tau_pass:.2f}, {tau_hold:.2f}). "
            f"Conduct visual inspection at review station."
        )
    elif action_label == "Hold":
        return (
            f"Local summary: Defect score {score_str} meets or exceeds {tau_hold:.2f} threshold. "
            f"Hold for supervisor review."
        )
    else:
        return f"Local summary: Disposition {action_label}. Prescribed next action: {next_action}."

def build_gemini_prompt(data):
    """Construct a strict, operator-facing prompt for Gemini based solely on QC scores."""
    action_label = data.get("action_label", "Unknown")
    next_action = data.get("next_action", "")
    p_supp = data.get("gatekeeper", {}).get("supported_probability", 0.0)
    p_def = data.get("defect_detector", {}).get("defective_probability", 0.0)
    tau_pass = data.get("defect_detector", {}).get("pass_threshold", 0.50)
    tau_hold = data.get("defect_detector", {}).get("hold_threshold", 0.76)

    if action_label == "Pass Candidate":
        action_instruction = "Use the exact phrase: 'Pass candidate. Follow your normal release procedure.'"
    elif action_label == "Hold":
        action_instruction = "Use the exact phrase: 'Hold for supervisor review.'"
    else:
        action_instruction = f"Instruct the operator: '{next_action}'."

    return (
        f"You are an industrial Quality Assurance assistant for pump impeller manufacturing.\n"
        f"Write a concise, professional 2-sentence operator QA summary based strictly on these inspection results:\n"
        f"- Gatekeeper Status: Supported Impeller (P_supported = {p_supp:.4f})\n"
        f"- Defect Inspection Score: {p_def:.4f} (Pass cutoff: < {tau_pass:.2f}, Hold cutoff: >= {tau_hold:.2f})\n"
        f"- Routing Decision: {action_label}\n"
        f"- Prescribed Next Action: {next_action}\n\n"
        f"Instructions:\n"
        f"1. State the routing decision ({action_label}) and defect probability score ({p_def:.4f}).\n"
        f"2. {action_instruction}\n"
        f"3. Strictly avoid wording that claims quality standards were verified or that a defect was confirmed.\n"
        f"4. Do NOT invent defect names (such as porosity, cracks, etc.).\n"
        f"5. Keep the entire response under 40 words."
    )

def _call_gemini_api(model_name, api_key, prompt, timeout=10, simulate=None):
    """Call Google Gemini generateContent endpoint for a specific model."""
    if simulate:
        sim = str(simulate).lower()
        if sim in ["404", "model_unavailable", "not_found"]:
            raise ModelUnavailableError(f"Simulated model unavailability (HTTP 404): models/{model_name} not found")
        elif sim in ["429", "quota", "rate_limit"]:
            raise QuotaLimitError(f"Simulated quota limit reached (HTTP 429): Resource exhausted for {model_name}")
        elif sim in ["500", "503", "service_error", "temporary_error"]:
            raise TemporaryServiceError(f"Simulated temporary service error (HTTP 503): Service unavailable for {model_name}")
        elif sim in ["timeout"]:
            raise TemporaryServiceError(f"Simulated network timeout connecting to {model_name}")
        elif sim in ["success", "mock_success"]:
            return f"QA summary ({model_name}): Part disposition complies with QC threshold policy."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 100
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
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
            else:
                raise TemporaryServiceError(f"Unexpected HTTP status {resp.status}")
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="ignore")
        if e.code == 404:
            raise ModelUnavailableError(f"Model unavailable (HTTP 404): models/{model_name} not found. Details: {err_msg[:120]}")
        elif e.code == 429:
            raise QuotaLimitError(f"Model quota limit exceeded (HTTP 429): {err_msg[:120]}")
        elif e.code in [500, 502, 503, 504]:
            raise TemporaryServiceError(f"Temporary service error (HTTP {e.code}): {err_msg[:120]}")
        elif e.code == 400 and ("not found" in err_msg.lower() or "not supported" in err_msg.lower()):
            raise ModelUnavailableError(f"Model unavailable (HTTP 400): {err_msg[:120]}")
        else:
            raise RuntimeError(f"Gemini API HTTP {e.code}: {err_msg[:120]}")
    except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
        raise TemporaryServiceError(f"Temporary network/service error: {e}")

def generate_gemini_qa_summary(data, api_key=None, config_override=None, simulation=None):
    """Generate an operator QA summary using Gemini with model fallback,
    or display a clearly labeled Local summary based on the existing QC decision if neither model can answer.
    """
    config = config_override or get_gemini_config()
    key = api_key or config.get("api_key")

    primary_model = config.get("primary_model", "gemini-3.5-flash-lite")
    fallback_model = config.get("fallback_model", "gemini-3.1-flash-lite")

    # If key is missing, neither model can answer -> show Local summary
    if not key:
        local_text = generate_local_summary(data)
        return {
            "success": True,
            "source": "local",
            "is_local": True,
            "label": "Local summary",
            "summary": local_text,
            "model_used": None,
            "fallback_used": False,
            "primary_model": primary_model,
            "fallback_model": fallback_model,
            "reason": "GEMINI_API_KEY is not configured in environment or Streamlit Secrets.",
            "error": None
        }

    prompt = build_gemini_prompt(data)

    primary_sim = None
    fallback_sim = None
    if isinstance(simulation, dict):
        primary_sim = simulation.get("primary")
        fallback_sim = simulation.get("fallback")
    elif isinstance(simulation, str):
        if simulation in ["primary_404", "primary_unavailable"]:
            primary_sim = "404"
        elif simulation in ["primary_429", "primary_quota"]:
            primary_sim = "429"
        elif simulation in ["primary_503", "primary_service_error"]:
            primary_sim = "503"
        elif simulation in ["both_fail", "both_failed"]:
            primary_sim = "429"
            fallback_sim = "503"
        elif simulation in ["mock_success"]:
            primary_sim = "mock_success"

    # Step 1: Attempt Primary Model
    primary_error = None
    should_fallback = False
    try:
        summary_text = _call_gemini_api(primary_model, key, prompt, simulate=primary_sim)
        return {
            "success": True,
            "source": "gemini",
            "is_local": False,
            "label": f"Gemini ({primary_model})",
            "summary": summary_text,
            "model_used": primary_model,
            "fallback_used": False,
            "primary_model": primary_model,
            "error": None
        }
    except (ModelUnavailableError, QuotaLimitError, TemporaryServiceError) as e:
        primary_error = str(e)
        should_fallback = True
    except Exception as e:
        primary_error = str(e)
        should_fallback = True

    # Step 2: Try fallback model ONCE
    fallback_error = None
    if should_fallback:
        try:
            summary_text = _call_gemini_api(fallback_model, key, prompt, simulate=fallback_sim)
            return {
                "success": True,
                "source": "gemini",
                "is_local": False,
                "label": f"Gemini ({fallback_model})",
                "summary": summary_text,
                "model_used": fallback_model,
                "fallback_used": True,
                "primary_model": primary_model,
                "primary_error": primary_error,
                "fallback_model": fallback_model,
                "error": None
            }
        except Exception as e:
            fallback_error = str(e)

    # Step 3: Neither model could answer -> Local summary fallback
    local_text = generate_local_summary(data)
    return {
        "success": True,
        "source": "local",
        "is_local": True,
        "label": "Local summary",
        "summary": local_text,
        "model_used": None,
        "fallback_used": True,
        "primary_model": primary_model,
        "primary_error": primary_error,
        "fallback_model": fallback_model,
        "fallback_error": fallback_error,
        "reason": f"Neither model could answer. Primary ({primary_model}): {primary_error}; Fallback ({fallback_model}): {fallback_error}",
        "error": None
    }

# ---------------------------------------------------------------------------
# Streamlit Application State
# ---------------------------------------------------------------------------
if "active_bytes" not in st.session_state:
    st.session_state["active_bytes"] = None
if "active_filename" not in st.session_state:
    st.session_state["active_filename"] = None
if "active_source" not in st.session_state:
    st.session_state["active_source"] = None
if "qa_summary" not in st.session_state:
    st.session_state["qa_summary"] = None
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

def select_image(img_bytes, filename, source="sample"):
    st.session_state["active_bytes"] = img_bytes
    st.session_state["active_filename"] = filename
    st.session_state["active_source"] = source
    # Clear any previous QA summary on image change
    st.session_state["qa_summary"] = None

# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------
def main():
    # Header Section
    st.markdown("""
    <div class="app-header">
      <h1 class="page-title">Pump Impeller Quality Inspector</h1>
      <p class="main-instruction">Upload a top-view photo of a pump impeller</p>
    </div>
    """, unsafe_allow_html=True)

    # Two-Column Layout (stacks on mobile)
    col_left, col_right = st.columns([5, 6], gap="large")

    # =========================================================================
    # LEFT COLUMN: Upload, Samples & Preview
    # =========================================================================
    with col_left:
        # Card 1: Image Input
        st.markdown('<div class="qc-card"><h2 class="section-heading">Image Input</h2>', unsafe_allow_html=True)
        
        uploaded = st.file_uploader(
            "Upload an image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            key=f"file_uploader_{st.session_state['uploader_key']}",
            help="Select a top-view photograph of a cast pump impeller."
        )

        if uploaded is not None:
            new_bytes = uploaded.getvalue()
            # If the user uploaded a new file, update active image
            if st.session_state.get("uploaded_name") != uploaded.name or st.session_state.get("active_bytes") != new_bytes:
                st.session_state["uploaded_name"] = uploaded.name
                select_image(new_bytes, uploaded.name, source="upload")
                st.rerun()

        # Try a sample section
        st.markdown('<p style="font-weight: 600; font-size: 14px; margin: 16px 0 8px 0; color: #475569;">Try a sample</p>', unsafe_allow_html=True)
        
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            if st.button("🚫 Unsupported part", use_container_width=True):
                info = SAMPLES["unsupported"]
                if info["path"].exists():
                    select_image(info["path"].read_bytes(), info["filename"], source="sample_unsupported")
                    st.rerun()

        with row1_col2:
            if st.button("✅ Pass Candidate", use_container_width=True):
                info = SAMPLES["pass"]
                if info["path"].exists():
                    select_image(info["path"].read_bytes(), info["filename"], source="sample_pass")
                    st.rerun()

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            if st.button("⚠️ Manual Review", use_container_width=True):
                info = SAMPLES["review"]
                if info["path"].exists():
                    select_image(info["path"].read_bytes(), info["filename"], source="sample_review")
                    st.rerun()

        with row2_col2:
            if st.button("🛑 Hold", use_container_width=True):
                info = SAMPLES["hold"]
                if info["path"].exists():
                    select_image(info["path"].read_bytes(), info["filename"], source="sample_hold")
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Card 2: Image Preview
        st.markdown('<div class="qc-card"><h2 class="section-heading">Image Preview</h2>', unsafe_allow_html=True)
        if st.session_state["active_bytes"]:
            try:
                img = Image.open(io.BytesIO(st.session_state["active_bytes"]))
                st.image(img, caption=f"Active: {st.session_state['active_filename']}", use_container_width=True)
            except Exception as e:
                st.error(f"Failed to display image: {e}")
        else:
            st.markdown('<div class="empty-preview">No image selected.<br>Upload an image or click a sample above.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # RIGHT COLUMN: Results, Next Action, QA Summary & Details
    # =========================================================================
    with col_right:
        if not st.session_state["active_bytes"]:
            st.markdown("""
            <div class="qc-card" style="text-align: center; padding: 48px 24px;">
              <h2 class="section-heading" style="color: #64748b;">Ready for Inspection</h2>
              <p style="color: #64748b; font-size: 15px;">
                Upload a top-view photo of a pump impeller or select one of the sample images on the left to view the automated quality control decision.
              </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Run two-stage inference
            inspection = pipeline.inspect(
                st.session_state["active_bytes"],
                filename=st.session_state["active_filename"]
            )
            action_label = inspection.get("action_label", "Unknown")
            status = inspection.get("status", "BLOCKED")
            next_action = inspection.get("next_action", "")
            gatekeeper = inspection.get("gatekeeper", {})
            defect_det = inspection.get("defect_detector", {})

            # Map status badge classes
            if status == "BLOCKED":
                badge_class = "badge-blocked"
                callout_class = "callout-blocked"
                badge_icon = "🚫"
            elif action_label == "Pass Candidate":
                badge_class = "badge-pass"
                callout_class = "callout-pass"
                badge_icon = "✅"
            elif action_label == "Manual Review":
                badge_class = "badge-review"
                callout_class = "callout-review"
                badge_icon = "⚠️"
            elif action_label == "Hold":
                badge_class = "badge-hold"
                callout_class = "callout-hold"
                badge_icon = "🛑"
            else:
                badge_class = "badge-blocked"
                callout_class = "callout-blocked"
                badge_icon = "ℹ️"

            # Primary Decision Card
            st.markdown(f"""
            <div class="qc-card">
              <h2 class="section-heading">Inspection Result</h2>
              <div>
                <span class="status-badge {badge_class}">
                  {badge_icon} {action_label}
                </span>
              </div>
              <div class="action-callout {callout_class}">
                <div class="action-callout-label">Prescribed Next Action</div>
                <div class="action-callout-text">{next_action}</div>
              </div>
              <p style="font-size: 14px; color: #475569; margin-top: 8px;">
                {inspection.get("message", "")}
              </p>
            </div>
            """, unsafe_allow_html=True)

            # Gemini QA Summary Card
            if status == "BLOCKED":
                st.markdown("""
                <div class="qc-card">
                  <h2 class="section-heading">QA Summary</h2>
                  <p style="color: #64748b; font-size: 14px;">
                    QA Summary is only available for supported impeller images. Please upload a supported top-view impeller photo to proceed with defect assessment.
                  </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="qc-card"><h2 class="section-heading">AI QA Summary</h2>', unsafe_allow_html=True)
                
                # Check if user clicked Generate QA Summary
                col_btn, col_info = st.columns([1, 1])
                with col_btn:
                    if st.button("Generate QA Summary", type="primary", use_container_width=True):
                        with st.spinner("Generating operator QA summary..."):
                            summary_res = generate_gemini_qa_summary(inspection)
                            st.session_state["qa_summary"] = summary_res

                # Display existing or newly generated QA Summary
                summary = st.session_state.get("qa_summary")
                if summary:
                    is_local = summary.get("is_local", False)
                    source_badge = "badge-local" if is_local else "badge-gemini"
                    source_label = summary.get("label", "Local summary")
                    summary_text = summary.get("summary", "")
                    reason = summary.get("reason")

                    reason_html = f'<p style="font-size: 12px; color: #64748b; margin-top: 8px; margin-bottom: 0;"><em>Note: {reason}</em></p>' if reason else ""

                    st.markdown(f"""
                    <div class="qa-summary-card">
                      <span class="qa-badge {source_badge}">{source_label}</span>
                      <div class="qa-text">{summary_text}</div>
                      {reason_html}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.caption("Click above to generate a concise, operator-facing QA summary.")

                st.markdown('</div>', unsafe_allow_html=True)

            # Collapsible Inspection Details Card
            with st.expander("Inspection details", expanded=False):
                p_supp = gatekeeper.get("supported_probability")
                supp_str = f"{p_supp:.4f}" if p_supp is not None else "N/A"
                gate_thresh = gatekeeper.get("threshold", 0.0049)

                p_def = defect_det.get("defective_probability")
                def_str = f"{p_def:.4f}" if p_def is not None else "Skipped (Blocked at Gatekeeper)"
                pass_thresh = defect_det.get("pass_threshold", 0.50)
                hold_thresh = defect_det.get("hold_threshold", 0.76)

                st.markdown(f"""
                <table class="details-table">
                  <tr>
                    <th>Pipeline Stage</th>
                    <th>Model</th>
                    <th>Model Score</th>
                    <th>Decision Threshold</th>
                    <th>Stage Outcome</th>
                  </tr>
                  <tr>
                    <td><strong>Stage 1: Gatekeeper</strong></td>
                    <td>MobileNetV2 Input Checker</td>
                    <td><code>P(supported) = {supp_str}</code></td>
                    <td>Accept &ge; {gate_thresh:.4f}</td>
                    <td>{gatekeeper.get("decision", "N/A")}</td>
                  </tr>
                  <tr>
                    <td><strong>Stage 2: Defect Inspector</strong></td>
                    <td>MobileNetV2 Defect Inspector</td>
                    <td><code>P(defective) = {def_str}</code></td>
                    <td>Pass &lt; {pass_thresh:.2f} | Hold &ge; {hold_thresh:.2f}</td>
                    <td>{defect_det.get("action", "Skipped")}</td>
                  </tr>
                </table>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    from streamlit.runtime import exists as st_runtime_exists
    if not st_runtime_exists():
        import sys
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
    else:
        main()
