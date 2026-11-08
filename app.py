"""Pump Impeller Quality Inspector - Interactive QC Demo.

Sequential Two-Stage Machine Vision Pipeline:
1. Input Checker Gatekeeper (tau = 0.0049)
2. Defect Inspection Classifier (Pass < 0.50, Hold >= 0.76)
With Gemini QA Summary integration.

Usage:
    .\\.venv\\Scripts\\python.exe app.py [--port 8080] [--host 127.0.0.1]
"""
import argparse
import base64
import io
import json
import os
import socket
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from pipeline import ImpellerQCPipeline

# Global pipeline and preloaded samples
pipeline = None
samples = {}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pump Impeller Quality Inspector</title>
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

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      font-size: 16px;
      color: var(--text-black);
      background-color: var(--bg-page);
      line-height: 1.5;
      padding: 32px 20px;
      min-height: 100vh;
    }

    .wrapper {
      max-width: 1100px;
      margin: 0 auto;
    }

    header {
      margin-bottom: 28px;
    }

    h1 {
      font-size: 32px;
      font-weight: 700;
      color: var(--text-black);
      letter-spacing: -0.02em;
      margin-bottom: 6px;
    }

    p.main-instruction {
      font-size: 16px;
      color: var(--text-muted);
      font-weight: 500;
    }

    /* Layout grid: left upload, right result */
    .layout-grid {
      display: grid;
      grid-template-columns: 440px 1fr;
      gap: 24px;
      align-items: start;
    }

    @media (max-width: 860px) {
      .layout-grid {
        grid-template-columns: 1fr;
      }
      h1 {
        font-size: 26px;
      }
    }

    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-card);
      border-radius: 12px;
      padding: 24px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    h2 {
      font-size: 20px;
      font-weight: 700;
      color: var(--text-black);
      margin-bottom: 16px;
    }

    /* Upload Dropzone */
    .dropzone {
      border: 2px dashed #cbd5e1;
      border-radius: 8px;
      padding: 24px 16px;
      text-align: center;
      cursor: pointer;
      background: #fafafa;
      transition: all 0.2s ease;
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: var(--teal-primary);
      background: #f0fdfa;
    }
    .dropzone-icon {
      font-size: 28px;
      margin-bottom: 6px;
      color: var(--teal-primary);
    }
    .dropzone-label {
      font-size: 16px;
      font-weight: 600;
      color: var(--text-black);
      margin-bottom: 4px;
    }
    .dropzone-sub {
      font-size: 14px;
      color: var(--text-muted);
    }
    #file-input { display: none; }

    /* Compact Image Preview */
    .preview-wrapper {
      margin-top: 16px;
    }
    .preview-box {
      border: 1px solid var(--border-card);
      border-radius: 8px;
      background: #f8fafc;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: height 0.2s ease;
    }
    .preview-box.compact {
      height: 48px;
    }
    .preview-box.loaded {
      height: auto;
      max-height: 320px;
      padding: 8px;
    }
    .preview-box img {
      max-width: 100%;
      max-height: 300px;
      object-fit: contain;
      border-radius: 4px;
    }
    .preview-empty-text {
      font-size: 14px;
      color: var(--text-muted);
    }

    /* Try a sample section */
    .sample-section {
      margin-top: 24px;
      border-top: 1px solid var(--border-card);
      padding-top: 20px;
    }
    .sample-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .sample-btn {
      background: #ffffff;
      border: 1px solid var(--border-card);
      padding: 10px 12px;
      border-radius: 8px;
      cursor: pointer;
      text-align: left;
      font-family: inherit;
      color: var(--text-black);
      transition: all 0.15s ease;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .sample-btn:hover {
      border-color: var(--teal-primary);
      background: #f0fdfa;
      transform: translateY(-1px);
    }
    .sample-btn .title {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-black);
    }
    .sample-btn .sub {
      font-size: 12px;
      color: var(--text-muted);
    }

    /* Results Column (Right) */
    .result-container {
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    /* Most Prominent Result Banner */
    .result-banner {
      border-radius: 12px;
      padding: 24px;
      border: 2px solid var(--border-card);
      background: #ffffff;
      transition: all 0.2s ease;
    }

    .result-tag-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
    }
    .status-badge {
      display: inline-block;
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 4px 10px;
      border-radius: 999px;
    }
    .badge-pass { background: var(--bg-pass); color: var(--color-pass); border: 1px solid var(--color-pass); }
    .badge-review { background: var(--bg-review); color: var(--color-review); border: 1px solid var(--color-review); }
    .badge-hold { background: var(--bg-hold); color: var(--color-hold); border: 1px solid var(--color-hold); }
    .badge-blocked { background: var(--bg-blocked); color: var(--color-blocked); border: 1px solid var(--color-blocked); }
    .badge-idle { background: #f1f5f9; color: var(--text-muted); border: 1px solid #cbd5e1; }

    .result-title {
      font-size: 26px;
      font-weight: 800;
      color: var(--text-black);
      margin-bottom: 10px;
      line-height: 1.2;
    }

    .next-action-box {
      margin-top: 12px;
      padding: 12px 14px;
      border-radius: 8px;
      font-size: 15px;
      font-weight: 600;
      line-height: 1.4;
    }
    .action-pass { background: var(--bg-pass); color: #14532d; border-left: 4px solid var(--color-pass); }
    .action-review { background: var(--bg-review); color: #78350f; border-left: 4px solid var(--color-review); }
    .action-hold { background: var(--bg-hold); color: #7f1d1d; border-left: 4px solid var(--color-hold); }
    .action-blocked { background: var(--bg-blocked); color: #334155; border-left: 4px solid var(--color-blocked); }

    .result-desc {
      font-size: 15px;
      color: var(--text-black);
      margin-top: 8px;
      line-height: 1.45;
    }

    /* Collapsed Inspection Details */
    details.inspection-details {
      margin-top: 16px;
      border-top: 1px solid var(--border-card);
      padding-top: 14px;
    }
    details.inspection-details summary {
      font-size: 15px;
      font-weight: 600;
      color: var(--teal-primary);
      cursor: pointer;
      user-select: none;
    }
    details.inspection-details summary:hover {
      text-decoration: underline;
    }
    .details-content {
      margin-top: 12px;
      padding: 14px;
      background: #f8fafc;
      border-radius: 8px;
      border: 1px solid var(--border-card);
      font-size: 14px;
      line-height: 1.6;
    }
    .detail-row {
      display: flex;
      justify-content: space-between;
      padding: 4px 0;
      border-bottom: 1px solid #edf2f7;
    }
    .detail-row:last-child { border-bottom: none; }
    .detail-label { color: var(--text-muted); }
    .detail-val { font-weight: 600; color: var(--text-black); font-family: monospace; }

    /* Gemini QA Summary Section */
    .gemini-section {
      margin-top: 20px;
      padding: 20px;
      background: #ffffff;
      border: 1px solid var(--border-card);
      border-radius: 12px;
    }
    .btn-teal {
      background-color: var(--teal-primary);
      color: #ffffff;
      border: none;
      padding: 10px 18px;
      border-radius: 8px;
      font-family: inherit;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .btn-teal:hover {
      background-color: var(--teal-hover);
    }
    .btn-teal:disabled {
      background-color: #94a3b8;
      cursor: not-allowed;
    }

    .qa-box {
      margin-top: 14px;
      padding: 16px;
      border-radius: 8px;
      font-size: 15px;
      color: var(--text-black);
      line-height: 1.5;
      background: #f8fafc;
      border: 1px solid #cbd5e1;
    }
    .qa-box.gemini {
      background: #f0fdfa;
      border: 1px solid #99f6e4;
      color: var(--text-black);
    }
    .qa-box.local {
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      color: var(--text-black);
    }
    .qa-box.error {
      background: #fff1f2;
      border-color: #fecdd3;
      color: #9f1239;
    }
    .qa-badge {
      display: inline-block;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 3px 8px;
      border-radius: 4px;
      margin-bottom: 8px;
    }
    .qa-badge.qa-badge-gemini {
      background: #ccfbf1;
      color: #0f766e;
    }
    .qa-badge.qa-badge-local {
      background: #e2e8f0;
      color: #334155;
    }
    .qa-badge.qa-badge-error {
      background: #ffe4e6;
      color: #9f1239;
    }
    .qa-title {
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--teal-primary);
      margin-bottom: 4px;
    }

    .spinner {
      display: inline-block;
      width: 14px;
      height: 14px;
      border: 2px solid rgba(255,255,255,0.4);
      border-radius: 50%;
      border-top-color: #ffffff;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="wrapper">
    <header>
      <h1>Pump Impeller Quality Inspector</h1>
      <p class="main-instruction">Upload a top-view photo of a pump impeller</p>
    </header>

    <div class="layout-grid">
      <!-- Left Column: Upload & Sample Controls -->
      <div class="card">
        <h2>Inspection Input</h2>

        <div class="dropzone" id="dropzone" onclick="document.getElementById('file-input').click()">
          <div class="dropzone-icon">&uarr;</div>
          <div class="dropzone-label">Click or drag an image here</div>
          <div class="dropzone-sub">Supports JPEG or PNG format</div>
        </div>
        <input type="file" id="file-input" accept="image/*" onchange="handleFileUpload(this.files[0])">

        <div class="preview-wrapper">
          <div class="preview-box compact" id="preview-box">
            <span class="preview-empty-text" id="preview-placeholder">No image selected</span>
            <img id="image-preview" src="" alt="Selected Impeller" style="display: none;">
          </div>
        </div>

        <div class="sample-section">
          <h2>Try a sample</h2>
          <div class="sample-grid">
            <button class="sample-btn" onclick="runSample('sample_unsupported')">
              <span class="title">🚫 Unsupported part</span>
              <span class="sub">Pump housing casing</span>
            </button>
            <button class="sample-btn" onclick="runSample('sample_pass')">
              <span class="title">✅ Pass Candidate</span>
              <span class="sub">Normal release procedure</span>
            </button>
            <button class="sample-btn" onclick="runSample('sample_review')">
              <span class="title">⚠️ Manual Review</span>
              <span class="sub">Station visual inspection</span>
            </button>
            <button class="sample-btn" onclick="runSample('sample_hold')">
              <span class="title">🛑 Hold</span>
              <span class="sub">Supervisor review</span>
            </button>
          </div>
        </div>
      </div>

      <!-- Right Column: Result and Next Action -->
      <div class="result-container">
        <!-- Most Prominent Result Banner -->
        <div class="card result-banner" id="result-banner">
          <div class="result-tag-row">
            <span class="status-badge badge-idle" id="status-badge">Awaiting Input</span>
            <span id="analyzing-spinner" style="display: none; font-size: 14px; color: var(--teal-primary);">
              <span class="spinner" style="border-top-color: var(--teal-primary);"></span> Analyzing...
            </span>
          </div>

          <div class="result-title" id="result-title">No Part Inspected</div>
          <div class="result-desc" id="result-desc">Upload a photo on the left or select a sample from "Try a sample" to begin inspection.</div>

          <div class="next-action-box action-blocked" id="next-action-box" style="display: none;">
            <strong>Prescribed Next Action:</strong> <span id="next-action-text">&mdash;</span>
          </div>

          <!-- Collapsed Inspection Details -->
          <details class="inspection-details" id="inspection-details" style="display: none;">
            <summary>Inspection details</summary>
            <div class="details-content">
              <div class="detail-row">
                <span class="detail-label">Gatekeeper Status:</span>
                <span class="detail-val" id="det-gate-status">&mdash;</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Supported Probability P(Supported):</span>
                <span class="detail-val" id="det-supp-prob">&mdash;</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Gatekeeper Threshold (&tau;):</span>
                <span class="detail-val" id="det-gate-thresh">0.0049</span>
              </div>
              <div class="detail-row" id="det-defect-row" style="display: none;">
                <span class="detail-label">Defect Probability P(Defective):</span>
                <span class="detail-val" id="det-def-prob">&mdash;</span>
              </div>
              <div class="detail-row" id="det-pass-row" style="display: none;">
                <span class="detail-label">Pass Threshold:</span>
                <span class="detail-val" id="det-pass-thresh">0.50</span>
              </div>
              <div class="detail-row" id="det-hold-row" style="display: none;">
                <span class="detail-label">Hold Threshold:</span>
                <span class="detail-val" id="det-hold-thresh">0.76</span>
              </div>
            </div>
          </details>
        </div>

        <!-- Gemini QA Summary Section (Visible only after accepted inspection) -->
        <div class="gemini-section" id="gemini-card" style="display: none;">
          <h2>AI QA Summary</h2>
          <p style="font-size: 15px; color: var(--text-muted); margin-bottom: 12px;">
            Generate an operator-facing summary synthesized by Gemini based on the verified QC decision and model scores.
          </p>

          <button class="btn-teal" id="qa-btn" onclick="generateQASummary()">
            <span>Generate QA Summary</span>
          </button>

          <div class="qa-box" id="qa-result-box" style="display: none;">
            <div class="qa-badge" id="qa-badge">QA Summary</div>
            <div id="qa-summary-text"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentInspection = null;

    // Drag and drop handlers
    const dropzone = document.getElementById('dropzone');
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    function setPreview(src) {
      const box = document.getElementById('preview-box');
      const img = document.getElementById('image-preview');
      const placeholder = document.getElementById('preview-placeholder');

      if (src) {
        img.src = src;
        img.style.display = 'block';
        placeholder.style.display = 'none';
        box.className = 'preview-box loaded';
      } else {
        img.src = '';
        img.style.display = 'none';
        placeholder.style.display = 'block';
        box.className = 'preview-box compact';
      }
    }

    function clearOldState() {
      // Clear Gemini summary when user picks a new image
      document.getElementById('gemini-card').style.display = 'none';
      const box = document.getElementById('qa-result-box');
      box.style.display = 'none';
      box.className = 'qa-box';
      const badge = document.getElementById('qa-badge');
      if (badge) {
        badge.className = 'qa-badge';
        badge.innerText = 'QA Summary';
      }
      document.getElementById('qa-summary-text').innerText = '';
      currentInspection = null;
    }

    async function runSample(sampleId) {
      clearOldState();
      document.getElementById('analyzing-spinner').style.display = 'inline-flex';
      setPreview('/samples/' + sampleId);

      try {
        const response = await fetch('/api/inspect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sample_id: sampleId })
        });
        const result = await response.json();
        currentInspection = result;
        renderResults(result);
      } catch (err) {
        alert('Inspection request failed: ' + err.message);
      } finally {
        document.getElementById('analyzing-spinner').style.display = 'none';
      }
    }

    async function handleFileUpload(file) {
      if (!file) return;
      clearOldState();
      document.getElementById('analyzing-spinner').style.display = 'inline-flex';

      const reader = new FileReader();
      reader.onload = async (e) => {
        const dataUrl = e.target.result;
        setPreview(dataUrl);

        try {
          const response = await fetch('/api/inspect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              image_base64: dataUrl.split(',')[1],
              filename: file.name
            })
          });
          const result = await response.json();
          currentInspection = result;
          renderResults(result);
        } catch (err) {
          alert('Upload failed: ' + err.message);
        } finally {
          document.getElementById('analyzing-spinner').style.display = 'none';
        }
      };
      reader.readAsDataURL(file);
    }

    function renderResults(res) {
      const banner = document.getElementById('result-banner');
      const badge = document.getElementById('status-badge');
      const title = document.getElementById('result-title');
      const desc = document.getElementById('result-desc');
      const actionBox = document.getElementById('next-action-box');
      const actionText = document.getElementById('next-action-text');

      const details = document.getElementById('inspection-details');
      const detGateStatus = document.getElementById('det-gate-status');
      const detSuppProb = document.getElementById('det-supp-prob');
      const detDefectRow = document.getElementById('det-defect-row');
      const detDefProb = document.getElementById('det-def-prob');
      const detPassRow = document.getElementById('det-pass-row');
      const detHoldRow = document.getElementById('det-hold-row');

      const geminiCard = document.getElementById('gemini-card');

      details.style.display = 'block';
      details.open = false; // Always keep collapsed by default

      const gate = res.gatekeeper;
      const defect = res.defect_detector;
      const pSupp = gate.supported_probability;

      detGateStatus.innerText = gate.accepted ? 'Accepted (Supported Impeller)' : 'Blocked (Unsupported Part)';
      detSuppProb.innerText = pSupp.toFixed(4);

      if (!gate.accepted) {
        // Outcome: Unsupported Image
        badge.className = 'status-badge badge-blocked';
        badge.innerText = 'Unsupported Image';
        title.innerText = 'Unsupported Image';
        desc.innerText = gate.message;

        actionBox.style.display = 'block';
        actionBox.className = 'next-action-box action-blocked';
        actionText.innerText = 'Upload a top-view photo of a pump impeller';

        detDefectRow.style.display = 'none';
        detPassRow.style.display = 'none';
        detHoldRow.style.display = 'none';

        // Do not show Gemini QA button for blocked parts
        geminiCard.style.display = 'none';
      } else {
        // Gatekeeper Accepted -> Show defect results
        detDefectRow.style.display = 'flex';
        detPassRow.style.display = 'flex';
        detHoldRow.style.display = 'flex';
        detDefProb.innerText = defect.defective_probability.toFixed(4);

        actionBox.style.display = 'block';
        actionText.innerText = res.next_action;

        if (res.final_action === 'PASS') {
          badge.className = 'status-badge badge-pass';
          badge.innerText = 'Pass Candidate';
          title.innerText = 'Pass Candidate';
          actionBox.className = 'next-action-box action-pass';
        } else if (res.final_action === 'MANUAL_REVIEW') {
          badge.className = 'status-badge badge-review';
          badge.innerText = 'Manual Review';
          title.innerText = 'Manual Review';
          actionBox.className = 'next-action-box action-review';
        } else {
          badge.className = 'status-badge badge-hold';
          badge.innerText = 'Hold';
          title.innerText = 'Hold';
          actionBox.className = 'next-action-box action-hold';
        }

        desc.innerText = res.message;

        // Show Gemini QA Summary button after accepted inspection
        geminiCard.style.display = 'block';
      }
    }

    async function generateQASummary() {
      if (!currentInspection) return;

      const btn = document.getElementById('qa-btn');
      const box = document.getElementById('qa-result-box');
      const badge = document.getElementById('qa-badge');
      const summaryText = document.getElementById('qa-summary-text');

      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Generating QA Summary...';

      try {
        const response = await fetch('/api/qa-summary', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action_label: currentInspection.action_label,
            final_action: currentInspection.final_action,
            next_action: currentInspection.next_action,
            supported_probability: currentInspection.gatekeeper.supported_probability,
            defective_probability: currentInspection.defect_detector ? currentInspection.defect_detector.defective_probability : null,
            gate_threshold: currentInspection.gatekeeper.threshold,
            pass_threshold: currentInspection.defect_detector ? currentInspection.defect_detector.pass_threshold : 0.50,
            hold_threshold: currentInspection.defect_detector ? currentInspection.defect_detector.hold_threshold : 0.76
          })
        });

        const data = await response.json();
        box.style.display = 'block';

        if (data.source === 'local' || data.is_local) {
          box.className = 'qa-box local';
          badge.className = 'qa-badge qa-badge-local';
          badge.innerText = 'Local summary';
          summaryText.innerText = data.summary;
        } else if (data.success && data.source === 'gemini') {
          box.className = 'qa-box gemini';
          badge.className = 'qa-badge qa-badge-gemini';
          const modelBadge = data.fallback_used
            ? `Gemini (${data.model_used} • Fallback)`
            : `Gemini (${data.model_used})`;
          badge.innerText = modelBadge;
          summaryText.innerText = data.summary;
        } else {
          box.className = 'qa-box error';
          badge.className = 'qa-badge qa-badge-error';
          badge.innerText = 'Error';
          summaryText.innerText = data.error || 'QA summary could not be generated.';
        }
      } catch (err) {
        box.style.display = 'block';
        box.className = 'qa-box error';
        badge.className = 'qa-badge qa-badge-error';
        badge.innerText = 'Request Failed';
        summaryText.innerText = 'QA summary request failed: ' + err.message;
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>Generate QA Summary</span>';
      }
    }
  </script>
</body>
</html>
"""

class GeminiCallError(Exception):
    pass

class ModelUnavailableError(GeminiCallError):
    pass

class QuotaLimitError(GeminiCallError):
    pass

class TemporaryServiceError(GeminiCallError):
    pass

def get_gemini_config():
    """Load Gemini configuration with environment variables taking precedence over .streamlit/secrets.toml.
    
    Order of precedence:
    1. Environment variables:
       - GEMINI_API_KEY
       - GEMINI_MODEL (default: gemini-3.5-flash-lite)
       - GEMINI_FALLBACK_MODEL (default: gemini-3.1-flash-lite)
    2. .streamlit/secrets.toml:
       - GEMINI_API_KEY / gemini_api_key / gemini.api_key
       - GEMINI_MODEL / model / model_id / primary_model / gemini.model
       - GEMINI_FALLBACK_MODEL / fallback_model / fallback_model_id / gemini.fallback_model
    3. Built-in defaults:
       - Primary: gemini-3.5-flash-lite
       - Fallback: gemini-3.1-flash-lite
    """
    secrets = {}
    secrets_path = Path(".streamlit") / "secrets.toml"
    if secrets_path.exists():
        try:
            with open(secrets_path, "rb") as f:
                secrets = tomllib.load(f)
        except Exception as e:
            sys.stderr.write(f"Warning: Failed to parse {secrets_path}: {e}\n")

    def lookup_secret(keys):
        for k in keys:
            if k in secrets and isinstance(secrets[k], str) and secrets[k].strip():
                return secrets[k].strip()
            gemini_sec = secrets.get("gemini", {})
            if isinstance(gemini_sec, dict) and k in gemini_sec and isinstance(gemini_sec[k], str) and gemini_sec[k].strip():
                return gemini_sec[k].strip()
        return None

    # 1. API Key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = lookup_secret(["GEMINI_API_KEY", "gemini_api_key", "api_key", "key"])

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
        "secrets_file_exists": secrets_path.exists(),
        "key_configured": bool(api_key),
        "key_source": "environment" if os.environ.get("GEMINI_API_KEY") else ("secrets.toml" if api_key else "none")
    }

def generate_local_summary(data):
    """Generate a deterministic, rule-governed Local QA summary based strictly on the existing QC decision."""
    action_label = data.get("action_label", "Pass Candidate")
    p_def = data.get("defective_probability")
    next_action = data.get("next_action", "")
    tau_pass = data.get("pass_threshold", 0.50)
    tau_hold = data.get("hold_threshold", 0.76)

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
    p_supp = data.get("supported_probability", 0.0)
    p_def = data.get("defective_probability", 0.0)
    tau_pass = data.get("pass_threshold", 0.50)
    tau_hold = data.get("hold_threshold", 0.76)

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
            return f"QA summary verified ({model_name}): Part disposition complies with QC threshold policy."

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
    """
    Generate an operator QA summary using Gemini with model fallback,
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
            "reason": "GEMINI_API_KEY is not configured in environment or .streamlit/secrets.toml.",
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

    # Step 2: Try fallback model ONCE for model unavailability, quota limits, or temporary service error
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

    # Step 3: If neither model can answer -> show a clearly labeled Local summary based on the existing QC decision!
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

class QCDemoHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {args[0]} {args[1]}\n")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        elif path.startswith("/samples/"):
            sample_id = path.replace("/samples/", "").strip()
            if sample_id in samples:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(samples[sample_id]["bytes"])
                return
            else:
                self.send_error(404, f"Sample '{sample_id}' not found")
                return

        elif path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            cfg = get_gemini_config()
            health_info = {
                "status": "healthy",
                "gatekeeper_threshold": pipeline.gate_threshold,
                "pass_threshold": pipeline.pass_threshold,
                "hold_threshold": pipeline.hold_threshold,
                "gemini_configured": cfg["key_configured"],
                "gemini_key_source": cfg["key_source"],
                "gemini_primary_model": cfg["primary_model"],
                "gemini_fallback_model": cfg["fallback_model"],
                "samples_loaded": list(samples.keys())
            }
            self.wfile.write(json.dumps(health_info).encode("utf-8"))
            return

        self.send_error(404, "Page Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))

        if parsed.path == "/api/inspect":
            body = self.rfile.read(content_length)
            content_type = self.headers.get("Content-Type", "")

            if "application/json" in content_type:
                data = json.loads(body.decode("utf-8"))
                if "sample_id" in data:
                    sample_id = data["sample_id"]
                    if sample_id in samples:
                        item = samples[sample_id]
                        res = pipeline.inspect(item["bytes"], filename=item["filename"])
                        self._send_json(res)
                        return
                    else:
                        self.send_error(400, f"Invalid sample_id: {sample_id}")
                        return
                elif "image_base64" in data:
                    raw_bytes = base64.b64decode(data["image_base64"])
                    filename = data.get("filename", "uploaded_image.jpg")
                    res = pipeline.inspect(raw_bytes, filename=filename)
                    self._send_json(res)
                    return
                else:
                    self.send_error(400, "Missing sample_id or image_base64")
                    return
            else:
                # Raw binary upload
                res = pipeline.inspect(body, filename="uploaded_image.jpg")
                self._send_json(res)
                return

        elif parsed.path == "/api/qa-summary":
            # Gemini QA summary request (sends ONLY inspection result and model scores)
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))
            
            # Allow mock_key or simulation for testing
            api_key = data.get("mock_api_key") or data.get("api_key")
            simulation = data.get("simulation") or data.get("simulate_error")

            qa_res = generate_gemini_qa_summary(data, api_key=api_key, simulation=simulation)
            self._send_json(qa_res)
            return

        self.send_error(404, "Endpoint not found")

    def _send_json(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

def preload_samples():
    """Extract sample images for 1-click testing from bundled sample_assets/ (or fallback to archives)."""
    global samples
    print("Pre-loading sample images for demo...")
    sample_dir = Path("sample_assets")

    # 1. Unsupported sample
    unsupp_bundled = sample_dir / "sample_unsupported.jpg"
    unsupp_raw = Path("unsupported_data/similar_parts/pump_housings/pump_housing_005_housing_for_modern_pump_at_westonzo.jpg")

    if unsupp_bundled.is_file():
        samples["sample_unsupported"] = {
            "bytes": unsupp_bundled.read_bytes(),
            "filename": unsupp_bundled.name,
            "label": "Pump Housing (Unsupported)"
        }
    elif unsupp_raw.is_file():
        samples["sample_unsupported"] = {
            "bytes": unsupp_raw.read_bytes(),
            "filename": unsupp_raw.name,
            "label": "Pump Housing (Unsupported)"
        }

    # 2. Supported samples
    sample_targets = {
        "sample_pass": {
            "bundled": sample_dir / "sample_pass.jpeg",
            "zip_path": "casting_data/test/ok_front/cast_ok_0_1093.jpeg",
            "label": "Pass Candidate"
        },
        "sample_review": {
            "bundled": sample_dir / "sample_review.jpeg",
            "zip_path": "casting_data/train/def_front/cast_def_0_2518.jpeg",
            "label": "Manual Review"
        },
        "sample_hold": {
            "bundled": sample_dir / "sample_hold.jpeg",
            "zip_path": "casting_data/test/def_front/cast_def_0_1292.jpeg",
            "label": "Hold"
        }
    }

    casting_zip = Path("casting_data.zip")
    zip_obj = ZipFile(casting_zip, "r") if casting_zip.is_file() else None

    for sid, info in sample_targets.items():
        if info["bundled"].is_file():
            samples[sid] = {
                "bytes": info["bundled"].read_bytes(),
                "filename": info["bundled"].name,
                "label": info["label"]
            }
        elif zip_obj is not None:
            try:
                samples[sid] = {
                    "bytes": zip_obj.read(info["zip_path"]),
                    "filename": Path(info["zip_path"]).name,
                    "label": info["label"]
                }
            except KeyError:
                pass

    if zip_obj:
        zip_obj.close()

    print(f"Loaded {len(samples)} sample images into memory for instant testing.")

def main():
    global pipeline
    parser = argparse.ArgumentParser(description="Pump Impeller Quality Inspector Demo")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)), help="Port to listen on (default: 8080 or $PORT)")
    parser.add_argument("--host", type=str, default=os.environ.get("HOST", "0.0.0.0"), help="Host interface (default: 0.0.0.0)")
    args = parser.parse_args()

    print("=" * 75)
    print("PUMP IMPELLER QUALITY INSPECTOR: DEMO APPLICATION")
    print("=" * 75)

    pipeline = ImpellerQCPipeline()
    preload_samples()

    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, QCDemoHandler)
    url = f"http://{args.host}:{args.port}"

    cfg = get_gemini_config()
    print("-" * 75)
    print(f"Web Application is LIVE at: {url}")
    print(f"  - Bound Host & Port:    {args.host}:{args.port}")
    if args.host == "0.0.0.0":
        print(f"  - Local Browser URL:    http://127.0.0.1:{args.port} or http://localhost:{args.port}")
    print(f"  - Gatekeeper Cutoff:    tau = {pipeline.gate_threshold:.4f}")
    print(f"  - Defect QC Cutoffs:    Pass < {pipeline.pass_threshold:.2f}, Hold >= {pipeline.hold_threshold:.2f}")
    print(f"  - Gemini Configuration: {cfg['primary_model']} (Fallback: {cfg['fallback_model']})")
    print(f"  - Gemini API Key:       {'Configured (' + cfg['key_source'] + ')' if cfg['key_configured'] else 'Not configured (Local summary active)'}")
    print(f"  - Press Ctrl+C in this terminal to stop the server.")
    print("=" * 75)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down QC Inspector server.")
        httpd.server_close()

if __name__ == "__main__":
    main()
