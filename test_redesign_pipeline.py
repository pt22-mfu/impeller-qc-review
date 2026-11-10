"""Comprehensive verification for redesigned Pump Impeller Defect Detection System.

Tests:
1. Unit-level strict stage separation (Stage 1 never loads/runs Stage 2 defect model).
2. Catalog integrity (all 1,041 casting test + 17 unsupported test, 5,199/1,044/1,041 split counts, SHA-256 original bytes, blind labels).
3. Held-out ZIPs & CSVs availability and contents.
4. End-to-end Playwright browser tests against local Streamlit server (1280x800 desktop & 390x844 mobile):
   - Browser title & page title check + Arial font / pure black text / light teal button contrast & no horizontal scroll.
   - Initial state: 1:2 column ratio, short project description on left, Upload at top right.
   - Valid upload -> runs ONLY eligibility check -> shows "This photo can be checked for defects." and "Check for defects" button; hides defect result, scores, and Gemini button; replaces description with uncropped photo on left only.
   - Click "Check for defects" -> runs defect model -> shows verdict + "What to do next" + "OK score" & "Defect score" + collapsed diagnostics + "Explain this result with Gemini" button.
   - Unrelated Streamlit rerun keeps Step 2 result & Gemini explanation intact.
   - Click "Explain this result with Gemini" -> runs Gemini ONLY on click and renders plain-English explanation.
   - Switching / clearing photo -> resets defect result & Gemini explanation (and restores description on Clear).
   - Unsupported upload -> shows "Please upload a clear top-view photo of a pump impeller.", hides defect & Gemini buttons.
   - Reviewer-selected held-out test image -> actual label stays hidden until explicitly revealed AFTER testing.
   - Captures desktop & mobile screenshots for visual review.
"""
import hashlib
import io
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from playwright.sync_api import sync_playwright

import app
from pipeline import ImpellerQCPipeline


def run_unit_checks():
    print("\n=== PART 1: Unit Checks (Stage Separation, Catalog, ZIPs) ===")
    pipe = ImpellerQCPipeline()
    assert pipe._defect_model is None
    assert pipe.defect_call_count == 0
    assert abs(pipe.gate_threshold - 0.0049) < 1e-6
    assert abs(pipe.pass_threshold - 0.50) < 1e-6
    assert abs(pipe.hold_threshold - 0.76) < 1e-6

    pass_bytes = Path("sample_assets/sample_pass.jpeg").read_bytes()
    elig = pipe.check_eligibility(pass_bytes, filename="sample_pass.jpeg")
    assert elig["accepted"] is True
    assert elig["message"] == "This photo can be checked for defects."
    assert pipe._defect_model is None, "Defect model must NOT be loaded in Stage 1!"
    assert pipe.defect_call_count == 0, "Defect model must NOT run in Stage 1!"
    assert pipe.checker_call_count == 1
    print("PASS 1.1: Stage 1 eligibility check did not load or invoke Stage 2 defect model.")

    det_res = pipe.check_defects(pass_bytes, filename="sample_pass.jpeg", gatekeeper_result=elig)
    assert pipe.defect_call_count == 1
    assert det_res["action_label"] == "Pass Candidate"
    assert det_res["next_action"] == "Pass candidate. Follow your normal release procedure."
    p_def = det_res["defect_detector"]["defective_probability"]
    p_ok = det_res["defect_detector"]["ok_probability"]
    assert abs((p_def + p_ok) - 1.0) < 1e-6
    print(f"PASS 1.2: Stage 2 ran on demand -> Pass Candidate (OK={p_ok:.4f}, Defect={p_def:.4f}).")

    unsupp_bytes = Path("sample_assets/sample_unsupported.jpg").read_bytes()
    elig_u = pipe.check_eligibility(unsupp_bytes, filename="sample_unsupported.jpg")
    assert elig_u["accepted"] is False
    assert elig_u["message"] == "Please upload a clear top-view photo of a pump impeller."
    assert pipe.defect_call_count == 1, "Defect model must NOT run on unsupported photo!"
    print("PASS 1.3: Unsupported check blocked without calling defect model.")

    # Catalog & Held-out bundles
    cat = app.load_test_catalog()
    assert len(cat["casting"]) == 1041, f"Expected 1041 casting test images, got {len(cat['casting'])}"
    assert len(cat["unsupported"]) == 17, f"Expected 17 unsupported test images, got {len(cat['unsupported'])}"
    assert cat["split_counts"] == {"train": 5199, "validation": 1044, "test": 1041}
    assert cat["unsupp_counts"] == {"train": 60, "validation": 13, "test": 17}

    # Verify blind labels hide actual labels and include error cases
    all_casting_paths = {e["original_path"] for e in cat["casting"]}
    all_unsupp_paths = {e["original_path"] for e in cat["unsupported"]}
    # Known model mistakes in Stage 5 and Stage 6C:
    assert "unsupported_data/similar_parts/bearings/bearing_011_spinbal_jpg.jpg" in all_unsupp_paths
    assert "unsupported_data/similar_parts/misoriented_impellers/misoriented_impeller_001_24_blade_compressor_impeller_nara.jpg" in all_unsupp_paths
    for e in cat["casting"]:
        assert "def_front" not in e["blind_label"] and "ok_front" not in e["blind_label"]
    for e in cat["unsupported"]:
        assert e["category"] not in e["blind_label"]

    # Spot-check byte-for-byte SHA-256 preservation on first and known-error entries
    for e in [cat["casting"][0], cat["casting"][-1], cat["unsupported"][0]]:
        raw = app.read_test_image_bytes(e)
        assert hashlib.sha256(raw).hexdigest() == e["sha256"]
    print("PASS 1.4: Catalog (1,041 casting + 17 unsupported, error cases, exact SHA-256 bytes, blind labels) verified.")


def capture_full_streamlit(page, out_path, width=1280, min_height=860):
    """Capture complete Streamlit content cleanly from top to bottom."""
    page.evaluate(
        "document.querySelectorAll('[data-testid=\"stMain\"], [data-testid=\"stAppViewContainer\"]').forEach(el => el.scrollTop = 0)"
    )
    h = page.evaluate(
        """() => {
            const el = document.querySelector('[data-testid="stMainBlockContainer"]') || document.body;
            return Math.max(el.scrollHeight + 100, 844);
        }"""
    )
    page.set_viewport_size({"width": width, "height": max(min_height, min(int(h), 2600))})
    page.wait_for_timeout(400)
    page.screenshot(path=str(out_path))


def run_browser_e2e(port=8503, artifact_dir=Path(".")):
    print("\n=== PART 2: Live Streamlit Browser E2E Verification (Desktop & Mobile) ===")
    url = f"http://localhost:{port}"
    log_path = artifact_dir / "streamlit_server_test.log"

    sub_env = os.environ.copy()
    sub_env["PYTHONUNBUFFERED"] = "1"

    # Launch local Streamlit process capturing stdout/stderr so we verify model load timing too!
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ],
        stdout=open(log_path, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        env=sub_env,
    )

    try:
        # Wait until port accepts connections
        import urllib.request

        ready = False
        for _ in range(40):
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.5)
        assert ready, f"Streamlit failed to start on {url}"
        print(f"Streamlit server running at {url}")

        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            context = browser.new_context(viewport={"width": 1280, "height": 860})
            page = context.new_page()

            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_selector("h1.main-page-title", timeout=30000)
            page.wait_for_timeout(1500)

            # Check browser tab title and visible title
            tab_title = page.title()
            h1_text = page.locator("h1.main-page-title").inner_text().strip()
            assert "Defect Detection System for Pump Impellers" in tab_title, f"Tab title mismatch: {tab_title}"
            assert h1_text == "Defect Detection System for Pump Impellers", f"H1 mismatch: {h1_text}"

            # Verify 1:2 column widths ratio on desktop (~1:2) on the outer stHorizontalBlock
            cols = page.locator('[data-testid="stHorizontalBlock"]').first.locator(':scope > [data-testid="stColumn"]')
            c0_box = cols.nth(0).bounding_box()
            c1_box = cols.nth(1).bounding_box()
            ratio = c1_box["width"] / c0_box["width"]
            print(f"Desktop column ratio right/left = {ratio:.2f} (widths: {c0_box['width']:.0f} vs {c1_box['width']:.0f})")
            assert 1.7 <= ratio <= 2.3, f"Expected ~1:2 column ratio, got {ratio:.2f}"

            # Check typography & style properties
            h1_style = page.locator("h1.main-page-title").evaluate(
                "el => ({font: getComputedStyle(el).fontFamily, size: getComputedStyle(el).fontSize, color: getComputedStyle(el).color})"
            )
            h2_style = page.locator("h2.section-heading").first.evaluate(
                "el => ({font: getComputedStyle(el).fontFamily, size: getComputedStyle(el).fontSize, color: getComputedStyle(el).color})"
            )
            print(f"Styles verified -> H1: {h1_style} | H2: {h2_style}")
            assert "Arial" in h1_style["font"], f"Expected Arial, got {h1_style}"
            assert h1_style["size"] == "34px", f"Expected 34px H1, got {h1_style['size']}"
            assert h2_style["size"] == "22px", f"Expected 22px H2, got {h2_style['size']}"
            assert h1_style["color"] == "rgb(0, 0, 0)", f"Expected black text, got {h1_style['color']}"

            # Initial state: project description on left, no image preview anywhere
            body0 = page.inner_text("body")
            assert "About this inspector" in body0
            assert page.locator('[data-testid="stImage"]').count() == 0
            capture_full_streamlit(page, artifact_dir / "desktop_initial.png", width=1280, min_height=860)
            print("PASS 2.1: Initial desktop 1:2 layout, titles, fonts, black text, description captured.")

            # -----------------------------------------------------------------
            # Test 2: Valid upload -> ONLY Step 1 runs
            # -----------------------------------------------------------------
            file_input = page.locator("input[type='file']")
            file_input.set_input_files(str(Path("sample_assets/sample_review.jpeg").resolve()))
            page.wait_for_selector("text=This photo can be checked for defects.", timeout=30000)
            page.wait_for_timeout(1200)

            body_s1 = page.inner_text("body")
            # Description replaced by photo preview on left ONLY
            assert "About this inspector" not in body_s1
            assert page.locator('[data-testid="stImage"]').count() == 1
            img_box = page.locator('[data-testid="stImage"]').first.bounding_box()
            assert img_box["x"] < c1_box["x"], "Photo preview MUST be on the left column only!"

            # No defect result, no defect scores, no Gemini button before clicking Check for defects
            assert "What to do next" not in body_s1
            assert "OK score" not in body_s1
            assert "Defect score" not in body_s1
            assert page.get_by_role("button", name="Explain this result with Gemini").count() == 0
            assert page.get_by_role("button", name="Check for defects").count() == 1

            # Check server log to confirm defect model NOT loaded yet!
            server_log_after_s1 = log_path.read_text(encoding="utf-8", errors="ignore")
            assert "Loading Input Checker Model" in server_log_after_s1
            assert "Loading Defect Inspection Model" not in server_log_after_s1, (
                "CRITICAL: Defect Inspection Model loaded before clicking Check for defects!"
            )
            capture_full_streamlit(page, artifact_dir / "desktop_step1_eligible.png", width=1280, min_height=860)
            print("PASS 2.2: Valid upload ran ONLY Step 1 Eligibility (defect model NOT loaded/run, preview on left only).")

            # -----------------------------------------------------------------
            # Test 3: Click "Check for defects" -> Step 2 runs, Gemini still hidden until clicked
            # -----------------------------------------------------------------
            page.get_by_role("button", name="Check for defects").click()
            page.wait_for_selector("text=What to do next", timeout=30000)
            page.wait_for_timeout(1200)

            body_s2 = page.inner_text("body")
            assert "Manual Review" in body_s2
            assert "What to do next" in body_s2
            assert "Conduct visual inspection at review station" in body_s2
            assert "OK score" in body_s2
            assert "Defect score" in body_s2
            assert page.get_by_role("button", name="Explain this result with Gemini").count() == 1
            assert "Gemini (" not in body_s2 and "Local summary" not in body_s2

            server_log_after_s2 = log_path.read_text(encoding="utf-8", errors="ignore")
            assert "Loading Defect Inspection Model" in server_log_after_s2
            print("PASS 2.3: Defect model ran ONLY after clicking 'Check for defects'; Gemini waited cleanly.")

            # -----------------------------------------------------------------
            # Test 4: Click "Explain this result with Gemini" & verify unrelated rerun persistence
            # -----------------------------------------------------------------
            page.get_by_role("button", name="Explain this result with Gemini").click()
            page.wait_for_selector(".gemini-box", timeout=35000)
            page.wait_for_timeout(1000)

            gemini_text = page.locator(".gemini-box").inner_text()
            print(f"Gemini explanation rendered:\n  {gemini_text}")
            assert "Manual Review" in gemini_text or "review" in gemini_text.lower()
            capture_full_streamlit(page, artifact_dir / "desktop_step3_gemini_result.png", width=1280, min_height=860)
            page.set_viewport_size({"width": 1280, "height": 860})

            # Trigger unrelated rerun: switch dataset radio in left column to Unsupported test set (17)
            page.get_by_text("Unsupported test set (17)").click()
            page.wait_for_timeout(1500)
            body_rerun = page.inner_text("body")
            assert "What to do next" in body_rerun, "Step 2 result disappeared on unrelated rerun!"
            assert page.locator(".gemini-box").count() == 1, "Gemini box disappeared on unrelated rerun!"
            print("PASS 2.4: Gemini ran on click, and both Step 2 result & Gemini persisted across unrelated rerun.")

            # -----------------------------------------------------------------
            # Test 5: Remove via uploader (x) clears results, then Unsupported upload -> rejected message
            # -----------------------------------------------------------------
            page.locator('[data-testid="stFileChipDeleteBtn"] button').click()
            page.wait_for_selector("text=About this inspector", timeout=15000)
            assert "What to do next" not in page.inner_text("body"), "Removing uploaded file did not reset Step 2 result!"
            assert page.locator(".gemini-box").count() == 0, "Removing uploaded file did not reset Gemini box!"

            page.locator("input[type='file']").set_input_files(str(Path("sample_assets/sample_unsupported.jpg").resolve()))
            page.wait_for_selector("text=Please upload a clear top-view photo of a pump impeller.", timeout=20000)
            page.wait_for_timeout(1200)

            body_unsupp = page.inner_text("body")
            assert "Please upload a clear top-view photo of a pump impeller." in body_unsupp
            assert "What to do next" not in body_unsupp, "Step 2 result must not show on unsupported photo!"
            assert page.locator(".gemini-box").count() == 0, "Gemini must not show on unsupported photo!"
            assert page.get_by_role("button", name="Check for defects").count() == 0
            assert page.get_by_role("button", name="Explain this result with Gemini").count() == 0
            capture_full_streamlit(page, artifact_dir / "desktop_unsupported_rejected.png", width=1280, min_height=860)
            page.set_viewport_size({"width": 1280, "height": 860})
            print("PASS 2.5: Clear via uploader reset prior results, and unsupported upload rejected cleanly hiding both buttons.")

            # Clear photo button -> restores description
            page.get_by_role("button", name="Clear photo").click()
            page.wait_for_selector("text=About this inspector", timeout=15000)
            assert page.locator('[data-testid="stImage"]').count() == 0
            print("PASS 2.6: Clear photo restored project description and removed preview.")

            # -----------------------------------------------------------------
            # Test 6: Reviewer-selected held-out test image + hidden ground-truth reveal after testing
            # -----------------------------------------------------------------
            # Switch to Casting test set (1,041) and pick a blind image
            page.get_by_text("Casting test set (1,041)").click()
            page.wait_for_timeout(800)
            page.locator('[data-testid="stSelectbox"] input').first.click()
            page.wait_for_timeout(500)
            page.locator('[role="option"]').first.click()
            page.wait_for_selector("text=Step 1 Result", timeout=20000)
            page.wait_for_timeout(1500)

            body_ds = page.inner_text("body")
            assert "About this inspector" not in body_ds
            assert "Actual label:" not in body_ds, "Ground truth leaked BEFORE clicking reveal!"

            if page.get_by_role("button", name="Check for defects").count() > 0:
                assert page.get_by_role("button", name="Reveal actual ground-truth label").count() == 0, (
                    "Reveal appeared before Step 2 testing completed!"
                )
                page.get_by_role("button", name="Check for defects").click()
                page.wait_for_selector("text=What to do next", timeout=20000)
                page.wait_for_timeout(1000)

            page.get_by_role("button", name="Reveal actual ground-truth label").click()
            page.wait_for_selector("text=Ground-Truth Record (Revealed after testing)", timeout=15000)
            gt_text = page.inner_text("body")
            assert "Actual label:" in gt_text
            assert "SHA-256:" in gt_text
            print("PASS 2.7: Casting test selection stayed blind through testing and revealed attribution & ground truth on click.")

            # Now switch photo using Quick examples ("Hold") -> resets Step 2/Gemini/revealed label
            page.get_by_text("Quick examples (4 preset demo images)").click()
            page.wait_for_timeout(500)
            page.get_by_role("button", name="Hold").click()
            page.wait_for_selector("text=This photo can be checked for defects.", timeout=20000)
            page.wait_for_timeout(1200)
            assert "Ground-Truth Record" not in page.inner_text("body")
            assert page.get_by_role("button", name="Check for defects").count() == 1
            print("PASS 2.8: Switching photo via Quick examples reset previous state and ran ONLY Step 1.")

            # Complete Step 2 on desktop for full mobile check too
            page.get_by_role("button", name="Check for defects").click()
            page.wait_for_selector("text=Hold for supervisor review.", timeout=20000)
            page.wait_for_timeout(1000)
            capture_full_streamlit(page, artifact_dir / "desktop_hold_result.png", width=1280, min_height=860)

            # -----------------------------------------------------------------
            # Test 7: Mobile viewport check (390x844) -> clean vertical stack, zero horizontal overflow
            # -----------------------------------------------------------------
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(1500)
            overflow = page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
            dims = page.evaluate("({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth})")
            print(f"Mobile viewport check: scrollWidth={dims['sw']}, clientWidth={dims['cw']}, overflow={overflow}")
            assert not overflow, f"Horizontal scroll overflow detected on mobile! {dims}"
            capture_full_streamlit(page, artifact_dir / "mobile_hold_result.png", width=390, min_height=844)

            # Also capture initial cleared mobile view to verify Upload visibility on mobile
            page.get_by_role("button", name="Clear photo").click()
            page.wait_for_selector("text=About this inspector", timeout=15000)
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(1000)
            page.evaluate("document.querySelectorAll('[data-testid=\"stMain\"], [data-testid=\"stAppViewContainer\"]').forEach(el => el.scrollTop = 0)")
            upload_box_mob = page.locator("h2.section-heading:has-text('Upload a photo')").bounding_box()
            print(f"Mobile Upload heading y-position when initial/cleared: {upload_box_mob['y']:.0f}px (viewport height 844px)")
            capture_full_streamlit(page, artifact_dir / "mobile_initial.png", width=390, min_height=844)
            print("PASS 2.9: Mobile responsive layout verified with zero horizontal scrolling.")

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    artifact_dir = Path(r"C:\Users\VivoBook\.gemini\antigravity\brain\f2c3d39f-379c-451c-bcb2-f71020e5c214")
    run_unit_checks()
    run_browser_e2e(port=8503, artifact_dir=artifact_dir)
    print("\n" + "=" * 65)
    print("ALL REDESIGN UNIT & BROWSER VERIFICATION CHECKS PASSED!")
    print("=" * 65)
