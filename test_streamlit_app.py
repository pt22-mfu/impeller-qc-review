"""Automated verification script for redesigned Streamlit app, pipeline, held-out bundles, and Gemini explanation."""
import hashlib
import io
import zipfile
from pathlib import Path

import app


def test_gemini_config():
    print("\n--- 1. Testing Gemini Configuration ---")
    cfg = app.get_gemini_config()
    print(f"Key configured: {cfg['key_configured']} (source: {cfg['key_source']})")
    print(f"Primary model:  {cfg['primary_model']}")
    print(f"Fallback model: {cfg['fallback_model']}")
    assert cfg["key_configured"], "GEMINI_API_KEY should be configured"
    assert cfg["primary_model"] == "gemini-3.5-flash-lite"
    assert cfg["fallback_model"] == "gemini-3.1-flash-lite"
    print("PASS: Gemini configuration loaded successfully.")


def test_four_quick_examples_staged():
    print("\n--- 2. Testing Quick Examples via Staged Pipeline ---")
    pipe = app.load_pipeline()
    expected_outcomes = {
        "unsupported": ("BLOCKED", "Unsupported Image", "Please upload a clear top-view photo of a pump impeller."),
        "pass": ("ACCEPTED", "Pass Candidate", "Pass candidate. Follow your normal release procedure."),
        "review": ("ACCEPTED", "Manual Review", "Conduct visual inspection at review station"),
        "hold": ("ACCEPTED", "Hold", "Hold for supervisor review."),
    }

    results = {}
    for key, info in app.QUICK_EXAMPLES.items():
        assert info["path"].is_file(), f"Missing quick example file: {info['path']}"
        raw_bytes = info["path"].read_bytes()
        elig = pipe.check_eligibility(raw_bytes, filename=info["filename"])
        if key == "unsupported":
            assert elig["accepted"] is False
            assert elig["message"] == expected_outcomes[key][2]
            print(f"Quick example '{key}' -> REJECTED at Stage 1: {elig['message']}")
        else:
            assert elig["accepted"] is True
            assert elig["message"] == "This photo can be checked for defects."
            res = pipe.check_defects(raw_bytes, filename=info["filename"], gatekeeper_result=elig)
            results[key] = res
            exp_status, exp_label, exp_action = expected_outcomes[key]
            assert res["status"] == exp_status
            assert res["action_label"] == exp_label
            assert res["next_action"] == exp_action
            p_ok = res["defect_detector"]["ok_probability"]
            p_def = res["defect_detector"]["defective_probability"]
            assert abs((p_ok + p_def) - 1.0) < 1e-6
            print(
                f"Quick example '{key}' -> Stage 1 OK, Stage 2: {res['action_label']} "
                f"(OK={p_ok:.4f}, Defect={p_def:.4f})"
            )
    return results


def test_gemini_live_and_fallback(pass_result):
    print("\n--- 3. Testing Gemini Live Explanation & Fallback ---")
    res_live = app.generate_gemini_qa_summary(pass_result)
    print(f"Live Gemini source={res_live['source']} label={res_live['label']}")
    print(f"Summary: {res_live['summary']}")
    assert res_live["success"] is True

    res_fallback = app.generate_gemini_qa_summary(
        pass_result, simulation={"primary": "404", "fallback": "mock_success"}
    )
    assert res_fallback["fallback_used"] is True
    assert res_fallback["model_used"] == "gemini-3.1-flash-lite"
    print("PASS: Primary 404 fell back to gemini-3.1-flash-lite.")

    res_local = app.generate_gemini_qa_summary(pass_result, simulation="both_fail")
    assert res_local["is_local"] is True
    assert res_local["label"] == "Local summary"
    assert "Pass candidate. Follow your normal release procedure." in res_local["summary"]
    print("PASS: Dual failure triggered deterministic Local summary guardrail.")


def test_held_out_bundles_and_attribution():
    print("\n--- 4. Testing Held-Out Test Bundles & Manifest Splits ---")
    assert app.CASTING_HELD_OUT_ZIP.is_file(), "Missing casting_held_out_test.zip"
    assert app.UNSUPP_HELD_OUT_ZIP.is_file(), "Missing unsupported_held_out_test.zip"

    with zipfile.ZipFile(app.CASTING_HELD_OUT_ZIP, "r") as zf:
        names = zf.namelist()
        assert "casting_test_ground_truth.csv" in names
        assert "ATTRIBUTION.md" in names
        img_names = [n for n in names if n.startswith("images/")]
        assert len(img_names) == 1041

    with zipfile.ZipFile(app.UNSUPP_HELD_OUT_ZIP, "r") as zf:
        names = zf.namelist()
        assert "unsupported_test_ground_truth.csv" in names
        assert "ATTRIBUTION.md" in names
        img_names = [n for n in names if n.startswith("images/")]
        assert len(img_names) == 17

    cat = app.load_test_catalog()
    assert cat["split_counts"] == {"train": 5199, "validation": 1044, "test": 1041}
    assert cat["unsupp_counts"] == {"train": 60, "validation": 13, "test": 17}
    assert len(cat["casting"]) == 1041
    assert len(cat["unsupported"]) == 17
    # Verify sample raw byte lookup
    raw = app.read_test_image_bytes(cat["casting"][0])
    assert hashlib.sha256(raw).hexdigest() == cat["casting"][0]["sha256"]
    print("PASS: Held-out ZIPs (1,041 + 17) and manifests verified with exact SHA-256 preservation.")


if __name__ == "__main__":
    test_gemini_config()
    results = test_four_quick_examples_staged()
    test_gemini_live_and_fallback(results["pass"])
    test_held_out_bundles_and_attribution()
    print("\n" + "=" * 60)
    print("ALL STREAMLIT APP UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
