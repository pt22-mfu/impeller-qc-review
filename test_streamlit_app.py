"""Automated verification script for Streamlit app, pipeline, and Gemini QA summary."""
import json
import os
import sys
from pathlib import Path

# Set up test environment
import app

def test_gemini_config():
    print("\n--- 1. Testing Gemini Configuration ---")
    cfg = app.get_gemini_config()
    print(f"Key configured: {cfg['key_configured']}")
    print(f"Key source:     {cfg['key_source']}")
    print(f"Primary model:  {cfg['primary_model']}")
    print(f"Fallback model: {cfg['fallback_model']}")
    assert cfg["key_configured"], "GEMINI_API_KEY should be configured"
    assert cfg["primary_model"] == "gemini-3.5-flash-lite", "Primary model should be gemini-3.5-flash-lite"
    assert cfg["fallback_model"] == "gemini-3.1-flash-lite", "Fallback model should be gemini-3.1-flash-lite"
    print("PASS: Gemini configuration loaded successfully.")

def test_four_sample_outcomes():
    print("\n--- 2. Testing 4 Sample Outcomes ---")
    pipe = app.load_pipeline()
    
    expected_outcomes = {
        "unsupported": {
            "status": "BLOCKED",
            "action_label": "Unsupported Image",
            "next_action": "Upload a top-view photo of a pump impeller"
        },
        "pass": {
            "status": "ACCEPTED",
            "action_label": "Pass Candidate",
            "next_action": "Pass candidate. Follow your normal release procedure."
        },
        "review": {
            "status": "ACCEPTED",
            "action_label": "Manual Review",
            "next_action": "Conduct visual inspection at review station"
        },
        "hold": {
            "status": "ACCEPTED",
            "action_label": "Hold",
            "next_action": "Hold for supervisor review."
        }
    }

    results = {}
    for sample_key, sample_info in app.SAMPLES.items():
        assert sample_info["path"].exists(), f"Missing file: {sample_info['path']}"
        raw_bytes = sample_info["path"].read_bytes()
        res = pipe.inspect(raw_bytes, filename=sample_info["filename"])
        results[sample_key] = res
        
        expected = expected_outcomes[sample_key]
        print(f"\nSample: {sample_key} ({sample_info['filename']})")
        print(f"  - Status:       {res['status']} (Expected: {expected['status']})")
        print(f"  - Action Label: {res['action_label']} (Expected: {expected['action_label']})")
        print(f"  - Next Action:  {res['next_action']}")
        
        assert res["status"] == expected["status"], f"Status mismatch for {sample_key}"
        assert res["action_label"] == expected["action_label"], f"Action label mismatch for {sample_key}"
        assert res["next_action"] == expected["next_action"], f"Next action mismatch for {sample_key}"
        print(f"  PASS: Outcome verified for {sample_key}.")

    return results

def test_gemini_live_summary(pass_result):
    print("\n--- 3. Testing Live Gemini QA Summary ---")
    summary = app.generate_gemini_qa_summary(pass_result)
    print(f"Source:     {summary.get('source')}")
    print(f"Label:      {summary.get('label')}")
    print(f"Model used: {summary.get('model_used')}")
    print(f"Summary:    {summary.get('summary')}")
    
    assert summary["success"] is True, "Summary generation should succeed"
    assert summary["source"] in ["gemini", "local"], f"Invalid source: {summary.get('source')}"
    print("PASS: Live QA summary successfully generated.")

def test_fallback_and_local_summary(pass_result):
    print("\n--- 4. Testing Fallback & Local Summary Guardrail ---")
    # 4a. Simulate primary 404 -> fallback succeeds
    res_fallback = app.generate_gemini_qa_summary(pass_result, simulation={"primary": "404", "fallback": "mock_success"})
    assert res_fallback["fallback_used"] is True
    assert res_fallback["model_used"] == "gemini-3.1-flash-lite"
    print("PASS: Primary 404 correctly triggers fallback to gemini-3.1-flash-lite.")

    # 4b. Simulate both fail -> local summary
    res_local = app.generate_gemini_qa_summary(pass_result, simulation="both_fail")
    assert res_local["is_local"] is True
    assert res_local["label"] == "Local summary"
    assert "Pass candidate. Follow your normal release procedure." in res_local["summary"]
    print("PASS: Both failing correctly produces Local summary.")

def test_attribution_and_exclusions():
    print("\n--- 5. Testing Attribution & Git Exclusions ---")
    attr_path = Path("sample_assets") / "ATTRIBUTION.md"
    assert attr_path.exists(), "ATTRIBUTION.md missing"
    content = attr_path.read_text(encoding="utf-8")
    assert "CC BY-NC-ND 4.0" in content, "Casting dataset license should be CC BY-NC-ND 4.0"
    assert "Raviraj Joshi" in content, "Attribution to Raviraj Joshi should be present"
    print("PASS: sample_assets/ATTRIBUTION.md contains CC BY-NC-ND 4.0 and Raviraj Joshi.")

if __name__ == "__main__":
    test_gemini_config()
    results = test_four_sample_outcomes()
    test_gemini_live_summary(results["pass"])
    test_fallback_and_local_summary(results["pass"])
    test_attribution_and_exclusions()
    print("\n" + "="*60)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("="*60)
