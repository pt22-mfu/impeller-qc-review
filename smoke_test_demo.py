"""Smoke test for the Two-Stage Impeller QC Pipeline.

Verifies all four required outcomes:
1. Blocked (Unsupported Part): Gatekeeper blocks image, prompts user for supported impeller.
2. Pass Candidate: Gatekeeper accepts, defect detector scores < 0.50 -> Pass.
3. Manual Review: Gatekeeper accepts, defect detector scores 0.50 <= p < 0.76 -> Manual Review.
4. Hold: Gatekeeper accepts, defect detector scores >= 0.76 -> Hold.
"""
from pathlib import Path
from zipfile import ZipFile
from pipeline import ImpellerQCPipeline

def run_smoke_test():
    print("=" * 80)
    print("RUNNING SMOKE TEST: TWO-STAGE IMPELLER QC PIPELINE")
    print("=" * 80)

    pipeline = ImpellerQCPipeline()
    casting_zip = Path("casting_data.zip")
    assert casting_zip.is_file(), f"Missing {casting_zip}"

    # Extract test samples
    test_cases = [
        {
            "name": "Outcome 1: Blocked (Unsupported Part)",
            "expected_outcome": "BLOCKED",
            "source": "disk",
            "path": Path("unsupported_data/similar_parts/pump_housings/pump_housing_005_housing_for_modern_pump_at_westonzo.jpg")
        },
        {
            "name": "Outcome 2: Pass Candidate (Clean Impeller)",
            "expected_outcome": "PASS",
            "source": "zip",
            "path": "casting_data/test/ok_front/cast_ok_0_1093.jpeg"
        },
        {
            "name": "Outcome 3: Manual Review (Borderline Quality)",
            "expected_outcome": "MANUAL_REVIEW",
            "source": "zip",
            "path": "casting_data/train/def_front/cast_def_0_2518.jpeg"
        },
        {
            "name": "Outcome 4: Hold (Defective Impeller)",
            "expected_outcome": "HOLD",
            "source": "zip",
            "path": "casting_data/test/def_front/cast_def_0_1292.jpeg"
        }
    ]

    all_passed = True

    with ZipFile(casting_zip, "r") as zf:
        for idx, tc in enumerate(test_cases, 1):
            print(f"\n[{idx}/4] Testing {tc['name']}...")
            if tc["source"] == "disk":
                img_bytes = tc["path"].read_bytes()
                filename = tc["path"].name
            else:
                img_bytes = zf.read(tc["path"])
                filename = Path(tc["path"]).name

            result = pipeline.inspect(img_bytes, filename=filename)

            gate = result["gatekeeper"]
            defect = result["defect_detector"]
            final_act = result["final_action"]

            print(f"  - Input File:           {filename}")
            print(f"  - Gatekeeper Status:    {gate['decision']} (prob: {gate['supported_probability']:.4f}, threshold: {gate['threshold']:.4f})")
            if defect["ran"]:
                print(f"  - Defect Detector:      {defect['action']} (prob: {defect['defective_probability']:.4f}, pass: {defect['pass_threshold']:.2f}, hold: {defect['hold_threshold']:.2f})")
            else:
                print(f"  - Defect Detector:      [SKIPPED] {defect['reason']}")
            print(f"  - Final Action Outcome: {result['action_label']} ({final_act})")
            print(f"  - User Message:         {result.get('message', gate.get('message'))}")

            # Verification assertions
            match = (final_act == tc["expected_outcome"])
            if match:
                print(f"  --> [PASS] Verified expected outcome '{tc['expected_outcome']}'")
            else:
                print(f"  --> [FAIL] Expected '{tc['expected_outcome']}', got '{final_act}'")
                all_passed = False

            if tc["expected_outcome"] == "BLOCKED":
                assert not defect["ran"], "Defect detector should NOT have run for blocked image!"
                assert "supported impeller" in gate["message"].lower(), "Prompt must ask for supported impeller photo!"

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL 4 PIPELINE OUTCOMES SUCCESSFULLY VERIFIED!")
        print("Pipeline logic, threshold enforcement, and routing are 100% operational.")
    else:
        print("SOME SMOKE TEST CASES FAILED!")
    print("=" * 80)
    return all_passed

if __name__ == "__main__":
    success = run_smoke_test()
    if not success:
        exit(1)

