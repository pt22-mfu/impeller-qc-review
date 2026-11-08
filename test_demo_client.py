"""HTTP Client Test for the Impeller QC Demo Server."""
import json
import urllib.request
from pathlib import Path

def test_demo_endpoints(port=8085):
    base_url = f"http://127.0.0.1:{port}"
    print(f"Connecting to QC Demo Server at {base_url}...")

    # 1. Test GET /
    req = urllib.request.Request(f"{base_url}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        html = resp.read().decode("utf-8")
        assert "Pump Impeller Quality Control Inspector" in html
    print("  [PASS] GET / -> HTTP 200 (HTML dashboard served)")

    # 2. Test GET /api/health
    req = urllib.request.Request(f"{base_url}/api/health")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        health = json.loads(resp.read().decode("utf-8"))
        assert health["status"] == "healthy"
        assert abs(health["gatekeeper_threshold"] - 0.0049) < 1e-4
        assert abs(health["pass_threshold"] - 0.50) < 1e-4
        assert abs(health["hold_threshold"] - 0.76) < 1e-4
    print("  [PASS] GET /api/health -> HTTP 200 (Policy thresholds verified: gate=0.0049, pass=0.50, hold=0.76)")

    # 3. Test Sample Inspections for all 4 outcomes
    cases = [
        {
            "id": "sample_unsupported",
            "name": "Outcome 1: Blocked (Unsupported Part)",
            "expected_outcome": "BLOCKED",
            "expect_gate_accepted": False,
            "expect_defect_ran": False
        },
        {
            "id": "sample_pass",
            "name": "Outcome 2: Pass Candidate (Clean Impeller)",
            "expected_outcome": "PASS",
            "expect_gate_accepted": True,
            "expect_defect_ran": True
        },
        {
            "id": "sample_review",
            "name": "Outcome 3: Manual Review (Borderline Quality)",
            "expected_outcome": "MANUAL_REVIEW",
            "expect_gate_accepted": True,
            "expect_defect_ran": True
        },
        {
            "id": "sample_hold",
            "name": "Outcome 4: Hold (Defective Impeller)",
            "expected_outcome": "HOLD",
            "expect_gate_accepted": True,
            "expect_defect_ran": True
        }
    ]

    for c in cases:
        payload = json.dumps({"sample_id": c["id"]}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/inspect",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode("utf-8"))
            
            gate = res["gatekeeper"]
            defect = res["defect_detector"]
            act = res["final_action"]

            assert act == c["expected_outcome"], f"Expected {c['expected_outcome']}, got {act}"
            assert gate["accepted"] == c["expect_gate_accepted"]
            assert defect["ran"] == c["expect_defect_ran"]
            
            if act == "BLOCKED":
                assert "supported impeller" in gate["message"].lower()

            print(f"  [PASS] POST /api/inspect ({c['id']}) -> {res['action_label']} ({act}) [p_supp={gate['supported_probability']:.4f}]")

    print("\nALL HTTP ENDPOINTS & ALL 4 OUTCOMES FULLY VERIFIED OVER HTTP!")

if __name__ == "__main__":
    test_demo_endpoints()
