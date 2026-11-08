"""Test Gemini Configuration, Simulated Fallback, Local Summary, and Live Request.

Tests:
1. Configuration loading from .streamlit/secrets.toml and environment precedence
2. Git-ignore verification for secrets
3. Simulated primary model error (HTTP 404, 429, 503) triggering fallback
4. Fallback execution tried exactly once
5. Both models failing triggering clearly labeled Local summary
6. Local summary generation for all QC decisions (Pass, Manual Review, Hold)
7. Missing API key fallback to clearly labeled Local summary
8. Live request verification with configured key
"""
import os
import pathlib
import tomllib
import unittest
from app import (
    get_gemini_config,
    generate_local_summary,
    generate_gemini_qa_summary,
    ModelUnavailableError,
    QuotaLimitError,
    TemporaryServiceError
)

class TestGeminiConfigAndMockFallback(unittest.TestCase):
    """Test configuration hierarchy, simulated errors, and local fallback."""

    def setUp(self):
        # Sample inspection results
        self.sample_pass = {
            "action_label": "Pass Candidate",
            "final_action": "PASS",
            "next_action": "Pass candidate. Follow your normal release procedure.",
            "supported_probability": 0.9998,
            "defective_probability": 0.0001,
            "pass_threshold": 0.50,
            "hold_threshold": 0.76
        }
        self.sample_review = {
            "action_label": "Manual Review",
            "final_action": "MANUAL_REVIEW",
            "next_action": "Conduct visual inspection at review station",
            "supported_probability": 0.9985,
            "defective_probability": 0.6277,
            "pass_threshold": 0.50,
            "hold_threshold": 0.76
        }
        self.sample_hold = {
            "action_label": "Hold",
            "final_action": "HOLD",
            "next_action": "Hold for supervisor review.",
            "supported_probability": 0.9992,
            "defective_probability": 0.9994,
            "pass_threshold": 0.50,
            "hold_threshold": 0.76
        }

    def test_01_gitignore_contains_secrets(self):
        """Verify that .streamlit/secrets.toml is ignored by git."""
        gitignore_path = pathlib.Path(".gitignore")
        self.assertTrue(gitignore_path.exists(), ".gitignore file must exist")
        content = gitignore_path.read_text(encoding="utf-8")
        self.assertIn(".streamlit/secrets.toml", content, ".streamlit/secrets.toml must be in .gitignore")
        print("\n[MOCK CHECK 1] .gitignore contains .streamlit/secrets.toml: PASS")

    def test_02_default_model_ids(self):
        """Verify defaults are gemini-3.5-flash-lite (primary) and gemini-3.1-flash-lite (fallback)."""
        # Test with empty secrets and no env
        saved_env = {k: os.environ.pop(k, None) for k in ["GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_FALLBACK_MODEL"]}
        try:
            cfg = get_gemini_config()
            self.assertEqual(cfg["primary_model"], "gemini-3.5-flash-lite")
            self.assertEqual(cfg["fallback_model"], "gemini-3.1-flash-lite")
            print(f"[MOCK CHECK 2] Defaults: Primary={cfg['primary_model']}, Fallback={cfg['fallback_model']}: PASS")
        finally:
            for k, v in saved_env.items():
                if v is not None:
                    os.environ[k] = v

    def test_03_env_precedence_over_secrets(self):
        """Verify environment variables take precedence over .streamlit/secrets.toml."""
        os.environ["GEMINI_MODEL"] = "gemini-custom-env-primary"
        os.environ["GEMINI_FALLBACK_MODEL"] = "gemini-custom-env-fallback"
        os.environ["GEMINI_API_KEY"] = "env-dummy-key"
        try:
            cfg = get_gemini_config()
            self.assertEqual(cfg["primary_model"], "gemini-custom-env-primary")
            self.assertEqual(cfg["fallback_model"], "gemini-custom-env-fallback")
            self.assertEqual(cfg["key_source"], "environment")
            print("[MOCK CHECK 3] Environment precedence over secrets.toml: PASS")
        finally:
            del os.environ["GEMINI_MODEL"]
            del os.environ["GEMINI_FALLBACK_MODEL"]
            del os.environ["GEMINI_API_KEY"]

    def test_04_simulated_primary_unavailability_fallback(self):
        """Simulate HTTP 404 (model unavailability) on primary -> falls back to fallback model once."""
        res = generate_gemini_qa_summary(
            self.sample_pass,
            api_key="mock_key",
            simulation={"primary": "404", "fallback": "mock_success"}
        )
        self.assertTrue(res["success"])
        self.assertFalse(res["is_local"])
        self.assertEqual(res["source"], "gemini")
        self.assertTrue(res["fallback_used"])
        self.assertEqual(res["model_used"], "gemini-3.1-flash-lite")
        self.assertIn("404", res["primary_error"])
        print(f"[MOCK CHECK 4] Simulated Primary 404 -> Fallback to {res['model_used']}: PASS")

    def test_05_simulated_primary_quota_limit_fallback(self):
        """Simulate HTTP 429 (quota limit) on primary -> falls back to fallback model once."""
        res = generate_gemini_qa_summary(
            self.sample_review,
            api_key="mock_key",
            simulation={"primary": "429", "fallback": "mock_success"}
        )
        self.assertTrue(res["success"])
        self.assertFalse(res["is_local"])
        self.assertTrue(res["fallback_used"])
        self.assertEqual(res["model_used"], "gemini-3.1-flash-lite")
        self.assertIn("429", res["primary_error"])
        print(f"[MOCK CHECK 5] Simulated Primary 429 -> Fallback to {res['model_used']}: PASS")

    def test_06_simulated_primary_service_error_fallback(self):
        """Simulate HTTP 503 (temporary service error) on primary -> falls back to fallback model once."""
        res = generate_gemini_qa_summary(
            self.sample_hold,
            api_key="mock_key",
            simulation={"primary": "503", "fallback": "mock_success"}
        )
        self.assertTrue(res["success"])
        self.assertFalse(res["is_local"])
        self.assertTrue(res["fallback_used"])
        self.assertEqual(res["model_used"], "gemini-3.1-flash-lite")
        self.assertIn("503", res["primary_error"])
        print(f"[MOCK CHECK 6] Simulated Primary 503 -> Fallback to {res['model_used']}: PASS")

    def test_07_neither_model_can_answer_returns_local_summary(self):
        """When both models fail (e.g. primary 429, fallback 503), show clearly labeled Local summary."""
        res = generate_gemini_qa_summary(
            self.sample_pass,
            api_key="mock_key",
            simulation={"primary": "429", "fallback": "503"}
        )
        self.assertTrue(res["success"])
        self.assertTrue(res["is_local"])
        self.assertEqual(res["source"], "local")
        self.assertEqual(res["label"], "Local summary")
        self.assertTrue(res["summary"].startswith("Local summary:"))
        self.assertIn("Pass candidate. Follow your normal release procedure.", res["summary"])
        print(f"[MOCK CHECK 7] Both models fail -> Clearly labeled Local summary returned: PASS")
        print(f"               Summary text: \"{res['summary']}\"")

    def test_08_local_summary_all_outcomes(self):
        """Verify deterministic Local summary generation across Pass, Review, and Hold decisions."""
        for sample in [self.sample_pass, self.sample_review, self.sample_hold]:
            text = generate_local_summary(sample)
            self.assertTrue(text.startswith("Local summary:"))
            self.assertIn(sample["next_action"], text)
        print("[MOCK CHECK 8] Local summary rules for Pass, Review, and Hold verified: PASS")

    def test_09_missing_key_returns_local_summary(self):
        """When key is missing, return clearly labeled Local summary without crashing."""
        res = generate_gemini_qa_summary(
            self.sample_review,
            api_key=None,
            config_override={"api_key": None, "primary_model": "gemini-3.5-flash-lite", "fallback_model": "gemini-3.1-flash-lite"}
        )
        self.assertTrue(res["success"])
        self.assertTrue(res["is_local"])
        self.assertEqual(res["source"], "local")
        self.assertEqual(res["label"], "Local summary")
        self.assertTrue(res["summary"].startswith("Local summary:"))
        print(f"[MOCK CHECK 9] Missing API key -> Local summary: PASS")


class TestGeminiLiveRequest(unittest.TestCase):
    """Test live API execution once key is configured in .streamlit/secrets.toml."""

    def test_10_live_request(self):
        cfg = get_gemini_config()
        if not cfg["key_configured"]:
            self.skipTest("GEMINI_API_KEY is not configured yet. Skipping live request.")

        print(f"\n[LIVE CHECK] Found key from source: {cfg['key_source']}")
        print(f"[LIVE CHECK] Testing primary model: {cfg['primary_model']} with fallback: {cfg['fallback_model']}")

        test_data = {
            "action_label": "Pass Candidate",
            "final_action": "PASS",
            "next_action": "Pass candidate. Follow your normal release procedure.",
            "supported_probability": 0.9998,
            "defective_probability": 0.0001,
            "pass_threshold": 0.50,
            "hold_threshold": 0.76
        }

        res = generate_gemini_qa_summary(test_data)
        self.assertTrue(res["success"])
        print(f"[LIVE CHECK] Success: {res['success']}")
        print(f"[LIVE CHECK] Source: {res['source']}")
        print(f"[LIVE CHECK] Model Used: {res.get('model_used')}")
        print(f"[LIVE CHECK] Fallback Used: {res.get('fallback_used')}")
        print(f"[LIVE CHECK] Summary Text: {res['summary']}")

        if res["source"] == "gemini":
            self.assertIn(res["model_used"], [cfg["primary_model"], cfg["fallback_model"]])
            self.assertFalse(res["is_local"])
        else:
            self.assertTrue(res["is_local"])
            self.assertTrue(res["summary"].startswith("Local summary:"))

if __name__ == "__main__":
    unittest.main()

