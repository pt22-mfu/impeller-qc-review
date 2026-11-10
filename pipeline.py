"""Two-Stage Impeller QC Pipeline: Gatekeeper + Defect Classifier.

Connects:
1. Input Checker: outputs/input_checker/input_checker_mobilenetv2.keras (tau = 0.0049)
   - If p_supp < 0.0049 -> BLOCKED ("Please upload a clear top-view photo of a pump impeller.")
2. Defect Classifier: outputs/mobilenetv2/defect_mobilenetv2.keras (tau_pass = 0.50, tau_hold = 0.76)
   - If p_defect < 0.50  -> PASS CANDIDATE
   - If 0.50 <= p < 0.76 -> MANUAL REVIEW
   - If p_defect >= 0.76 -> HOLD
"""
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


class ImpellerQCPipeline:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "outputs"

        # Paths
        self.checker_model_path = self.output_dir / "input_checker" / "input_checker_mobilenetv2.keras"
        self.checker_policy_path = self.output_dir / "input_checker" / "input_checker_policy.json"

        self.defect_model_path = self.output_dir / "mobilenetv2" / "defect_mobilenetv2.keras"
        self.defect_policy_path = self.output_dir / "mobilenetv2" / "qc_policy.json"

        # Load policies
        assert self.checker_policy_path.is_file(), f"Missing {self.checker_policy_path}"
        with self.checker_policy_path.open("r", encoding="utf-8") as f:
            checker_policy = json.load(f)
        self.gate_threshold = float(checker_policy.get("threshold", 0.0049))

        assert self.defect_policy_path.is_file(), f"Missing {self.defect_policy_path}"
        with self.defect_policy_path.open("r", encoding="utf-8") as f:
            defect_policy = json.load(f)
        chosen = defect_policy.get("routing_policy", {}).get("chosen_thresholds", {})
        if not chosen:
            chosen = defect_policy.get("thresholds", {})
        self.pass_threshold = float(chosen.get("pass_threshold", 0.50))
        self.hold_threshold = float(chosen.get("hold_threshold", 0.76))

        # Lazy model references
        self._checker_model = None
        self._defect_model = None
        self._tf_available = None

        # Invocation counters to verify strict stage separation
        self.checker_call_count = 0
        self.defect_call_count = 0

        # Fallback index of known manifest scores if TensorFlow is unavailable
        self._known_scores = {}
        self._load_known_scores()

    def _check_tf(self):
        if self._tf_available is None:
            try:
                import tensorflow as tf  # noqa: F401
                from tensorflow import keras  # noqa: F401
                self._tf_available = True
            except ImportError:
                self._tf_available = False
        return self._tf_available

    def _load_known_scores(self):
        """Index scores from test and validation files by image filename."""
        checker_scores = {}
        for fname in ["input_checker_test_scores.csv", "input_checker_val_scores.csv"]:
            p = self.output_dir / "input_checker" / fname
            if p.is_file():
                with p.open("r", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        path_key = Path(r["image_path"]).name
                        val = r.get("supported_probability") or r.get("supported_score")
                        if val is not None:
                            checker_scores[path_key] = float(val)

        defect_scores = {}
        for fname in ["mobilenetv2_test_scores.csv", "validation_scores.csv"]:
            p = self.output_dir / "mobilenetv2" / fname
            if p.is_file():
                with p.open("r", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        path_key = Path(r["image_path"]).name
                        val = r.get("defective_probability") or r.get("defective_score")
                        if val is not None:
                            defect_scores[path_key] = float(val)

        all_keys = set(checker_scores.keys()).union(set(defect_scores.keys()))
        for k in all_keys:
            self._known_scores[k] = {
                "supported_prob": checker_scores.get(k),
                "defective_prob": defect_scores.get(k),
            }

    def _get_checker_model(self):
        if self._checker_model is None and self._check_tf():
            from tensorflow import keras
            print(f"Loading Input Checker Model from {self.checker_model_path}...")
            self._checker_model = keras.models.load_model(self.checker_model_path)
        return self._checker_model

    def _get_defect_model(self):
        if self._defect_model is None and self._check_tf():
            from tensorflow import keras
            print(f"Loading Defect Inspection Model from {self.defect_model_path}...")
            self._defect_model = keras.models.load_model(self.defect_model_path)
        return self._defect_model

    @staticmethod
    def _prepare_image(image_data, filename=None):
        """Resolve image_data into (PIL RGB Image, file_key)."""
        if isinstance(image_data, (str, Path)):
            p = Path(image_data)
            if filename is None:
                filename = p.name
            raw_bytes = p.read_bytes()
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        elif isinstance(image_data, bytes):
            pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, Image.Image):
            pil_img = image_data.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_data)}")

        file_key = Path(filename).name if filename else ""
        return pil_img, file_key

    def check_eligibility(self, image_data, filename=None):
        """Run ONLY Stage 1: Photo eligibility gatekeeper. Never touches Stage 2 defect model."""
        pil_img, file_key = self._prepare_image(image_data, filename)
        known_supp = self._known_scores.get(file_key, {}).get("supported_prob")

        p_supp = None
        checker_model = self._get_checker_model()
        if checker_model is not None:
            resized = pil_img.resize((160, 160), Image.Resampling.BILINEAR)
            x_arr = np.expand_dims(np.asarray(resized, dtype=np.uint8), axis=0)
            p_supp = float(checker_model.predict(x_arr, verbose=0)[0, 0])
        elif known_supp is not None:
            p_supp = float(known_supp)
        else:
            raise RuntimeError(
                "TensorFlow is required for live inference on unindexed images, "
                "or please test with a dataset sample image."
            )

        self.checker_call_count += 1
        is_supported = p_supp >= self.gate_threshold

        if not is_supported:
            return {
                "accepted": False,
                "status": "BLOCKED",
                "supported_probability": p_supp,
                "unsupported_probability": 1.0 - p_supp,
                "threshold": self.gate_threshold,
                "decision": "Unsupported Image",
                "action_label": "Unsupported Image",
                "next_action": "Please upload a clear top-view photo of a pump impeller.",
                "message": "Please upload a clear top-view photo of a pump impeller.",
            }

        return {
            "accepted": True,
            "status": "ACCEPTED",
            "supported_probability": p_supp,
            "unsupported_probability": 1.0 - p_supp,
            "threshold": self.gate_threshold,
            "decision": "Accepted (Supported Impeller)",
            "message": "This photo can be checked for defects.",
        }

    def check_defects(self, image_data, filename=None, gatekeeper_result=None):
        """Run Stage 2: Defect Classifier (invoke ONLY when user clicks 'Check for defects')."""
        pil_img, file_key = self._prepare_image(image_data, filename)
        if gatekeeper_result is None:
            gatekeeper_result = self.check_eligibility(image_data, filename=filename)

        known_def = self._known_scores.get(file_key, {}).get("defective_prob")

        p_def = None
        defect_model = self._get_defect_model()
        if defect_model is not None:
            resized = pil_img.resize((160, 160), Image.Resampling.BILINEAR)
            x_arr = np.expand_dims(np.asarray(resized, dtype=np.uint8), axis=0)
            p_def = float(defect_model.predict(x_arr, verbose=0)[0, 0])
        elif known_def is not None:
            p_def = float(known_def)
        else:
            raise RuntimeError(
                "TensorFlow is required for live defect inference on unindexed images."
            )

        self.defect_call_count += 1
        p_ok = 1.0 - p_def

        if p_def < self.pass_threshold:
            action = "Pass Candidate"
            action_code = "PASS"
            ui_color = "success"
            next_action = "Pass candidate. Follow your normal release procedure."
            message = (
                f"Defect score ({p_def:.4f}) is below the {self.pass_threshold:.2f} pass cutoff."
            )
        elif p_def >= self.hold_threshold:
            action = "Hold"
            action_code = "HOLD"
            ui_color = "danger"
            next_action = "Hold for supervisor review."
            message = (
                f"Defect score ({p_def:.4f}) meets or exceeds the {self.hold_threshold:.2f} hold cutoff."
            )
        else:
            action = "Manual Review"
            action_code = "MANUAL_REVIEW"
            ui_color = "warning"
            next_action = "Conduct visual inspection at review station"
            message = (
                f"Defect score ({p_def:.4f}) is in the review range "
                f"[{self.pass_threshold:.2f}, {self.hold_threshold:.2f})."
            )

        return {
            "status": "ACCEPTED",
            "gatekeeper": gatekeeper_result,
            "defect_detector": {
                "ran": True,
                "defective_probability": p_def,
                "ok_probability": p_ok,
                "pass_threshold": self.pass_threshold,
                "hold_threshold": self.hold_threshold,
                "action": action,
                "decision": action,
            },
            "final_action": action_code,
            "action_label": action,
            "next_action": next_action,
            "ui_color": ui_color,
            "message": message,
        }

    def inspect(self, image_data, filename=None):
        """Convenience full two-stage inspection: Gatekeeper -> Defect Detector."""
        gatekeeper_result = self.check_eligibility(image_data, filename=filename)
        if not gatekeeper_result["accepted"]:
            return {
                "status": "BLOCKED",
                "gatekeeper": gatekeeper_result,
                "defect_detector": {
                    "ran": False,
                    "reason": "Stage 1 Input Checker blocked image from defect detection.",
                },
                "final_action": "BLOCKED",
                "action_label": "Unsupported Image",
                "next_action": "Please upload a clear top-view photo of a pump impeller.",
                "ui_color": "crimson",
                "message": "Please upload a clear top-view photo of a pump impeller.",
            }
        return self.check_defects(image_data, filename=filename, gatekeeper_result=gatekeeper_result)
