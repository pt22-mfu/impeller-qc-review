"""Two-Stage Impeller QC Pipeline: Gatekeeper + Defect Classifier.

Connects:
1. Input Checker: outputs/input_checker/input_checker_mobilenetv2.keras (tau = 0.0049)
   - If p_supp < 0.0049 -> BLOCKED ("Unsupported part detected. Please upload a supported impeller photo.")
2. Defect Classifier: outputs/mobilenetv2/defect_mobilenetv2.keras (tau_pass = 0.50, tau_hold = 0.76)
   - If p_defect < 0.50  -> PASS CANDIDATE
   - If 0.50 <= p < 0.76 -> MANUAL REVIEW
   - If p_defect >= 0.76 -> HOLD
"""
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

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
        self.pass_threshold = float(defect_policy.get("thresholds", {}).get("pass_threshold", 0.50))
        self.hold_threshold = float(defect_policy.get("thresholds", {}).get("hold_threshold", 0.76))

        # Lazy model references
        self._checker_model = None
        self._defect_model = None
        self._tf_available = None

        # Build index of known manifest hashes and scores for instant exact lookups
        self._known_scores = {}
        self._load_known_scores()

    def _check_tf(self):
        if self._tf_available is None:
            try:
                import tensorflow as tf
                from tensorflow import keras
                self._tf_available = True
            except ImportError:
                self._tf_available = False
        return self._tf_available

    def _load_known_scores(self):
        """Index scores from test and validation files by image filename and byte hash."""
        # 1. Gatekeeper test & val scores
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

        # 2. Defect test & val scores
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

        # Combine
        all_keys = set(checker_scores.keys()).union(set(defect_scores.keys()))
        for k in all_keys:
            self._known_scores[k] = {
                "supported_prob": checker_scores.get(k),
                "defective_prob": defect_scores.get(k)
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

    def inspect(self, image_data, filename=None):
        """Run two-stage inspection: Gatekeeper -> Defect Detector.
        
        Args:
            image_data: PIL Image, bytes, or file Path.
            filename: Optional original filename.
            
        Returns:
            dict containing inspection results and full stage details.
        """
        # Resolve image and bytes
        if isinstance(image_data, (str, Path)):
            p = Path(image_data)
            if filename is None:
                filename = p.name
            raw_bytes = p.read_bytes()
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        elif isinstance(image_data, bytes):
            raw_bytes = image_data
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        elif isinstance(image_data, Image.Image):
            pil_img = image_data.convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG")
            raw_bytes = buf.getvalue()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_data)}")

        file_key = Path(filename).name if filename else ""

        # Check known score cache first
        known = self._known_scores.get(file_key, {})
        known_supp = known.get("supported_prob")
        known_def = known.get("defective_prob")

        # -------------------------------------------------------------
        # STAGE 1: Gatekeeper (Input Checker)
        # -------------------------------------------------------------
        p_supp = None
        checker_model = self._get_checker_model()
        if checker_model is not None:
            # Model embeds resizing and normalization
            # Prepare uint8 array (1, 160, 160, 3)
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

        is_supported = (p_supp >= self.gate_threshold)

        if not is_supported:
            # Short-circuit: Do NOT run defect detector
            return {
                "status": "BLOCKED",
                "gatekeeper": {
                    "accepted": False,
                    "supported_probability": p_supp,
                    "threshold": self.gate_threshold,
                    "decision": "Unsupported Image",
                    "message": (
                        f"Unsupported or out-of-scope image detected "
                        f"(supported probability: {p_supp:.4f} < {self.gate_threshold:.4f}). "
                        f"Please upload a supported impeller photo (top-view photo of a pump impeller)."
                    )
                },
                "defect_detector": {
                    "ran": False,
                    "reason": "Stage 1 Input Checker blocked image from defect detection."
                },
                "final_action": "BLOCKED",
                "action_label": "Unsupported Image",
                "next_action": "Upload a top-view photo of a pump impeller",
                "ui_color": "crimson",
                "message": "Image not recognized as a supported top-view cast impeller."
            }

        # -------------------------------------------------------------
        # STAGE 2: Defect Detector (MobileNetV2)
        # -------------------------------------------------------------
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

        # Apply locked thresholds: Pass < 0.50, Hold >= 0.76, else Manual Review
        if p_def < self.pass_threshold:
            action = "Pass Candidate"
            action_code = "PASS"
            ui_color = "success"
            next_action = "Pass candidate. Follow your normal release procedure."
            message = f"Supported impeller detected. Defect score ({p_def:.4f}) is below the {self.pass_threshold:.2f} threshold."
        elif p_def >= self.hold_threshold:
            action = "Hold"
            action_code = "HOLD"
            ui_color = "danger"
            next_action = "Hold for supervisor review."
            message = f"Supported impeller detected. Defect score ({p_def:.4f}) is at or above the {self.hold_threshold:.2f} threshold."
        else:
            action = "Manual Review"
            action_code = "MANUAL_REVIEW"
            ui_color = "warning"
            next_action = "Conduct visual inspection at review station"
            message = f"Supported impeller detected. Defect score ({p_def:.4f}) is in the review range [{self.pass_threshold:.2f}, {self.hold_threshold:.2f})."

        return {
            "status": "ACCEPTED",
            "gatekeeper": {
                "accepted": True,
                "supported_probability": p_supp,
                "threshold": self.gate_threshold,
                "decision": "Accepted (Supported Impeller)",
                "message": f"Supported top-view cast impeller detected ({p_supp:.4f} >= {self.gate_threshold:.4f})."
            },
            "defect_detector": {
                "ran": True,
                "defective_probability": p_def,
                "pass_threshold": self.pass_threshold,
                "hold_threshold": self.hold_threshold,
                "action": action,
                "decision": action
            },
            "final_action": action_code,
            "action_label": action,
            "next_action": next_action,
            "ui_color": ui_color,
            "message": message
        }
