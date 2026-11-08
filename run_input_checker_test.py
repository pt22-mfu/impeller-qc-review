"""Stage 6C: Held-out test evaluation for the Input Checker.

Evaluates on:
- 1,041 supported test images (from outputs/data/split_manifest.csv)
  - 439 OK (good) impellers
  - 602 Defective impellers
- 17 unsupported test images (from outputs/data/unsupported_manifest.csv)

Uses the exact saved threshold (tau = 0.0049) and routing rule from outputs/input_checker/input_checker_policy.json.
Reports:
- Labeled Confusion Matrix [0: Unsupported, 1: Supported]
- Supported images wrongly blocked (FN) and rate
- Supported wrongly blocked breakdown by good (OK) vs defective
- Unsupported images wrongly accepted (FP) and rate
- Unsupported rejection breakdown by category

Saves:
- outputs/input_checker/input_checker_test_scores.csv
- outputs/input_checker/test_metrics.json
"""
import csv
import io
import json
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np

def run_test_evaluation():
    output_dir = Path("outputs/input_checker")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = output_dir / "stage6c_test_outputs.zip"
    scores_path = output_dir / "input_checker_test_scores.csv"
    policy_path = output_dir / "input_checker_policy.json"
    supp_manifest_path = Path("outputs/data/split_manifest.csv")
    unsupp_manifest_path = Path("outputs/data/unsupported_manifest.csv")
    model_path = output_dir / "input_checker_mobilenetv2.keras"
    casting_zip_path = Path("casting_data.zip")

    print("=" * 80)
    print("STAGE 6C: HELD-OUT TEST EVALUATION FOR INPUT CHECKER")
    print("=" * 80)

    # Auto-extract from zip if present
    if not scores_path.is_file() and zip_path.is_file():
        print(f"Extracting outputs from {zip_path}...")
        with ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)

    assert policy_path.is_file(), f"Policy missing: {policy_path}"
    assert supp_manifest_path.is_file(), f"Supported manifest missing: {supp_manifest_path}"
    assert unsupp_manifest_path.is_file(), f"Unsupported manifest missing: {unsupp_manifest_path}"

    # Load locked policy
    with policy_path.open("r", encoding="utf-8") as f:
        policy = json.load(f)
    tau = float(policy["threshold"])
    print(f"Loaded locked policy from {policy_path}")
    print(f"  - Gatekeeper Threshold: tau = {tau:.4f}")
    print(f"  - Decision Rule: Accepted if score >= {tau:.4f}, else Blocked")

    # Extract test rows from manifests to verify
    supp_test_rows = []
    with supp_manifest_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                supp_test_rows.append(r)

    unsupp_test_rows = []
    with unsupp_manifest_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                unsupp_test_rows.append(r)

    n_supp = len(supp_test_rows)
    n_unsupp = len(unsupp_test_rows)
    total_test = n_supp + n_unsupp

    print(f"Manifest Test Verification:")
    print(f"  - Supported test rows:   {n_supp:,} (Expected: 1,041)")
    print(f"  - Unsupported test rows: {n_unsupp:,} (Expected: 17)")
    print(f"  - Total test samples:    {total_test:,}")

    assert n_supp == 1041, f"Expected 1,041 supported test samples, got {n_supp}"
    assert n_unsupp == 17, f"Expected 17 unsupported test samples, got {n_unsupp}"

    supp_ok_count = sum(1 for r in supp_test_rows if r["target"] == "0")
    supp_def_count = sum(1 for r in supp_test_rows if r["target"] == "1")
    assert supp_ok_count == 439, f"Expected 439 OK supported, got {supp_ok_count}"
    assert supp_def_count == 602, f"Expected 602 Defective supported, got {supp_def_count}"

    test_meta = []

    # If scores file already exists, load it directly
    if scores_path.is_file():
        print(f"\nLoading existing test scores from {scores_path}...")
        with scores_path.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                prob = float(r["supported_probability"])
                pred = 1 if prob >= tau else 0
                test_meta.append({
                    "image_path": r["image_path"],
                    "true_target": int(r["true_target"]),
                    "is_supported": int(r.get("is_supported", r["true_target"])),
                    "impeller_defect_label": r.get("impeller_defect_label", "N/A"),
                    "category": r["category"],
                    "supported_probability": prob,
                    "predicted_label": pred,
                    "routing_decision": "Accepted" if pred == 1 else "Blocked"
                })
        print(f"Loaded {len(test_meta):,} scored test samples.")
    else:
        # Need to run inference using TensorFlow
        print(f"\n[INFO] {scores_path} not found. Running inference via TensorFlow...")
        try:
            import tensorflow as tf
            from tensorflow import keras
            from PIL import Image
        except ImportError as e:
            print(f"\n[WAITING] TensorFlow is required for inference but not found: {e}")
            print("Please run evaluate_input_checker_test.ipynb in Colab,")
            print(f"and place input_checker_test_scores.csv or stage6c_test_outputs.zip into {output_dir}/.")
            return None

        # Load images into memory
        target_size = (160, 160)
        X_test = np.empty((total_test, target_size[0], target_size[1], 3), dtype=np.uint8)

        t0 = time.time()
        with ZipFile(casting_zip_path, "r") as zf:
            for idx, r in enumerate(supp_test_rows):
                raw_bytes = zf.read(r["image_path"])
                with Image.open(io.BytesIO(raw_bytes)) as img:
                    rgb = img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
                    X_test[idx] = np.asarray(rgb, dtype=np.uint8)
                test_meta.append({
                    "image_path": r["image_path"],
                    "true_target": 1,
                    "is_supported": 1,
                    "impeller_defect_label": "Defective" if r["target"] == "1" else "OK",
                    "category": "supported_impeller"
                })

        offset = n_supp
        for j, r in enumerate(unsupp_test_rows):
            idx = offset + j
            img_path = Path(r["image_path"])
            with Image.open(img_path) as img:
                rgb = img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
                X_test[idx] = np.asarray(rgb, dtype=np.uint8)
            test_meta.append({
                "image_path": r["image_path"],
                "true_target": 0,
                "is_supported": 0,
                "impeller_defect_label": "N/A",
                "category": r["category"]
            })

        elapsed = time.time() - t0
        print(f"Loaded {total_test:,} images in {elapsed:.2f}s | RAM: {X_test.nbytes / (1024**2):.1f} MB")

        # Load model and predict
        print(f"Loading {model_path}...")
        model = keras.models.load_model(model_path)
        scores = model.predict(X_test, batch_size=32, verbose=1).ravel()

        for idx, s in enumerate(scores):
            prob = float(s)
            test_meta[idx]["supported_probability"] = prob
            pred = 1 if prob >= tau else 0
            test_meta[idx]["predicted_label"] = pred
            test_meta[idx]["routing_decision"] = "Accepted" if pred == 1 else "Blocked"

        # Save test scores
        with scores_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "image_path", "true_target", "is_supported", "impeller_defect_label",
                "category", "supported_probability", "predicted_label", "routing_decision"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(test_meta)
        print(f"Saved test scores to {scores_path}")

    # Compute Evaluation Metrics
    supp_results = [r for r in test_meta if r["is_supported"] == 1]
    unsupp_results = [r for r in test_meta if r["is_supported"] == 0]

    assert len(supp_results) == 1041, f"Expected 1,041 supported test, got {len(supp_results)}"
    assert len(unsupp_results) == 17, f"Expected 17 unsupported test, got {len(unsupp_results)}"

    # Supported evaluation
    supp_acc_count = sum(1 for r in supp_results if r["predicted_label"] == 1)
    supp_blocked_count = sum(1 for r in supp_results if r["predicted_label"] == 0)

    # Supported breakdown by good (OK) vs defective
    ok_results = [r for r in supp_results if r["impeller_defect_label"] == "OK"]
    def_results = [r for r in supp_results if r["impeller_defect_label"] == "Defective"]

    ok_total = len(ok_results)
    def_total = len(def_results)
    ok_blocked = sum(1 for r in ok_results if r["predicted_label"] == 0)
    ok_accepted = sum(1 for r in ok_results if r["predicted_label"] == 1)
    def_blocked = sum(1 for r in def_results if r["predicted_label"] == 0)
    def_accepted = sum(1 for r in def_results if r["predicted_label"] == 1)

    # Unsupported evaluation
    unsupp_rej_count = sum(1 for r in unsupp_results if r["predicted_label"] == 0)
    unsupp_acc_count = sum(1 for r in unsupp_results if r["predicted_label"] == 1)

    # Confusion matrix: Actual [0: Unsupported, 1: Supported], Pred [0: Unsupported, 1: Supported]
    tn = unsupp_rej_count
    fp = unsupp_acc_count
    fn = supp_blocked_count
    tp = supp_acc_count

    # Unsupported breakdown by category
    cat_summary = {}
    for r in unsupp_results:
        cat = r["category"]
        if cat not in cat_summary:
            cat_summary[cat] = {"total": 0, "rejected": 0, "wrongly_accepted": 0, "scores": []}
        cat_summary[cat]["total"] += 1
        cat_summary[cat]["scores"].append(r["supported_probability"])
        if r["predicted_label"] == 0:
            cat_summary[cat]["rejected"] += 1
        else:
            cat_summary[cat]["wrongly_accepted"] += 1

    # Print Formatted Report
    print("\n" + "=" * 80)
    print("STAGE 6C: HELD-OUT TEST EVALUATION RESULTS")
    print("=" * 80)
    print(f"Policy Threshold: tau = {tau:.4f} (from {policy_path.name})")
    print(f"Decision Rule:    supported_probability >= {tau:.4f} -> Accepted, else Blocked")
    print("-" * 80)

    print("\n1. CONFUSION MATRIX (Labels: [0: Unsupported, 1: Supported]):")
    print(f"{'':22} | {'Pred Unsupported (0)':<22} | {'Pred Supported (1)':<20} | {'Total'}")
    print("-" * 80)
    print(f"{'Actual Unsupported (0)':<22} | {tn:<22d} | {fp:<20d} | {tn+fp}")
    print(f"{'Actual Supported (1)':<22}   | {fn:<22d} | {tp:<20d} | {fn+tp}")
    print("-" * 80)
    print(f"{'Total':<22} | {tn+fn:<22d} | {fp+tp:<20d} | {total_test}")

    print("\n2. ERROR COUNTS & ACCURACY:")
    print(f"  - Supported images wrongly blocked (FN):   {fn:,} / {n_supp:,} ({fn / n_supp:.2%})")
    print(f"  - Unsupported images wrongly accepted (FP): {fp:,} / {n_unsupp:,} ({fp / n_unsupp:.2%})")
    print(f"  - Supported Acceptance Rate:                {tp:,} / {n_supp:,} ({tp / n_supp:.2%})")
    print(f"  - Unsupported Rejection Rate:               {tn:,} / {n_unsupp:,} ({tn / n_unsupp:.2%})")
    print(f"  - Overall Held-Out Test Accuracy:           {(tp + tn):,} / {total_test:,} ({(tp + tn) / total_test:.2%})")

    print("\n3. SUPPORTED IMAGES WRONGLY BLOCKED BREAKDOWN (GOOD VS. DEFECTIVE):")
    print(f"  - Good / OK Impellers:       {ok_blocked:,} wrongly blocked / {ok_total:,} ({ok_blocked / ok_total:.2%}) | {ok_accepted:,} accepted ({ok_accepted / ok_total:.2%})")
    print(f"  - Defective Impellers:       {def_blocked:,} wrongly blocked / {def_total:,} ({def_blocked / def_total:.2%}) | {def_accepted:,} accepted ({def_accepted / def_total:.2%})")
    print(f"  - Total Supported Impellers: {fn:,} wrongly blocked / {n_supp:,} ({fn / n_supp:.2%}) | {tp:,} accepted ({tp / n_supp:.2%})")

    print("\n4. UNSUPPORTED REJECTION BREAKDOWN BY CATEGORY:")
    print(f"{'Category':<26} | {'Total':<6} | {'Rejected':<10} | {'Wrongly Acc':<12} | {'Rejection Rate':<14} | {'Mean Score':<10} | {'Max Score'}")
    print("-" * 96)
    for cat, stats in sorted(cat_summary.items()):
        rej_pct = stats["rejected"] / stats["total"] * 100.0
        m_score = float(np.mean(stats["scores"]))
        max_s = float(np.max(stats["scores"]))
        print(f"{cat:<26} | {stats['total']:<6} | {stats['rejected']:<10} | {stats['wrongly_accepted']:<12} | {rej_pct:>12.1f}%  | {m_score:<10.4f} | {max_s:.4f}")
    print("=" * 96)

    # Save Test Metrics JSON
    metrics_path = output_dir / "test_metrics.json"
    test_metrics = {
        "evaluation_split": "held_out_test",
        "threshold": float(tau),
        "total_test_samples": total_test,
        "overall_accuracy": float((tp + tn) / total_test),
        "supported_evaluation": {
            "total_supported": n_supp,
            "supported_accepted": tp,
            "supported_wrongly_blocked": fn,
            "supported_acceptance_rate": float(tp / n_supp),
            "breakdown_by_impeller_defect_status": {
                "ok_good_impellers": {
                    "total": ok_total,
                    "accepted": ok_accepted,
                    "wrongly_blocked": ok_blocked,
                    "wrongly_blocked_rate": float(ok_blocked / ok_total)
                },
                "defective_impellers": {
                    "total": def_total,
                    "accepted": def_accepted,
                    "wrongly_blocked": def_blocked,
                    "wrongly_blocked_rate": float(def_blocked / def_total)
                }
            }
        },
        "unsupported_evaluation": {
            "total_unsupported": n_unsupp,
            "unsupported_rejected": tn,
            "unsupported_wrongly_accepted": fp,
            "unsupported_rejection_rate": float(tn / n_unsupp),
            "category_breakdown": {
                cat: {
                    "total": stats["total"],
                    "rejected": stats["rejected"],
                    "wrongly_accepted": stats["wrongly_accepted"],
                    "rejection_rate": float(stats["rejected"] / stats["total"]),
                    "mean_score": float(np.mean(stats["scores"])),
                    "max_score": float(np.max(stats["scores"]))
                }
                for cat, stats in sorted(cat_summary.items())
            }
        },
        "confusion_matrix": {
            "labels": ["Unsupported (0)", "Supported (1)"],
            "matrix": [
                [tn, fp],
                [fn, tp]
            ],
            "true_unsupported_pred_unsupported_TN": tn,
            "true_unsupported_pred_supported_FP": fp,
            "true_supported_pred_unsupported_FN": fn,
            "true_supported_pred_supported_TP": tp
        }
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nSaved test metrics to {metrics_path}")

    return test_metrics

if __name__ == "__main__":
    run_test_evaluation()
