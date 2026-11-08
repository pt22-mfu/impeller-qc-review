"""Compare Baseline and MobileNetV2 on the held-out test set and evaluate locked QC policy.

Verifies:
  1. Both baseline_test_scores.csv and mobilenetv2_test_scores.csv match
     outputs/data/split_manifest.csv exactly by image_path and true_target.
  2. Compares test accuracy, precision, recall, F1, and confusion matrices at 0.50.
  3. Verifies that qc_policy.json specifies Pass = 0.50 and Hold = 0.76, then routes
     the MobileNetV2 test predictions accordingly.
  4. Exports outputs/test_benchmark.json.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def load_manifest_test_rows(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [r for r in reader if r["split"] == "test"]


def load_and_verify_scores(csv_path: Path, manifest_test_rows: list[dict], model_name: str):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {model_name} scores file: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if len(reader) != len(manifest_test_rows):
        raise ValueError(
            f"Row count mismatch for {model_name}: expected {len(manifest_test_rows)}, got {len(reader)}"
        )

    for i, (score_row, manifest_row) in enumerate(zip(reader, manifest_test_rows)):
        if score_row["image_path"] != manifest_row["image_path"]:
            raise ValueError(
                f"Path mismatch at index {i} for {model_name}:\n"
                f"  Scores:   {score_row['image_path']}\n"
                f"  Manifest: {manifest_row['image_path']}"
            )
        if int(score_row["true_target"]) != int(manifest_row["target"]):
            raise ValueError(
                f"Target mismatch at index {i} ({score_row['image_path']}) for {model_name}:\n"
                f"  Scores:   {score_row['true_target']}\n"
                f"  Manifest: {manifest_row['target']}"
            )

    return reader


def compute_metrics(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1))
    rec = float(recall_score(y_true, y_pred, pos_label=1))
    f1 = float(f1_score(y_true, y_pred, pos_label=1))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": {
            "class_order": ["OK", "Defective"],
            "matrix": [[tn, fp], [fn, tp]],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def run_test_comparison(
    manifest_path: Path = Path("outputs/data/split_manifest.csv"),
    baseline_csv_path: Path = Path("outputs/baseline/baseline_test_scores.csv"),
    mobilenet_csv_path: Path = Path("outputs/mobilenetv2/mobilenetv2_test_scores.csv"),
    qc_policy_path: Path = Path("outputs/mobilenetv2/qc_policy.json"),
    output_benchmark_path: Path = Path("outputs/test_benchmark.json"),
):
    print("=" * 75)
    print("Stage 5: Final Held-Out Test Evaluation & Comparison Benchmark")
    print("=" * 75)

    # 1. Load and verify manifest test rows
    print(f"\n[1/5] Loading test manifest from: {manifest_path}")
    manifest_test_rows = load_manifest_test_rows(manifest_path)
    total_test = len(manifest_test_rows)
    ok_count = sum(1 for r in manifest_test_rows if int(r["target"]) == 0)
    def_count = sum(1 for r in manifest_test_rows if int(r["target"]) == 1)

    print(f"      Verified {total_test:,} held-out test rows (OK={ok_count:,}, Defective={def_count:,})")
    assert total_test == 1041, f"Expected 1041 test samples, got {total_test}"

    # 2. Check baseline score file
    print(f"\n[2/5] Verifying Baseline scores from: {baseline_csv_path}")
    baseline_rows = load_and_verify_scores(baseline_csv_path, manifest_test_rows, "Baseline")
    print(f"      Baseline scores verified ({len(baseline_rows):,} rows match manifest exactly).")

    # 3. Check MobileNetV2 score file
    if not mobilenet_csv_path.is_file():
        alt_path = Path("mobilenetv2_test_scores.csv")
        if alt_path.is_file():
            mobilenet_csv_path = alt_path

    if not mobilenet_csv_path.is_file():
        print(f"\n[!] Awaiting MobileNetV2 test scores file: {mobilenet_csv_path}")
        print("    Please run evaluate_mobilenetv2_test.ipynb in Colab and place")
        print("    mobilenetv2_test_scores.csv into outputs/mobilenetv2/.")
        return False

    print(f"\n[3/5] Verifying MobileNetV2 scores from: {mobilenet_csv_path}")
    mobilenet_rows = load_and_verify_scores(mobilenet_csv_path, manifest_test_rows, "MobileNetV2")
    print(f"      MobileNetV2 scores verified ({len(mobilenet_rows):,} rows match manifest exactly).")

    # Extract arrays
    y_test = np.array([int(r["target"]) for r in manifest_test_rows], dtype=np.int64)
    baseline_scores = np.array([float(r["defective_score"]) for r in baseline_rows])
    mobilenet_scores = np.array([float(r["defective_score"]) for r in mobilenet_rows])

    CUTOFF = 0.50
    baseline_preds = (baseline_scores >= CUTOFF).astype(int)
    mobilenet_preds = (mobilenet_scores >= CUTOFF).astype(int)

    base_metrics = compute_metrics(y_test, baseline_preds)
    mobile_metrics = compute_metrics(y_test, mobilenet_preds)

    # 4. Report side-by-side comparison
    print(f"\n[4/5] Held-Out Test Evaluation Comparison (Cutoff = {CUTOFF:.2f}):")
    print("+" + "-" * 73 + "+")
    print(f"| {'Metric':<25} | {'Baseline (LogReg)':>18} | {'MobileNetV2':>14} | {'Gain':>8} |")
    print("+" + "-" * 73 + "+")
    metrics_display = [
        ("Test Accuracy", base_metrics["accuracy"], mobile_metrics["accuracy"]),
        ("Defective Precision", base_metrics["precision"], mobile_metrics["precision"]),
        ("Defective Recall", base_metrics["recall"], mobile_metrics["recall"]),
        ("Defective F1-score", base_metrics["f1"], mobile_metrics["f1"]),
    ]
    for label, b_val, m_val in metrics_display:
        diff = m_val - b_val
        print(f"| {label:<25} | {b_val:>17.2%} | {m_val:>13.2%} | {diff:>+7.2%} |")
    print("+" + "-" * 73 + "+")

    b_cm = base_metrics["confusion_matrix"]
    m_cm = mobile_metrics["confusion_matrix"]
    print(f"| {'True OK (TN)':<25} | {b_cm['tn']:>18,d} | {m_cm['tn']:>14,d} | {m_cm['tn']-b_cm['tn']:>+8d} |")
    print(f"| {'False Defective (FP)':<25} | {b_cm['fp']:>18,d} | {m_cm['fp']:>14,d} | {m_cm['fp']-b_cm['fp']:>+8d} |")
    print(f"| {'False OK / Missed (FN)':<25} | {b_cm['fn']:>18,d} | {m_cm['fn']:>14,d} | {m_cm['fn']-b_cm['fn']:>+8d} |")
    print(f"| {'True Defective (TP)':<25} | {b_cm['tp']:>18,d} | {m_cm['tp']:>14,d} | {m_cm['tp']-b_cm['tp']:>+8d} |")
    print("+" + "-" * 73 + "+")

    # 5. Apply Locked QC Policy to MobileNetV2
    print(f"\n[5/5] Applying Locked QC Policy from: {qc_policy_path}")
    assert qc_policy_path.is_file(), f"Missing {qc_policy_path}"
    with qc_policy_path.open("r", encoding="utf-8") as f:
        policy_data = json.load(f)

    chosen_thresh = policy_data["routing_policy"]["chosen_thresholds"]
    pass_thresh = float(chosen_thresh["pass_threshold"])
    hold_thresh = float(chosen_thresh["hold_threshold"])

    print(f"      Loaded Thresholds: Pass = {pass_thresh:.2f}, Hold = {hold_thresh:.2f}")
    assert round(pass_thresh, 2) == 0.50, f"Expected pass_threshold 0.50, got {pass_thresh}"
    assert round(hold_thresh, 2) == 0.76, f"Expected hold_threshold 0.76, got {hold_thresh}"
    print("      Thresholds verified: Pass = 0.50, Hold = 0.76 strictly locked.")

    # Route MobileNetV2 scores
    pass_ok = 0
    pass_def = 0
    hold_ok = 0
    hold_def = 0
    rev_ok = 0
    rev_def = 0

    for score, target in zip(mobilenet_scores, y_test):
        if score < pass_thresh:
            if target == 0:
                pass_ok += 1
            else:
                pass_def += 1
        elif score >= hold_thresh:
            if target == 0:
                hold_ok += 1
            else:
                hold_def += 1
        else:
            if target == 0:
                rev_ok += 1
            else:
                rev_def += 1

    total_pass = pass_ok + pass_def
    total_hold = hold_ok + hold_def
    total_rev = rev_ok + rev_def
    hold_prec = (hold_def / total_hold) if total_hold > 0 else 0.0

    total_checks = total_rev + total_hold
    saved_checks = total_test - total_checks
    pct_saved = saved_checks / total_test

    print("\nHeld-Out Test QC Routing Breakdown:")
    print("+" + "-" * 68 + "+")
    print(f"| {'Action':<18} | {'Actual OK':>11} | {'Actual Defective':>16} | {'Total Routed':>14} |")
    print("+" + "-" * 68 + "+")
    print(f"| {'Pass Candidate':<18} | {pass_ok:>11,d} | {pass_def:>16,d} | {total_pass:>14,d} |")
    print(f"| {'Hold':<18} | {hold_ok:>11,d} | {hold_def:>16,d} | {total_hold:>14,d} |")
    print(f"| {'Manual Review':<18} | {rev_ok:>11,d} | {rev_def:>16,d} | {total_rev:>14,d} |")
    print("+" + "-" * 68 + "+")
    print(f"| {'Total Test':<18} | {ok_count:>11,d} | {def_count:>16,d} | {total_test:>14,d} |")
    print("+" + "-" * 68 + "+")

    print("\nHeld-Out Operational Workload & QC Safety Metrics:")
    print(f"  * Defective parts escaped to Pass: {pass_def} / {def_count} ({pass_def/def_count:.2%})")
    print(f"  * Hold Precision:                 {hold_prec:.2%} ({hold_def} / {total_hold} correct holds)")
    print(f"  * OK false alarms in Hold:        {hold_ok} / {ok_count} ({hold_ok/ok_count:.2%})")
    print(f"  * Manual Review queue:            {total_rev} parts ({rev_ok} OK, {rev_def} Defective)")
    print(f"  * Total Operator Checks:          {total_checks} parts ({total_rev} review + {total_hold} hold)")
    print(f"  * Inspection Workload Savings:    {saved_checks} checks avoided ({pct_saved:.1%} reduction vs. all {total_test})")

    # Export benchmark JSON
    benchmark_payload = {
        "stage": "Stage 5 - Held-Out Test Evaluation & Comparison Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "split": "test",
            "samples_evaluated": total_test,
            "actual_ok_samples": ok_count,
            "actual_defective_samples": def_count,
        },
        "model_comparison_at_0_50": {
            "baseline_logistic_regression": base_metrics,
            "mobilenetv2_transfer_learning": mobile_metrics,
        },
        "locked_qc_policy_test_performance": {
            "pass_threshold": pass_thresh,
            "hold_threshold": hold_thresh,
            "routing_actions": {
                "pass_candidate": {
                    "total": total_pass,
                    "actual_ok": pass_ok,
                    "actual_defective": pass_def,
                    "defect_escape_rate_pct": round((pass_def / def_count) * 100, 2),
                },
                "hold": {
                    "total": total_hold,
                    "actual_ok": hold_ok,
                    "actual_defective": hold_def,
                    "hold_precision_pct": round(hold_prec * 100, 2),
                },
                "manual_review": {
                    "total": total_rev,
                    "actual_ok": rev_ok,
                    "actual_defective": rev_def,
                },
            },
            "operator_workload": {
                "manual_reviews": total_rev,
                "hold_confirmations": total_hold,
                "total_operator_checks": total_checks,
                "baseline_all_checks": total_test,
                "checks_saved": saved_checks,
                "workload_reduction_pct": round(pct_saved * 100, 2),
            },
        },
    }

    output_benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    with output_benchmark_path.open("w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)

    print(f"\nFinal test benchmark saved to: {output_benchmark_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/data/split_manifest.csv"),
        help="Path to split_manifest.csv",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("outputs/baseline/baseline_test_scores.csv"),
        help="Path to baseline_test_scores.csv",
    )
    parser.add_argument(
        "--mobilenet-csv",
        type=Path,
        default=Path("outputs/mobilenetv2/mobilenetv2_test_scores.csv"),
        help="Path to mobilenetv2_test_scores.csv",
    )
    parser.add_argument(
        "--qc-policy",
        type=Path,
        default=Path("outputs/mobilenetv2/qc_policy.json"),
        help="Path to qc_policy.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/test_benchmark.json"),
        help="Output path for test_benchmark.json",
    )
    args = parser.parse_args()

    run_test_comparison(
        manifest_path=args.manifest,
        baseline_csv_path=args.baseline_csv,
        mobilenet_csv_path=args.mobilenet_csv,
        qc_policy_path=args.qc_policy,
        output_benchmark_path=args.output,
    )

