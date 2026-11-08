"""Run held-out test evaluation for the Stage 2 Logistic Regression baseline.

Loads outputs/baseline/model.joblib, reads 1,041 test images from casting_data.zip
according to outputs/data/split_manifest.csv, extracts 32x32 grayscale flattened
features, computes test predictions and defective probabilities, reports metrics at
cutoff 0.50, and exports outputs/baseline/baseline_test_scores.csv.
"""
import argparse
import csv
import io
import time
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

IMAGE_SIZE = (32, 32)
CLASS_NAMES = ["OK", "Defective"]


def load_test_manifest_rows(manifest_path: Path):
    test_rows = []
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["split"] == "test":
                test_rows.append(r)
    return test_rows


def extract_test_features(archive: ZipFile, rows: list[dict], image_size=IMAGE_SIZE):
    n_samples = len(rows)
    n_features = image_size[0] * image_size[1]
    X = np.empty((n_samples, n_features), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.int64)

    for i, row in enumerate(rows):
        raw_bytes = archive.read(row["image_path"])
        with Image.open(io.BytesIO(raw_bytes)) as img:
            gray = img.convert("L").resize(image_size)
            X[i] = np.asarray(gray, dtype=np.float32).ravel()
        y[i] = int(row["target"])

    return X, y


def evaluate_baseline_test(
    archive_path: Path = Path("casting_data.zip"),
    manifest_path: Path = Path("outputs/data/split_manifest.csv"),
    model_path: Path = Path("outputs/baseline/model.joblib"),
    output_scores_path: Path = Path("outputs/baseline/baseline_test_scores.csv"),
    cutoff: float = 0.50,
):
    print("=" * 70)
    print("Stage 5: Baseline Logistic Regression Held-Out Test Evaluation")
    print("=" * 70)

    assert archive_path.is_file(), f"Missing {archive_path}"
    assert manifest_path.is_file(), f"Missing {manifest_path}"
    assert model_path.is_file(), f"Missing {model_path}"

    # 1. Load test split from manifest
    print(f"\n[1/4] Loading test manifest rows from: {manifest_path}")
    test_rows = load_test_manifest_rows(manifest_path)
    total_test = len(test_rows)
    ok_count = sum(1 for r in test_rows if int(r["target"]) == 0)
    def_count = sum(1 for r in test_rows if int(r["target"]) == 1)

    print(f"      Total held-out test images: {total_test:,}")
    print(f"      - Actual OK (0):        {ok_count:,} ({ok_count/total_test:.1%})")
    print(f"      - Actual Defective (1): {def_count:,} ({def_count/total_test:.1%})")

    assert total_test == 1041, f"Expected 1,041 test samples, got {total_test}"
    assert ok_count == 439, f"Expected 439 OK test samples, got {ok_count}"
    assert def_count == 602, f"Expected 602 Defective test samples, got {def_count}"

    # 2. Extract 32x32 grayscale features
    print(f"\n[2/4] Extracting 32x32 grayscale features directly from {archive_path}...")
    t0_extract = time.time()
    with ZipFile(archive_path) as archive:
        X_test, y_test = extract_test_features(archive, test_rows)
    extract_time = time.time() - t0_extract
    print(f"      Extracted {X_test.shape[0]} test samples in {extract_time:.2f}s | Feature dim: {X_test.shape[1]}")

    # 3. Load baseline model and predict
    print(f"\n[3/4] Running inference with {model_path}...")
    pipeline = joblib.load(model_path)
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= cutoff).astype(int)

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, pos_label=1))
    rec = float(recall_score(y_test, preds, pos_label=1))
    f1 = float(f1_score(y_test, preds, pos_label=1))

    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    print("\nBaseline Held-Out Test Performance (Cutoff = 0.50):")
    print(f"  - Test Accuracy:             {acc:.4f} ({acc*100:.2f}%)")
    print(f"  - Defective Precision:       {prec:.4f} ({prec*100:.2f}%)")
    print(f"  - Defective Recall:          {rec:.4f} ({rec*100:.2f}%)")
    print(f"  - Defective F1-score:        {f1:.4f} ({f1*100:.2f}%)")
    print("\nConfusion Matrix (Class Order: [0] OK, [1] Defective):")
    print("+" + "-" * 50 + "+")
    print(f"| {'Actual \\ Predicted':<20} | {'Pred OK':>11} | {'Pred Defective':>14} | {'Total':>8} |")
    print("+" + "-" * 50 + "+")
    print(f"| {'Actual OK':<20} | {tn:>11,d} | {fp:>14,d} | {tn+fp:>8,d} |")
    print(f"| {'Actual Defective':<20} | {fn:>11,d} | {tp:>14,d} | {fn+tp:>8,d} |")
    print("+" + "-" * 50 + "+")
    print(f"| {'Total':<20} | {tn+fn:>11,d} | {fp+tp:>14,d} | {total_test:>8,d} |")
    print("+" + "-" * 50 + "+")
    print(f"Detailed counts: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # 4. Export test scores
    print(f"\n[4/4] Exporting baseline test scores to: {output_scores_path}...")
    output_scores_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "true_target",
        "true_label",
        "defective_score",
        "predicted_target_0_50",
        "predicted_label_0_50",
        "is_correct",
    ]
    with output_scores_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for i, row in enumerate(test_rows):
            score_val = round(float(probs[i]), 6)
            pred_val = int(preds[i])
            writer.writerow({
                "image_path": row["image_path"],
                "true_target": int(row["target"]),
                "true_label": row["label"],
                "defective_score": score_val,
                "predicted_target_0_50": pred_val,
                "predicted_label_0_50": CLASS_NAMES[pred_val],
                "is_correct": bool(pred_val == int(row["target"])),
            })

    print(f"      Saved {total_test:,} baseline test predictions to {output_scores_path}")
    print("\nLocal baseline test run completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("casting_data.zip"), help="Path to casting_data.zip")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/data/split_manifest.csv"), help="Path to split_manifest.csv")
    parser.add_argument("--model", type=Path, default=Path("outputs/baseline/model.joblib"), help="Path to baseline model.joblib")
    parser.add_argument("--output-scores", type=Path, default=Path("outputs/baseline/baseline_test_scores.csv"), help="Output CSV path")
    parser.add_argument("--cutoff", type=float, default=0.50, help="Probability cutoff for defective class")
    args = parser.parse_args()

    evaluate_baseline_test(
        archive_path=args.archive,
        manifest_path=args.manifest,
        model_path=args.model,
        output_scores_path=args.output_scores,
        cutoff=args.cutoff,
    )

