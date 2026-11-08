"""Stage 2: Logistic Regression baseline for impeller defect classification.

Extracts 32x32 grayscale flattened features from casting images in casting_data.zip,
trains a scikit-learn Pipeline (StandardScaler + LogisticRegression) on the train split,
and evaluates performance on the validation split with a fixed 0.50 threshold.
The test split is strictly held out.
"""
import argparse
import csv
import io
import json
import time
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IMAGE_SIZE = (32, 32)
CLASS_NAMES = ["OK", "Defective"]
DEFAULT_SEED = 1501418


def load_split_rows(manifest_path: Path):
    """Load train, validation, and test rows from split_manifest.csv."""
    train_rows = []
    val_rows = []
    test_count = 0
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if split == "train":
                train_rows.append(row)
            elif split == "validation":
                val_rows.append(row)
            elif split == "test":
                test_count += 1
            else:
                raise ValueError(f"Unknown split name: {split}")
    return train_rows, val_rows, test_count


def extract_features_and_labels(archive: ZipFile, rows: list[dict], image_size=IMAGE_SIZE):
    """Read images from the ZIP archive, convert to grayscale 32x32, and flatten."""
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


def train_and_evaluate(
    archive_path: Path = Path("casting_data.zip"),
    manifest_path: Path = Path("outputs/data/split_manifest.csv"),
    output_dir: Path = Path("outputs/baseline"),
    cutoff: float = 0.50,
    max_iter: int = 1000,
    seed: int = DEFAULT_SEED,
):
    print("=" * 65)
    print("Stage 2: Impeller QC Logistic Regression Baseline")
    print("=" * 65)

    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # 1. Load split assignments from manifest
    print(f"\n[1/5] Loading split manifest: {manifest_path}")
    train_rows, val_rows, test_count = load_split_rows(manifest_path)
    print(f"      - Training images:   {len(train_rows):,}")
    print(f"      - Validation images: {len(val_rows):,}")
    print(f"      - Held-out test:     {test_count:,}")

    if len(train_rows) != 5199 or len(val_rows) != 1044:
        print(f"      WARNING: Expected 5,199 train and 1,044 val images, got {len(train_rows)} and {len(val_rows)}")

    # 2. Extract 32x32 grayscale features directly from ZIP
    print(f"\n[2/5] Reading images and extracting 32x32 grayscale features from {archive_path}...")
    t0_extract = time.time()
    with ZipFile(archive_path) as archive:
        X_train, y_train = extract_features_and_labels(archive, train_rows)
        X_val, y_val = extract_features_and_labels(archive, val_rows)
    extract_time = time.time() - t0_extract
    print(f"      Extracted {X_train.shape[0]} train and {X_val.shape[0]} val samples in {extract_time:.2f}s")
    print(f"      Feature matrix: {X_train.shape[1]} flattened pixels per image")

    train_ok = int(np.sum(y_train == 0))
    train_def = int(np.sum(y_train == 1))
    val_ok = int(np.sum(y_val == 0))
    val_def = int(np.sum(y_val == 1))
    print(f"      Train class balance: OK={train_ok:,} ({train_ok/len(y_train):.1%}), Defective={train_def:,} ({train_def/len(y_train):.1%})")
    print(f"      Val class balance:   OK={val_ok:,} ({val_ok/len(y_val):.1%}), Defective={val_def:,} ({val_def/len(y_val):.1%})")

    # 3. Fit StandardScaler + LogisticRegression pipeline
    print("\n[3/5] Fitting Pipeline(StandardScaler, LogisticRegression)...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=max_iter, random_state=seed, solver="lbfgs")),
    ])
    t0_fit = time.time()
    pipeline.fit(X_train, y_train)
    fit_time = time.time() - t0_fit
    clf: LogisticRegression = pipeline.named_steps["classifier"]
    print(f"      Pipeline fitted in {fit_time:.2f}s (iterations: {clf.n_iter_[0]}/{max_iter})")

    # 4. Evaluate on validation set
    print(f"\n[4/5] Evaluating on validation set (threshold = {cutoff:.2f})...")
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= cutoff).astype(int)

    acc = float(accuracy_score(y_val, val_preds))
    prec = float(precision_score(y_val, val_preds, pos_label=1))
    rec = float(recall_score(y_val, val_preds, pos_label=1))
    f1 = float(f1_score(y_val, val_preds, pos_label=1))

    # Confusion matrix: rows = true (0=OK, 1=Defective), cols = pred (0=OK, 1=Defective)
    cm = confusion_matrix(y_val, val_preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    print("\nValidation Performance:")
    print(f"  - Accuracy:                  {acc:.4f} ({acc*100:.2f}%)")
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
    print(f"| {'Total':<20} | {tn+fn:>11,d} | {fp+tp:>14,d} | {len(y_val):>8,d} |")
    print("+" + "-" * 50 + "+")
    print(f"Detailed counts: TN (True OK)={tn}, FP (False Defective)={fp}, FN (False OK)={fn}, TP (True Defective)={tp}")

    # 5. Save artifacts
    print(f"\n[5/5] Saving model and metrics to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    metrics_path = output_dir / "validation_metrics.json"

    joblib.dump(pipeline, model_path)
    print(f"      Saved fitted pipeline: {model_path}")

    metrics_payload = {
        "stage": "Stage 2 - Baseline Logistic Regression",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "archive": str(archive_path),
            "manifest": str(manifest_path),
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "held_out_test_samples": test_count,
            "train_counts": {"OK": train_ok, "Defective": train_def},
            "val_counts": {"OK": val_ok, "Defective": val_def},
        },
        "preprocessing": {
            "color_mode": "grayscale",
            "image_size": list(IMAGE_SIZE),
            "feature_dim": X_train.shape[1],
            "scaling": "StandardScaler",
        },
        "model": {
            "type": "Pipeline(StandardScaler, LogisticRegression)",
            "solver": clf.solver,
            "max_iter": max_iter,
            "n_iter": int(clf.n_iter_[0]),
            "random_state": seed,
            "fit_time_seconds": round(fit_time, 4),
        },
        "validation_evaluation": {
            "probability_threshold": cutoff,
            "positive_class": "Defective (1)",
            "negative_class": "OK (0)",
            "accuracy": round(acc, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1_score": round(f1, 6),
            "confusion_matrix": {
                "class_order": CLASS_NAMES,
                "matrix": [[tn, fp], [fn, tp]],
                "true_negative_ok": tn,
                "false_positive_defective": fp,
                "false_negative_ok": fn,
                "true_positive_defective": tp,
            },
        },
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"      Saved validation metrics: {metrics_path}")

    print("\nStage 2 baseline training and validation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("casting_data.zip"), help="Path to casting_data.zip")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/data/split_manifest.csv"), help="Path to split_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"), help="Output directory for model and metrics")
    parser.add_argument("--cutoff", type=float, default=0.50, help="Probability threshold for Defective class (default: 0.50)")
    parser.add_argument("--max-iter", type=int, default=1000, help="Max iterations for Logistic Regression solver (default: 1000)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 1501418)")
    args = parser.parse_args()

    train_and_evaluate(
        archive_path=args.archive,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        cutoff=args.cutoff,
        max_iter=args.max_iter,
        seed=args.seed,
    )

