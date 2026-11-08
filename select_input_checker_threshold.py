"""Stage 6B: Select input checker threshold on validation data and export policy.

Rules:
1. Constraint: Accept at least 95.0% of supported validation images (>= 992 / 1,044).
2. Objective: Maximize rejected unsupported validation images (out of 13).
3. Tie-breaker: Prefer higher supported acceptance rate.
4. Save policy to outputs/input_checker/input_checker_policy.json and metrics to validation_metrics.json.
"""
import csv
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np

def run_threshold_selection():
    output_dir = Path("outputs/input_checker")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = output_dir / "input_checker_outputs.zip"
    scores_path = output_dir / "input_checker_val_scores.csv"
    
    # Auto-extract from zip if present
    if not scores_path.is_file() and zip_path.is_file():
        print(f"Extracting outputs from {zip_path}...")
        with ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
            
    if not scores_path.is_file():
        print(f"[WAITING] Validation scores not found at {scores_path}.")
        print("Please train the input checker using train_input_checker.ipynb in Colab,")
        print(f"and place either input_checker_val_scores.csv or input_checker_outputs.zip into {output_dir}/.")
        return None

    # Load validation scores
    val_rows = []
    with scores_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            val_rows.append({
                "image_path": r["image_path"],
                "true_target": int(r["true_target"]),
                "is_supported": int(r.get("is_supported", r["true_target"])),
                "category": r.get("category", "supported_impeller" if int(r["true_target"]) == 1 else "unsupported"),
                "supported_probability": float(r["supported_probability"])
            })

    print(f"Loaded {len(val_rows):,} validation prediction scores from {scores_path}")
    
    supp_rows = [r for r in val_rows if r["is_supported"] == 1]
    unsupp_rows = [r for r in val_rows if r["is_supported"] == 0]
    
    n_supp = len(supp_rows)
    n_unsupp = len(unsupp_rows)
    
    print(f"  - Supported validation samples:   {n_supp:,} (Expected: 1,044)")
    print(f"  - Unsupported validation samples: {n_unsupp:,} (Expected: 13)")
    
    assert n_supp == 1044, f"Expected 1,044 supported validation rows, found {n_supp}"
    assert n_unsupp == 13, f"Expected 13 unsupported validation rows, found {n_unsupp}"

    supp_scores = np.array([r["supported_probability"] for r in supp_rows])
    unsupp_scores = np.array([r["supported_probability"] for r in unsupp_rows])

    # Build dense candidate threshold set
    candidates = np.unique(np.concatenate([
        np.linspace(0.0001, 0.9999, 5000),
        np.clip(supp_scores, 1e-6, 1.0 - 1e-6),
        np.clip(unsupp_scores, 1e-6, 1.0 - 1e-6)
    ]))
    candidates.sort()

    best_tau = None
    best_unsupp_rej = -1
    best_supp_acc = -1

    for tau in candidates:
        supp_acc = int(np.sum(supp_scores >= tau))
        supp_acc_rate = supp_acc / n_supp
        unsupp_rej = int(np.sum(unsupp_scores < tau))
        
        # Constraint: >= 95% supported acceptance
        if supp_acc_rate >= 0.95:
            if (unsupp_rej > best_unsupp_rej) or (
                unsupp_rej == best_unsupp_rej and supp_acc > best_supp_acc
            ):
                best_unsupp_rej = unsupp_rej
                best_supp_acc = supp_acc
                best_tau = float(tau)

    print("\n" + "=" * 70)
    print(f"OPTIMAL INPUT CHECKER THRESHOLD: tau = {best_tau:.4f}")
    print("=" * 70)
    print(f"Supported Acceptance:  {best_supp_acc:,} / {n_supp:,} ({best_supp_acc / n_supp:.2%}) [>= 95.0% Constraint MET]")
    print(f"Unsupported Rejection: {best_unsupp_rej:,} / {n_unsupp:,} ({best_unsupp_rej / n_unsupp:.2%}) [Maximized]")

    # Confusion matrix
    # Actual: 0=Unsupported, 1=Supported
    # Pred:   0=Unsupported (< tau), 1=Supported (>= tau)
    tn = best_unsupp_rej
    fp = n_unsupp - best_unsupp_rej
    tp = best_supp_acc
    fn = n_supp - best_supp_acc

    print("\n" + "-" * 70)
    print("VALIDATION CONFUSION MATRIX (Labels: [0: Unsupported, 1: Supported])")
    print("-" * 70)
    print(f"{'':22} | {'Pred Unsupported (0)':<22} | {'Pred Supported (1)':<20} | {'Total'}")
    print("-" * 75)
    print(f"{'Actual Unsupported (0)':<22} | {tn:<22d} | {fp:<20d} | {tn+fp}")
    print(f"{'Actual Supported (1)':<22}   | {fn:<22d} | {tp:<20d} | {fn+tp}")
    print("-" * 75)
    print(f"{'Total':<22} | {tn+fn:<22d} | {fp+tp:<20d} | {len(val_rows)}")
    print("-" * 70)
    print(f"Supported wrongly blocked (FN): {fn:,} / {n_supp:,} ({fn / n_supp:.2%})")
    print(f"Unsupported wrongly accepted (FP): {fp:,} / {n_unsupp:,} ({fp / n_unsupp:.2%})")

    # Category breakdown for unsupported
    cat_stats = {}
    for r in unsupp_rows:
        cat = r["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "rejected": 0, "wrongly_accepted": 0, "scores": []}
        cat_stats[cat]["total"] += 1
        cat_stats[cat]["scores"].append(r["supported_probability"])
        if r["supported_probability"] < best_tau:
            cat_stats[cat]["rejected"] += 1
        else:
            cat_stats[cat]["wrongly_accepted"] += 1

    print("\n" + "-" * 85)
    print("UNSUPPORTED VALIDATION REJECTION BREAKDOWN BY CATEGORY")
    print("-" * 85)
    print(f"{'Category':<26} | {'Total':<6} | {'Rejected':<10} | {'Wrongly Acc':<12} | {'Rejection Rate':<14} | {'Mean Score'}")
    print("-" * 85)
    for cat, stats in sorted(cat_stats.items()):
        rej_pct = stats["rejected"] / stats["total"] * 100.0
        m_score = float(np.mean(stats["scores"]))
        print(f"{cat:<26} | {stats['total']:<6} | {stats['rejected']:<10} | {stats['wrongly_accepted']:<12} | {rej_pct:>12.1f}%  | {m_score:.4f}")
    print("=" * 85)

    # Save policy json
    policy_path = output_dir / "input_checker_policy.json"
    policy_data = {
        "model_name": "input_checker_mobilenetv2.keras",
        "target_definition": {
            "1": "Supported (valid top-view cast impeller)",
            "0": "Unsupported (out-of-scope part, wrong-view impeller, unrelated object)"
        },
        "routing_rule": {
            "accepted": f"supported_probability >= {best_tau:.4f}",
            "blocked": f"supported_probability < {best_tau:.4f}"
        },
        "threshold": float(best_tau),
        "optimization_constraint": "Supported Acceptance Rate >= 95.0% on validation set",
        "optimization_objective": "Maximize Unsupported Rejection on validation set",
        "validation_summary": {
            "supported_total": n_supp,
            "supported_accepted": tp,
            "supported_wrongly_blocked": fn,
            "supported_acceptance_rate": float(tp / n_supp),
            "unsupported_total": n_unsupp,
            "unsupported_rejected": tn,
            "unsupported_wrongly_accepted": fp,
            "unsupported_rejection_rate": float(tn / n_unsupp),
            "confusion_matrix_labels": ["Unsupported (0)", "Supported (1)"],
            "confusion_matrix": [
                [tn, fp],
                [fn, tp]
            ]
        },
        "unsupported_category_breakdown": {
            cat: {
                "total": stats["total"],
                "rejected": stats["rejected"],
                "wrongly_accepted": stats["wrongly_accepted"],
                "rejection_rate": float(stats["rejected"] / stats["total"]),
                "mean_score": float(np.mean(stats["scores"]))
            }
            for cat, stats in sorted(cat_stats.items())
        }
    }

    with policy_path.open("w", encoding="utf-8") as f:
        json.dump(policy_data, f, indent=2)
    print(f"\nSaved input checker policy to {policy_path}")

    # Save metrics json
    metrics_path = output_dir / "validation_metrics.json"
    metrics_data = {
        "threshold": float(best_tau),
        "accuracy": float((tp + tn) / len(val_rows)),
        "supported_precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "supported_recall": float(tp / n_supp),
        "unsupported_precision": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
        "unsupported_recall": float(tn / n_unsupp),
        "confusion_matrix": {
            "true_unsupported_pred_unsupported_TN": tn,
            "true_unsupported_pred_supported_FP": fp,
            "true_supported_pred_unsupported_FN": fn,
            "true_supported_pred_supported_TP": tp
        }
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved validation metrics to {metrics_path}")
    return policy_data

if __name__ == "__main__":
    run_threshold_selection()

