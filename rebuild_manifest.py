"""Rebuild unsupported_manifest.csv from downloaded images and metadata with category stratification."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 1501418


def refine_group_id(title: str, path: str, category: str) -> str:
    path_lower = path.lower()
    title_lower = title.lower()

    if category == "misoriented_impeller":
        if "24_blade_compressor" in path_lower or "17451139" in title_lower or "17451140" in title_lower:
            return "grp_impeller_nara_24_blade"
        elif "chrysler" in path_lower or "chrysler" in title_lower:
            return "grp_impeller_chrysler_gas_turbine"
        elif "damaged_compressor_impeller" in path_lower:
            return "grp_impeller_nara_damaged"
        elif "damaged_compressor_parts_of_r_31" in path_lower or "r_31" in title_lower:
            return "grp_impeller_r31_damaged_parts"
        elif "roots" in path_lower or "case_impeller" in path_lower:
            return "grp_impeller_case_roots"
        elif "174432" in path_lower or "17443204" in title_lower:
            return "grp_impeller_nara_single"
        else:
            return f"grp_impeller_{Path(path).stem[:25]}"

    # Multi-view pump casing sets
    if "pumphus" in path_lower or "pumphus" in title_lower:
        return "grp_water_pump_casing_set"
    if "sandfields" in path_lower:
        return "grp_pump_housing_sandfields"

    # Multi-view dredging flanges
    if "baggerwerkzaamheden" in path_lower:
        return "grp_flange_langwarder_dredging"
    if "blue_grass" in path_lower:
        return "grp_flange_blue_grass"
    if "haltern_am_see" in path_lower:
        return "grp_flange_haltern_quarzwerke"

    # Bearings multi-view
    if "ladinas_andiast" in path_lower:
        return "grp_bearing_ladinas_andiast"
    if "zirconium_dioxide" in path_lower:
        return "grp_bearing_zro2_ceramic"

    # General objects multi-view
    if "banana_white_background_sm" in path_lower:
        return "grp_banana_white_background_sm_series"

    stem = Path(path).stem[:30]
    return f"grp_{category}_{stem}"


def rebuild_and_split():
    meta_path = Path("outputs/data/unsupported_metadata.json")
    with meta_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} existing unsupported records.")

    # Refine groups
    for r in records:
        title = r.get("source_url", "").split("/")[-1]
        r["group_id"] = refine_group_id(title, r["image_path"], r["category"])

    # Category stratified group splitting
    rng = np.random.default_rng(SEED)
    by_category = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r)

    group_split_map = {}

    for cat in sorted(by_category.keys()):
        cat_records = by_category[cat]
        cat_groups = sorted(list(set(r["group_id"] for r in cat_records)))
        rng.shuffle(cat_groups)
        n = len(cat_groups)

        if n == 1:
            group_split_map[cat_groups[0]] = "train"
        elif n == 2:
            group_split_map[cat_groups[0]] = "train"
            group_split_map[cat_groups[1]] = "validation"
        elif n == 3:
            group_split_map[cat_groups[0]] = "train"
            group_split_map[cat_groups[1]] = "validation"
            group_split_map[cat_groups[2]] = "test"
        else:
            n_val = max(1, int(round(n * 0.15)))
            n_test = max(1, int(round(n * 0.15)))
            n_train = n - n_val - n_test
            for g in cat_groups[:n_train]:
                group_split_map[g] = "train"
            for g in cat_groups[n_train : n_train + n_val]:
                group_split_map[g] = "validation"
            for g in cat_groups[n_train + n_val :]:
                group_split_map[g] = "test"

    for r in records:
        r["split"] = group_split_map[r["group_id"]]

    # Write Manifest
    manifest_path = Path("outputs/data/unsupported_manifest.csv")
    fieldnames = [
        "image_path",
        "category",
        "source",
        "license",
        "group_id",
        "split",
        "is_supported",
        "width",
        "height",
        "sha256",
        "source_url",
        "creator",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    total_images = len(records)
    n_groups = len(set(r["group_id"] for r in records))
    split_counts = Counter(r["split"] for r in records)

    print("\n" + "=" * 75)
    print("STAGE 6A: UNSUPPORTED DATASET SUMMARY (Category-Stratified)")
    print("=" * 75)
    print(f"Total Images:        {total_images} (Target: 60–100)")
    print(f"Total Unique Groups: {n_groups}")

    print("\nOverall Split Distribution:")
    for s in ["train", "validation", "test"]:
        cnt = split_counts[s]
        print(f"  - {s:<12}: {cnt:2d} images ({cnt/total_images:.1%})")

    # Category by split matrix
    cat_split_counts = defaultdict(Counter)
    for r in records:
        cat_split_counts[r["category"]][r["split"]] += 1

    print("\nCategory Counts by Split:")
    print("+" + "-" * 72 + "+")
    print(f"| {'Category':<26} | {'Train':>8} | {'Val':>8} | {'Test':>8} | {'Total':>8} |")
    print("+" + "-" * 72 + "+")
    for cat in sorted(cat_split_counts.keys()):
        counts = cat_split_counts[cat]
        c_train = counts["train"]
        c_val = counts["validation"]
        c_test = counts["test"]
        c_tot = c_train + c_val + c_test
        print(f"| {cat:<26} | {c_train:>8d} | {c_val:>8d} | {c_test:>8d} | {c_tot:>8d} |")
    print("+" + "-" * 72 + "+")
    print(f"| {'Total':<26} | {split_counts['train']:>8d} | {split_counts['validation']:>8d} | {split_counts['test']:>8d} | {total_images:>8d} |")
    print("+" + "-" * 72 + "+")

    # Check for any leakage across splits
    group_splits = defaultdict(set)
    for r in records:
        group_splits[r["group_id"]].add(r["split"])

    leaked = [g for g, s in group_splits.items() if len(s) > 1]
    if leaked:
        raise RuntimeError(f"Data leakage detected! Groups crossing splits: {leaked}")
    print(f"\nLeakage Verification: PASSED (0 of {n_groups} groups cross splits).")


if __name__ == "__main__":
    rebuild_and_split()

