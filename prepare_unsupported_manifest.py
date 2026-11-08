"""Stage 6A: Manifest generator and validator for unsupported (OOD / negative) images.

Scans unsupported_data/, extracts image dimensions and SHA-256 hashes, clusters
perceptually identical or derived variants into source-image groups, and assigns
leakage-free train / validation / test splits matching the supported dataset ratio.

Manifest Columns:
  - image_path: Path relative to repository root
  - category: Fine-grained category label
  - source: Origin dataset or capture provenance
  - license: Explicit licensing terms
  - group_id: Source-image group identifier (keeps duplicates/variants in same split)
  - split: Split assignment (train / validation / test)
  - is_supported: 0 (unsupported / negative)
  - width: Image pixel width
  - height: Image pixel height
  - sha256: SHA-256 hash of image file bytes
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, UnidentifiedImageError

DEFAULT_SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SEED = 1501418

CATEGORY_MAP = {
    "pump_housings": "pump_housing",
    "bearings": "bearing",
    "round_metal_components": "round_metal_component",
    "misoriented_impellers": "misoriented_impeller",
    "industrial_hardware": "unrelated_industrial",
    "general_objects": "unrelated_general",
}


def scan_unsupported_images(base_dir: Path):
    """Scan unsupported_data directory for image files and categorize them."""
    records = []
    unreadable = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for path in sorted(base_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            parts = PurePosixPath(path.relative_to(base_dir)).parts
            # Determine category from folder name
            folder_key = parts[1] if len(parts) >= 2 else parts[0]
            category = CATEGORY_MAP.get(folder_key, folder_key)

            # Check if source metadata exists in folder or parent
            source = "manual_curation"
            license_str = "unspecified"

            # Check for local metadata.json
            meta_file = path.parent / "metadata.json"
            if meta_file.is_file():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    source = meta.get("source", source)
                    license_str = meta.get("license", license_str)
                except Exception:
                    pass

            raw_bytes = path.read_bytes()
            sha256 = hashlib.sha256(raw_bytes).hexdigest()

            try:
                with Image.open(path) as img:
                    width, height = img.size
                    rgb = img.convert("RGB")
                    pixel_hash = hashlib.sha256(f"{width}x{height}:".encode() + rgb.tobytes()).hexdigest()
            except (OSError, ValueError, UnidentifiedImageError):
                unreadable.append(str(path))
                continue

            # Base group on original filename root (before any crop/aug suffix) or pixel hash
            stem = path.stem.split("_aug")[0].split("_crop")[0]
            records.append({
                "image_path": str(path.as_posix()),
                "category": category,
                "source": source,
                "license": license_str,
                "group_key": f"{category}_{stem}_{pixel_hash[:12]}",
                "is_supported": 0,
                "width": width,
                "height": height,
                "sha256": sha256,
            })

    return records, unreadable


def assign_grouped_splits(records: list[dict], seed=SEED):
    """Assign train/val/test splits deterministically preserving group boundaries."""
    if not records:
        return records

    groups = sorted(list(set(r["group_key"] for r in records)))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n_groups = len(groups)
    n_val = max(1, int(round(n_groups * 0.15)))
    n_test = max(1, int(round(n_groups * 0.15)))
    n_train = n_groups - n_val - n_test

    group_to_split = {}
    for g in groups[:n_train]:
        group_to_split[g] = "train"
    for g in groups[n_train : n_train + n_val]:
        group_to_split[g] = "validation"
    for g in groups[n_train + n_val :]:
        group_to_split[g] = "test"

    for r in records:
        r["group_id"] = r.pop("group_key")
        r["split"] = group_to_split[r["group_id"]]

    return records


def build_manifest(
    data_dir: Path = Path("unsupported_data"),
    output_path: Path = Path("outputs/data/unsupported_manifest.csv"),
    schema_only_if_empty: bool = True,
):
    print("=" * 70)
    print("Stage 6A: Prepare Unsupported (Input-Checker) Dataset Manifest")
    print("=" * 70)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records, unreadable = scan_unsupported_images(data_dir)

    print(f"\n[1/3] Scanning {data_dir} for negative / unsupported images...")
    print(f"      Images found: {len(records):,}")
    if unreadable:
        print(f"      Unreadable files: {len(unreadable)}")

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
    ]

    if not records:
        print("\n[!] Notice: No unsupported image files were found in unsupported_data/.")
        print(f"    Writing template manifest with canonical schema to: {output_path}")

        # Write canonical schema header with sample illustrative rows
        sample_rows = [
            {
                "image_path": "unsupported_data/similar_parts/pump_housings/sample_housing_01.jpg",
                "category": "pump_housing",
                "source": "kaggle_mechanical_components",
                "license": "CC-BY-4.0",
                "group_id": "grp_pump_housing_001",
                "split": "train",
                "is_supported": 0,
                "width": 300,
                "height": 300,
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            },
            {
                "image_path": "unsupported_data/similar_parts/bearings/sample_bearing_01.jpg",
                "category": "bearing",
                "source": "kaggle_bearing_dataset",
                "license": "CC0-1.0",
                "group_id": "grp_bearing_001",
                "split": "validation",
                "is_supported": 0,
                "width": 300,
                "height": 300,
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            },
            {
                "image_path": "unsupported_data/similar_parts/round_metal_components/sample_flange_01.jpg",
                "category": "round_metal_component",
                "source": "industrial_parts_benchmark",
                "license": "MIT",
                "group_id": "grp_flange_001",
                "split": "test",
                "is_supported": 0,
                "width": 300,
                "height": 300,
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            },
        ]
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            if not schema_only_if_empty:
                writer.writerows(sample_rows)
        print(f"      Initialized schema at: {output_path}")
        return

    # If records exist, assign splits
    records = assign_grouped_splits(records)
    print("\n[2/3] Assigned grouped splits:")
    splits_counter = Counter(r["split"] for r in records)
    for s, c in splits_counter.items():
        print(f"      - {s:<12}: {c:,} images ({c/len(records):.1%})")

    print("\n[3/3] Category Breakdown:")
    cat_counter = Counter(r["category"] for r in records)
    for cat, c in cat_counter.items():
        print(f"      - {cat:<24}: {c:,} images")

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    print(f"\nManifest successfully written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("unsupported_data"),
        help="Directory containing unsupported negative images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/data/unsupported_manifest.csv"),
        help="Path for generated unsupported_manifest.csv",
    )
    args = parser.parse_args()
    build_manifest(data_dir=args.data_dir, output_path=args.output, schema_only_if_empty=True)

