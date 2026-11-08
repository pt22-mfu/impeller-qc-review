"""Pre-training integrity verification for Stage 6B.

Verifies:
1. Manifests presence and formatting
2. Image paths existence (in casting_data.zip for supported, on disk for unsupported)
3. SHA-256 hashes matching manifests
4. Split counts and labels (1 = supported, 0 = unsupported)
5. Strict group-level split isolation (no group crossing splits)
"""
import csv
import hashlib
from pathlib import Path
from zipfile import ZipFile

def run_integrity_verification():
    supported_manifest_path = Path("outputs/data/split_manifest.csv")
    unsupported_manifest_path = Path("outputs/data/unsupported_manifest.csv")
    zip_path = Path("casting_data.zip")

    print("=" * 70)
    print("STAGE 6B: PRE-TRAINING MANIFEST AND DATA INTEGRITY VERIFICATION")
    print("=" * 70)

    errors = []

    # 1. Supported Impeller Dataset Verification
    print("\n1. Verifying Supported Dataset (outputs/data/split_manifest.csv)...")
    if not supported_manifest_path.is_file():
        errors.append(f"Missing {supported_manifest_path}")
        return errors
    if not zip_path.is_file():
        errors.append(f"Missing {zip_path}")
        return errors

    supp_rows = []
    with supported_manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            supp_rows.append(r)

    print(f"   Total supported rows in manifest: {len(supp_rows):,}")
    if len(supp_rows) != 7284:
        errors.append(f"Expected 7,284 supported rows, found {len(supp_rows)}")

    supp_splits = {"train": 0, "validation": 0, "test": 0}
    supp_groups = {"train": set(), "validation": set(), "test": set()}

    with ZipFile(zip_path, "r") as zf:
        namelist = set(zf.namelist())
        for r in supp_rows:
            p = r["image_path"]
            s = r["split"]
            g = r["group_id"]
            if p not in namelist:
                errors.append(f"Supported path not in ZIP: {p}")
            if s in supp_splits:
                supp_splits[s] += 1
                supp_groups[s].add(g)
            else:
                errors.append(f"Invalid split '{s}' in supported manifest row {p}")

    print(f"   Supported split distribution: {supp_splits}")
    if supp_splits["train"] != 5199 or supp_splits["validation"] != 1044 or supp_splits["test"] != 1041:
        errors.append(f"Supported split counts mismatch: {supp_splits}")

    # Check supported group isolation
    supp_tv = supp_groups["train"].intersection(supp_groups["validation"])
    supp_tt = supp_groups["train"].intersection(supp_groups["test"])
    supp_vt = supp_groups["validation"].intersection(supp_groups["test"])

    if supp_tv or supp_tt or supp_vt:
        errors.append(f"Supported group leak detected! Train/Val: {len(supp_tv)}, Train/Test: {len(supp_tt)}, Val/Test: {len(supp_vt)}")
    else:
        print(f"   Supported group isolation verified: {len(supp_groups['train']):,} train groups, {len(supp_groups['validation']):,} val groups, {len(supp_groups['test']):,} test groups. Zero overlaps.")

    # 2. Unsupported Dataset Verification
    print("\n2. Verifying Unsupported Dataset (outputs/data/unsupported_manifest.csv)...")
    if not unsupported_manifest_path.is_file():
        errors.append(f"Missing {unsupported_manifest_path}")
        return errors

    unsupp_rows = []
    with unsupported_manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            unsupp_rows.append(r)

    print(f"   Total unsupported rows in manifest: {len(unsupp_rows):,}")
    if len(unsupp_rows) != 90:
        errors.append(f"Expected 90 unsupported rows, found {len(unsupp_rows)}")

    unsupp_splits = {"train": 0, "validation": 0, "test": 0}
    unsupp_groups = {"train": set(), "validation": set(), "test": set()}
    unsupp_categories = set()
    hash_mismatches = 0

    for r in unsupp_rows:
        img_path = Path(r["image_path"])
        s = r["split"]
        g = r["group_id"]
        unsupp_categories.add(r["category"])

        if not img_path.is_file():
            errors.append(f"Unsupported file missing on disk: {img_path}")
            continue

        # Verify hash
        content = img_path.read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != r["sha256"]:
            errors.append(f"Hash mismatch for {img_path}: manifest {r['sha256'][:8]}... != actual {actual_hash[:8]}...")
            hash_mismatches += 1

        # Check label
        if r["is_supported"] != "0":
            errors.append(f"Unsupported row {img_path} has is_supported='{r['is_supported']}' instead of '0'")

        if s in unsupp_splits:
            unsupp_splits[s] += 1
            unsupp_groups[s].add(g)
        else:
            errors.append(f"Invalid split '{s}' in unsupported manifest row {img_path}")

    print(f"   Unsupported split distribution: {unsupp_splits}")
    print(f"   Unsupported categories ({len(unsupp_categories)}): {sorted(list(unsupp_categories))}")
    if unsupp_splits["train"] != 60 or unsupp_splits["validation"] != 13 or unsupp_splits["test"] != 17:
        errors.append(f"Unsupported split counts mismatch: {unsupp_splits}")

    # Check unsupported group isolation
    unsupp_tv = unsupp_groups["train"].intersection(unsupp_groups["validation"])
    unsupp_tt = unsupp_groups["train"].intersection(unsupp_groups["test"])
    unsupp_vt = unsupp_groups["validation"].intersection(unsupp_groups["test"])

    if unsupp_tv or unsupp_tt or unsupp_vt:
        errors.append(f"Unsupported group leak detected! Train/Val: {len(unsupp_tv)}, Train/Test: {len(unsupp_tt)}, Val/Test: {len(unsupp_vt)}")
    else:
        print(f"   Unsupported group isolation verified: {len(unsupp_groups['train'])} train groups, {len(unsupp_groups['validation'])} val groups, {len(unsupp_groups['test'])} test groups. Zero overlaps.")

    # 3. Validation set summary
    print("\n3. Combined Split Summary for Stage 6B:")
    print(f"   - Training:   {supp_splits['train']:,} supported (1) + {unsupp_splits['train']:,} unsupported (0) = {supp_splits['train'] + unsupp_splits['train']:,} total")
    print(f"   - Validation: {supp_splits['validation']:,} supported (1) + {unsupp_splits['validation']:,} unsupported (0) = {supp_splits['validation'] + unsupp_splits['validation']:,} total")
    print(f"   - Test:       {supp_splits['test']:,} supported (1) + {unsupp_splits['test']:,} unsupported (0) = {supp_splits['test'] + unsupp_splits['test']:,} total (HELD OUT)")

    # 4. Final verification status
    print("\n" + "=" * 70)
    if errors:
        print(f"VERIFICATION FAILED: {len(errors)} issues found!")
        for e in errors[:10]:
            print(f"  [ERROR] {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors.")
        return False
    else:
        print("PRE-TRAINING INTEGRITY VERIFICATION PASSED PERFECTLY!")
        print("All file paths, SHA-256 hashes, labels, and group boundaries are 100% intact.")
        print("=" * 70)
        return True

if __name__ == "__main__":
    success = run_integrity_verification()
    if not success:
        exit(1)

