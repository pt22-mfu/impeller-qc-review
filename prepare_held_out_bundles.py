"""Package held-out test images into byte-exact ZIP bundles with ground-truth CSVs and attributions.

Outputs (excluded from full archive ignores, tracked for independent testing):
- data/held_out_test/casting_held_out_test.zip (1,041 held-out casting images + CSV + ATTRIBUTION.md)
- data/held_out_test/unsupported_held_out_test.zip (17 held-out unsupported images + CSV + ATTRIBUTION.md)
"""
import csv
import hashlib
import io
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

BASE_DIR = Path(".")
SPLIT_MANIFEST = BASE_DIR / "outputs" / "data" / "split_manifest.csv"
UNSUPP_MANIFEST = BASE_DIR / "outputs" / "data" / "unsupported_manifest.csv"
CASTING_ZIP = BASE_DIR / "casting_data.zip"
UNSUPP_ZIP = BASE_DIR / "unsupported_data.zip"
OUT_DIR = BASE_DIR / "data" / "held_out_test"


def build_casting_held_out_zip():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_zip = OUT_DIR / "casting_held_out_test.zip"

    with SPLIT_MANIFEST.open("r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]

    assert len(rows) == 1041, f"Expected 1,041 casting test rows, found {len(rows)}"
    assert CASTING_ZIP.is_file(), f"Missing source archive: {CASTING_ZIP}"

    csv_buf = io.StringIO()
    fieldnames = [
        "zip_entry_path",
        "original_image_path",
        "split",
        "label",
        "target",
        "ground_truth_class",
        "width",
        "height",
        "sha256",
        "pixel_sha256",
        "phash",
        "group_id",
        "source_dataset",
        "creator",
        "license",
    ]
    writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
    writer.writeheader()

    attribution_md = """# Held-Out Casting Test Set Attribution

- **Dataset**: [Real-Life Industrial Dataset of Casting Product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
- **Creator / Dataset Provider**: Raviraj Joshi
- **License**: [CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
- **Scope Included**: All 1,041 strictly held-out `test` split images from `outputs/data/split_manifest.csv` (602 `def_front`, 439 `ok_front`), preserving exact original image bytes verified via SHA-256.
"""

    with ZipFile(CASTING_ZIP, "r") as src_zf, ZipFile(out_zip, "w", ZIP_DEFLATED) as dst_zf:
        for r in rows:
            orig_path = r["image_path"]
            raw_bytes = src_zf.read(orig_path)
            actual_sha = hashlib.sha256(raw_bytes).hexdigest()
            assert actual_sha == r["sha256"], f"SHA-256 mismatch for {orig_path}"

            # Keep clean relative structure under images/
            rel_subpath = orig_path.replace("casting_data/test/", "")
            zip_entry = f"images/{rel_subpath}"
            dst_zf.writestr(zip_entry, raw_bytes)

            gt_class = "Defective" if r["target"] == "1" else "OK"
            writer.writerow({
                "zip_entry_path": zip_entry,
                "original_image_path": orig_path,
                "split": r["split"],
                "label": r["label"],
                "target": r["target"],
                "ground_truth_class": gt_class,
                "width": r["width"],
                "height": r["height"],
                "sha256": r["sha256"],
                "pixel_sha256": r["pixel_sha256"],
                "phash": r["phash"],
                "group_id": r["group_id"],
                "source_dataset": "Kaggle Real-Life Industrial Dataset of Casting Product",
                "creator": "Raviraj Joshi",
                "license": "CC BY-NC-ND 4.0",
            })

        dst_zf.writestr("casting_test_ground_truth.csv", csv_buf.getvalue().encode("utf-8"))
        dst_zf.writestr("ATTRIBUTION.md", attribution_md.encode("utf-8"))

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Built {out_zip} ({len(rows)} images, {size_mb:.2f} MB)")


def build_unsupported_held_out_zip():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_zip = OUT_DIR / "unsupported_held_out_test.zip"

    with UNSUPP_MANIFEST.open("r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]

    assert len(rows) == 17, f"Expected 17 unsupported test rows, found {len(rows)}"

    csv_buf = io.StringIO()
    fieldnames = [
        "zip_entry_path",
        "original_image_path",
        "split",
        "category",
        "is_supported",
        "width",
        "height",
        "sha256",
        "group_id",
        "source",
        "creator",
        "license",
        "source_url",
    ]
    writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
    writer.writeheader()

    attr_lines = [
        "# Held-Out Unsupported Test Set Attribution",
        "",
        "All 17 strictly held-out `test` split images from `outputs/data/unsupported_manifest.csv`, preserving exact original bytes verified via SHA-256.",
        "",
        "| Filename | Category | Creator | License | Source URL | SHA-256 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    with ZipFile(out_zip, "w", ZIP_DEFLATED) as dst_zf:
        src_zf = ZipFile(UNSUPP_ZIP, "r") if UNSUPP_ZIP.is_file() else None
        try:
            for r in rows:
                orig_path = r["image_path"]
                disk_path = BASE_DIR / orig_path
                if disk_path.is_file():
                    raw_bytes = disk_path.read_bytes()
                elif src_zf is not None:
                    raw_bytes = src_zf.read(orig_path)
                else:
                    raise FileNotFoundError(f"Could not find {orig_path} on disk or in {UNSUPP_ZIP}")

                actual_sha = hashlib.sha256(raw_bytes).hexdigest()
                assert actual_sha == r["sha256"], f"SHA-256 mismatch for {orig_path}"

                fname = Path(orig_path).name
                zip_entry = f"images/{fname}"
                dst_zf.writestr(zip_entry, raw_bytes)

                writer.writerow({
                    "zip_entry_path": zip_entry,
                    "original_image_path": orig_path,
                    "split": r["split"],
                    "category": r["category"],
                    "is_supported": r["is_supported"],
                    "width": r["width"],
                    "height": r["height"],
                    "sha256": r["sha256"],
                    "group_id": r["group_id"],
                    "source": r["source"],
                    "creator": r["creator"],
                    "license": r["license"],
                    "source_url": r["source_url"],
                })
                attr_lines.append(
                    f"| `{fname}` | `{r['category']}` | {r['creator']} | {r['license']} | {r['source_url']} | `{r['sha256'][:12]}...` |"
                )
        finally:
            if src_zf is not None:
                src_zf.close()

        dst_zf.writestr("unsupported_test_ground_truth.csv", csv_buf.getvalue().encode("utf-8"))
        dst_zf.writestr("ATTRIBUTION.md", "\n".join(attr_lines).encode("utf-8"))

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Built {out_zip} ({len(rows)} images, {size_mb:.2f} MB)")


if __name__ == "__main__":
    build_casting_held_out_zip()
    build_unsupported_held_out_zip()

