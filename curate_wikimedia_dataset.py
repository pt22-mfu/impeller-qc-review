"""Curate a first batch of 60-100 unsupported negative photos from Wikimedia Commons.

Extracts:
  - image_path: Local saved path in unsupported_data/
  - category: bearing, pump_housing, round_metal_component, misoriented_impeller, unrelated_industrial, unrelated_general
  - source: 'Wikimedia Commons'
  - license: Verifiable license name (CC BY, CC BY-SA, CC0, Public Domain)
  - group_id: Source/part group (keeps multiple views of the same part together)
  - split: Leakage-free train / validation / test assignment
  - is_supported: 0
  - width, height: Pixel dimensions
  - sha256: SHA-256 checksum of raw image bytes
  - source_url: Original file-page URL on Wikimedia Commons
  - creator: Original author / artist

Also exports outputs/data/unsupported_manifest.csv and detailed metadata.
"""
import csv
import hashlib
import io
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

USER_AGENT = "ImpellerQCResearchBot/1.0 (industrial-qc-research; contact@impeller-qc.local)"
SEED = 1501418

# Configuration for category curation
CATEGORIES_CONFIG = {
    "bearing": {
        "folder": Path("unsupported_data/similar_parts/bearings"),
        "target_count": 20,
        "queries": [
            "incategory:Ball_bearings filetype:bitmap",
            "incategory:Roller_bearings filetype:bitmap",
            "incategory:Spherical_roller_bearings filetype:bitmap",
            "ball bearing white background filetype:bitmap",
            "thrust bearing filetype:bitmap",
        ],
        "keywords": ["bearing", "kugellager", "roulement", "cuscinetto"],
    },
    "pump_housing": {
        "folder": Path("unsupported_data/similar_parts/pump_housings"),
        "target_count": 16,
        "queries": [
            "pump housing filetype:bitmap",
            "pump casing filetype:bitmap",
            "volute casing filetype:bitmap",
            "water pump casing filetype:bitmap",
            "centrifugal pump casing filetype:bitmap",
        ],
        "keywords": ["pump", "casing", "housing", "volute", "gehäuse", "pumphus"],
    },
    "round_metal_component": {
        "folder": Path("unsupported_data/similar_parts/round_metal_components"),
        "target_count": 20,
        "queries": [
            "incategory:Flanges filetype:bitmap",
            "incategory:Gears filetype:bitmap",
            "incategory:Pulleys filetype:bitmap",
            "incategory:Sprockets filetype:bitmap",
            "metal flange filetype:bitmap",
            "metal washer white background filetype:bitmap",
            "brake rotor filetype:bitmap",
        ],
        "keywords": ["flange", "gear", "pulley", "sprocket", "washer", "rotor", "disc", "scheibe"],
    },
    "misoriented_impeller": {
        "folder": Path("unsupported_data/similar_parts/misoriented_impellers"),
        "target_count": 10,
        "queries": [
            "compressor impeller filetype:bitmap",
            "water pump impeller filetype:bitmap",
            "impeller blade filetype:bitmap",
            "centrifugal impeller filetype:bitmap",
            "incategory:Impellers filetype:bitmap",
        ],
        "keywords": ["impeller", "runner", "compressor", "turbine"],
    },
    "unrelated_industrial": {
        "folder": Path("unsupported_data/unrelated_objects/industrial_hardware"),
        "target_count": 12,
        "queries": [
            "incategory:Wrenches filetype:bitmap",
            "incategory:Calipers filetype:bitmap",
            "incategory:Pliers filetype:bitmap",
            "spanner tool white background filetype:bitmap",
            "adjustable wrench white background filetype:bitmap",
        ],
        "keywords": ["wrench", "spanner", "caliper", "pliers", "tool", "schlüssel"],
    },
    "unrelated_general": {
        "folder": Path("unsupported_data/unrelated_objects/general_objects"),
        "target_count": 12,
        "queries": [
            "banana white background filetype:bitmap",
            "single banana filetype:bitmap",
            "apple white background filetype:bitmap",
            "orange white background filetype:bitmap",
            "bunch of bananas filetype:bitmap",
        ],
        "keywords": ["banana", "bananas", "apple", "fruit", "orange"],
    },
}

ALLOWED_LICENSES = {
    "public domain", "cc0", "cc-zero", "cc by", "cc by-sa",
    "cc-by", "cc-by-sa", "gfdl", "fal", "free art", "mit"
}


def clean_html(text: str) -> str:
    if not text:
        return "Unknown"
    text = re.sub(r"<[^>]+>", " ", text)
    cleaned = " ".join(text.split())
    return cleaned if cleaned else "Unknown"


def is_permissive_license(license_str: str) -> bool:
    if not license_str:
        return False
    lic_lower = license_str.lower()
    return any(allowed in lic_lower for allowed in ALLOWED_LICENSES)


def search_commons_category(query: str, limit: int = 15):
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 600,
        "format": "json",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"      Search failed for '{query}': {e}")
        return []

    pages = data.get("query", {}).get("pages", {})
    results = []

    for page in pages.values():
        if "imageinfo" not in page:
            continue
        ii = page["imageinfo"][0]
        mime = ii.get("mime", "")
        if mime not in ("image/jpeg", "image/png"):
            continue

        meta = ii.get("extmetadata", {})
        license_name = meta.get("LicenseShortName", {}).get("value", "")
        if not is_permissive_license(license_name):
            continue

        title = page.get("title", "")
        # Filter out obvious non-photo titles (e.g. diagrams, icons, logos, flags, maps)
        title_lower = title.lower()
        if any(skip in title_lower for skip in ["diagram", "icon", "logo", "flag", "map", "schematic", "drawing", "symbol", "chart"]):
            continue

        artist = clean_html(meta.get("Artist", {}).get("value", "Wikimedia Contributor"))
        desc = clean_html(meta.get("ImageDescription", {}).get("value", ""))
        desc_url = ii.get("descriptionurl", "")
        thumb_url = ii.get("thumburl", ii.get("url"))

        results.append({
            "title": title,
            "thumb_url": thumb_url,
            "original_url": ii.get("url"),
            "desc_url": desc_url,
            "width": ii.get("width"),
            "height": ii.get("height"),
            "license": license_name,
            "artist": artist,
            "description": desc,
        })

    return results


def download_and_verify_image(url: str):
    """Download image, verify with PIL, and convert to RGB."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_bytes = resp.read()
    except Exception as e:
        return None, None, None

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            rgb = img.convert("RGB")
            # Ensure sensible dimensions
            if rgb.width < 100 or rgb.height < 100:
                return None, None, None
            # Standardize format to high-quality JPEG
            buffer = io.BytesIO()
            rgb.save(buffer, format="JPEG", quality=95)
            final_bytes = buffer.getvalue()
            return rgb.width, rgb.height, final_bytes
    except Exception:
        return None, None, None


def derive_group_id(title: str, category: str, existing_groups: dict) -> str:
    """Detect multi-view sets or assign clean part group."""
    clean_title = re.sub(r"^File:", "", title)
    clean_title = re.sub(r"\.(jpg|jpeg|png)$", "", clean_title, flags=re.I).strip()
    base_stem = re.sub(r"[^a-zA-Z0-9]+", "_", clean_title[:45].lower()).strip("_")

    if category == "misoriented_impeller":
        if "24_blade_compressor" in base_stem:
            return "grp_impeller_nara_24_blade"
        elif "chrysler" in base_stem:
            return "grp_impeller_chrysler_gas_turbine"
        elif "damaged_compressor_impeller" in base_stem:
            return "grp_impeller_nara_damaged"
        elif "damaged_compressor_parts_of_r_31" in base_stem:
            return "grp_impeller_r31_damaged_parts"
        elif "roots" in base_stem or "case_impeller" in base_stem:
            return "grp_impeller_case_roots"
        elif "174432" in base_stem:
            return "grp_impeller_nara_compressor_single"
        else:
            return f"grp_impeller_{base_stem[:25]}"

    # Check for multi-view pump casing sets
    if "pumphus" in base_stem:
        return "grp_water_pump_casing_set"

    # Check for multi-view dredging flanges
    if "baggerwerkzaamheden" in base_stem:
        return "grp_round_metal_langwarder_flanges"
    if "blue_grass_chemical" in base_stem:
        return "grp_round_metal_blue_grass_flanges"

    return f"grp_{category}_{base_stem[:30]}"


def assign_category_stratified_group_splits(records: list[dict], seed: int = SEED):
    """Assign train / validation / test splits stratified by category while keeping groups atomic."""
    rng = np.random.default_rng(seed)
    by_category = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r)

    group_split_map = {}

    for cat, cat_records in sorted(by_category.items()):
        cat_groups = sorted(list(set(r["group_id"] for r in cat_records)))
        rng.shuffle(cat_groups)
        n = len(cat_groups)

        if n == 1:
            group_split_map[cat_groups[0]] = "train"
        elif n == 2:
            group_split_map[cat_groups[0]] = "train"
            group_split_map[cat_groups[1]] = "validation"
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

    return records


def curate_dataset():
    print("=" * 75)
    print("Stage 6A: Curate 60–100 Unsupported Negative Photos from Wikimedia Commons")
    print("=" * 75)

    all_downloaded = []
    seen_titles = set()
    category_counts = Counter()

    for category, cfg in CATEGORIES_CONFIG.items():
        folder = cfg["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        target = cfg["target_count"]
        print(f"\n[+] Curating category '{category}' (Target: ~{target} images)...")

        cat_downloaded = 0
        existing_groups = {}

        for query in cfg["queries"]:
            if cat_downloaded >= target:
                break
            print(f"    Searching: '{query}'...")
            candidates = search_commons_category(query, limit=20)
            time.sleep(0.3)  # Polite API pause

            for item in candidates:
                if cat_downloaded >= target:
                    break
                title = item["title"]
                if title in seen_titles:
                    continue

                width, height, img_bytes = download_and_verify_image(item["thumb_url"])
                if img_bytes is None:
                    continue

                seen_titles.add(title)
                cat_downloaded += 1
                category_counts[category] += 1

                # Clean filename
                safe_stem = re.sub(r"[^a-zA-Z0-9]+", "_", title.replace("File:", "")[:35].lower()).strip("_")
                filename = f"{category}_{cat_downloaded:03d}_{safe_stem}.jpg"
                file_path = folder / filename
                file_path.write_bytes(img_bytes)

                sha256 = hashlib.sha256(img_bytes).hexdigest()
                group_id = derive_group_id(title, category, existing_groups)

                all_downloaded.append({
                    "image_path": file_path.as_posix(),
                    "category": category,
                    "source": "Wikimedia Commons",
                    "license": item["license"],
                    "group_id": group_id,
                    "is_supported": 0,
                    "width": width,
                    "height": height,
                    "sha256": sha256,
                    "source_url": item["desc_url"],
                    "creator": item["artist"][:100],
                })
                print(f"      [{cat_downloaded:02d}/{target}] Saved: {filename} ({width}x{height}, {item['license']})")
                time.sleep(0.1)

        print(f"    Completed '{category}': {cat_downloaded} images collected.")

    total_images = len(all_downloaded)
    print(f"\nTotal downloaded images: {total_images}")

    # Assign Leakage-Free Grouped Splits Stratified by Category
    print("\nAssigning group-aware train / validation / test splits stratified by category...")
    all_downloaded = assign_category_stratified_group_splits(all_downloaded, seed=SEED)
    unique_groups = sorted(list(set(r["group_id"] for r in all_downloaded)))
    n_groups = len(unique_groups)

    # Write Manifest
    manifest_path = Path("outputs/data/unsupported_manifest.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
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
        writer.writerows(all_downloaded)

    print(f"\nSaved manifest to: {manifest_path}")

    # Also save detailed JSON metadata
    meta_path = Path("outputs/data/unsupported_metadata.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(all_downloaded, f, indent=2)
    print(f"Saved extended metadata to: {meta_path}")

    # Report Category Counts by Split
    print("\n" + "=" * 70)
    print("STAGE 6A: UNSUPPORTED DATASET SUMMARY")
    print("=" * 70)
    print(f"Total Images Collected: {total_images} (Target: 60–100)")
    print(f"Total Unique Groups:   {n_groups}")

    split_counts = Counter(r["split"] for r in all_downloaded)
    print(f"\nOverall Split Breakdown:")
    for s in ["train", "validation", "test"]:
        cnt = split_counts[s]
        print(f"  - {s:<12}: {cnt:2d} images ({cnt/total_images:.1%})")

    # Category by split matrix
    print("\nCategory Counts by Split:")
    cat_split_counts = defaultdict(Counter)
    for r in all_downloaded:
        cat_split_counts[r["category"]][r["split"]] += 1

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

    # Report any categories that could not reach target
    print("\nCategory Fulfillment Report:")
    for cat, cfg in CATEGORIES_CONFIG.items():
        actual = category_counts[cat]
        target = cfg["target_count"]
        status = "FILLED" if actual >= target else f"PARTIAL ({actual}/{target})"
        print(f"  * {cat:<26}: {actual:2d} images (Target: {target}) -> {status}")


if __name__ == "__main__":
    curate_dataset()
