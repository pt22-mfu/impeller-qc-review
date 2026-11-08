"""Audit the casting archive and create a reproducible, grouped split."""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.fft import dctn
from sklearn.model_selection import StratifiedGroupKFold

LABELS = {"ok_front": 0, "def_front": 1}
SPLITS = ("train", "validation", "test")
SEED = 1501418
HASH_DISTANCE = 2


def read_archive(archive_path):
    records = []
    unreadable = []
    with ZipFile(archive_path) as archive:
        members = sorted(archive.infolist(), key=lambda member: member.filename)
        for member in members:
            parts = PurePosixPath(member.filename).parts
            if member.is_dir() or len(parts) < 3:
                continue
            split, label, filename = parts[-3:]
            if split not in ("train", "test") or label not in LABELS:
                continue
            if Path(filename).suffix.lower() not in (".jpeg", ".jpg", ".png"):
                continue
            raw = archive.read(member)
            try:
                with Image.open(io.BytesIO(raw)) as image:
                    rgb = image.convert("RGB")
                    gray = rgb.convert("L")
            except (OSError, ValueError, UnidentifiedImageError):
                unreadable.append(member.filename)
                continue
            small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
            frequencies = dctn(small, norm="ortho")[:8, :8].ravel()[1:]
            hash_bits = np.packbits(frequencies > np.median(frequencies))
            records.append({
                "image_path": member.filename,
                "original_split": split,
                "label": label,
                "target": LABELS[label],
                "width": rgb.width,
                "height": rgb.height,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "pixel_sha256": hashlib.sha256(
                    f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()
                ).hexdigest(),
                "phash": np.uint64(int.from_bytes(hash_bits.tobytes(), "big")),
                "brightness": round(float(small.mean()), 4),
                "contrast": round(float(small.std()), 4),
            })
    if not records or set(record["target"] for record in records) != {0, 1}:
        raise ValueError("Expected both casting classes in train/test folders inside the ZIP.")
    return records, unreadable


def group_images(records):
    """Group similar images conservatively to keep candidate variants together."""
    parent = list(range(len(records)))

    def root(index):
        while index != parent[index]:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    hashes = np.array([record["phash"] for record in records], dtype=np.uint64)
    pair_count = 0
    cross_label_pairs = 0
    for index, fingerprint in enumerate(hashes):
        distances = np.bitwise_count(hashes[index + 1:] ^ fingerprint)
        matches = np.flatnonzero(distances <= HASH_DISTANCE) + index + 1
        pair_count += len(matches)
        for match in matches:
            cross_label_pairs += records[index]["target"] != records[match]["target"]
            left, right = root(index), root(int(match))
            if left != right:
                parent[max(left, right)] = min(left, right)
    groups = [root(index) for index in range(len(records))]
    return groups, pair_count, cross_label_pairs


def assign_splits(records, groups):
    targets = np.array([record["target"] for record in records])
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    folds = [indices for _, indices in splitter.split(targets, targets, groups)]
    ideal_counts = np.bincount(targets) * 0.15
    fold_counts = [np.bincount(targets[indices], minlength=2) for indices in folds]
    ranked = sorted(range(len(folds)), key=lambda index: sum(abs(fold_counts[index] - ideal_counts)))
    split_for_index = ["train"] * len(records)
    for split, fold in zip(("validation", "test"), ranked):
        for index in folds[fold]:
            split_for_index[index] = split
    observed = defaultdict(set)
    for index, record in enumerate(records):
        split = split_for_index[index]
        observed[groups[index]].add(split)
        record["group_id"] = f"g{groups[index]:05d}"
        record["split"] = split
        record["phash"] = f"{record['phash']:016x}"
    if any(len(splits) > 1 for splits in observed.values()):
        raise RuntimeError("A similarity group crosses splits.")
    if any(len({r["target"] for r in records if r["split"] == split}) != 2 for split in SPLITS):
        raise RuntimeError("A split is missing a class. Review the similarity grouping.")


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(archive_path, output_dir):
    all_records, unreadable = read_archive(archive_path)
    by_pixels = defaultdict(list)
    for record in all_records:
        by_pixels[record["pixel_sha256"]].append(record)

    duplicates = []
    retained = []
    cross_split_exact = 0
    for copies in by_pixels.values():
        if len({copy["original_split"] for copy in copies}) > 1:
            cross_split_exact += 1
        conflict = len({copy["label"] for copy in copies}) > 1
        canonical = copies[0] if not conflict else None
        if canonical is not None:
            retained.append(canonical)
        for copy in copies if conflict else copies[1:]:
            duplicates.append({
                "removed_path": copy["image_path"],
                "retained_path": canonical["image_path"] if canonical is not None else "",
                "reason": "conflicting labels" if conflict else "identical pixels",
            })

    groups, pair_count, cross_label_pairs = group_images(retained)
    assign_splits(retained, groups)
    counts = {
        split: dict(Counter(row["label"] for row in retained if row["split"] == split))
        for split in SPLITS
    }
    summary = {
        "readable_images": len(all_records),
        "unreadable_images": unreadable,
        "removed_images": len(duplicates),
        "retained_images": len(retained),
        "original_cross_split_exact_pixel_groups": cross_split_exact,
        "phash_max_hamming_distance": HASH_DISTANCE,
        "candidate_pair_count": int(pair_count),
        "candidate_cross_label_pair_count": int(cross_label_pairs),
        "group_count": len(set(groups)),
        "largest_group": max(Counter(groups).values()),
        "seed": SEED,
        "counts_by_split_and_class": counts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "split_manifest.csv", list(retained[0]), retained)
    write_csv(output_dir / "removed_images.csv", ["removed_path", "retained_path", "reason"], duplicates)
    (output_dir / "audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to the casting dataset ZIP")
    parser.add_argument("--output", type=Path, default=Path("outputs/data"))
    args = parser.parse_args()
    prepare(args.archive, args.output)
