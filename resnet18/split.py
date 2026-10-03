#!/usr/bin/env python3
"""Create deterministic, chronological train/validation splits.

The supplied images come from one video per class.  This script therefore
holds out the last temporal segment of each class and removes a guard band at
the boundary instead of randomly mixing neighboring frames.
"""

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def create_splits(metadata_path: Path, val_fraction: float = 0.2, gap: int = 2) -> dict:
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows or not {"path", "label", "frame_index"}.issubset(fieldnames):
        raise ValueError("metadata.csv must contain path, label, and frame_index columns")
    if not 0.05 <= val_fraction <= 0.5:
        raise ValueError("val_fraction must be between 0.05 and 0.5")
    if gap < 0:
        raise ValueError("gap must be non-negative")

    by_class = {}
    for row in rows:
        by_class.setdefault(row["label"], []).append(row)
    if len(by_class) < 2:
        raise ValueError("At least two classes are required")

    assignments = {}
    paths = [row['path'] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate image paths in metadata')
    for label, class_rows in by_class.items():
        ordered = sorted(class_rows, key=lambda row: int(row["frame_index"]))
        if len(ordered) < 4:
            raise ValueError(f"Class {label!r} needs at least four images")
        val_count = max(1, math.ceil(len(ordered) * val_fraction))
        boundary = len(ordered) - val_count
        train_end = max(0, boundary - gap)
        val_start = min(len(ordered), boundary + gap)
        if train_end == 0 or val_start == len(ordered):
            raise ValueError(f'Gap is too large for class {label!r}; train and validation must both be non-empty')
        for row in ordered[:train_end]:
            assignments[row["path"]] = "train"
        for row in ordered[val_start:]:
            assignments[row["path"]] = "validation"

    if "split" not in fieldnames:
        fieldnames.append("split")
    for row in rows:
        row["split"] = assignments.get(row["path"], "excluded")
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for label in sorted(by_class):
        summary[label] = {
            split: sum(r["label"] == label and r["split"] == split for r in rows)
            for split in ("train", "validation", "excluded")
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "metadata.csv")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--gap", type=int, default=2, help="sampled frames excluded on EACH side of the split boundary")
    args = parser.parse_args()
    for label, counts in create_splits(args.metadata, args.val_fraction, args.gap).items():
        print(f"{label}: train={counts['train']} validation={counts['validation']} excluded={counts['excluded']}")


if __name__ == "__main__":
    main()
