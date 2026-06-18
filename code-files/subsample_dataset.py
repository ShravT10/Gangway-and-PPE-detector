"""
Subsample a YOLO Dataset — Keep 70/10/20 Train/Valid/Test Ratio
-------------------------------------------------------------------
Cuts down a large YOLO-format dataset to a smaller target size while
preserving the 70% train / 20% test / 10% valid split ratio. Copies
matched image+label pairs into a new output directory, and copies the
original data.yaml alongside it (paths inside data.yaml are relative,
so no edits are needed as long as folder names match).

Usage:
  python subsample_dataset.py --source dataset --dest dataset_5k --total 5000
"""

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# split name -> target fraction of the new total
SPLIT_RATIOS = {
    "train": 0.70,
    "test":  0.20,
    "valid": 0.10,
}


def find_image_label_pairs(images_dir: Path, labels_dir: Path):
    """Return list of (image_path, label_path) for every image that has a matching label file."""
    pairs = []
    if not images_dir.exists():
        return pairs
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            print(f"  [WARN] No label found for {img_path.name} — skipping")
    return pairs


def copy_pairs(pairs, dest_images_dir: Path, dest_labels_dir: Path):
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_labels_dir.mkdir(parents=True, exist_ok=True)
    for img_path, label_path in pairs:
        shutil.copy2(img_path, dest_images_dir / img_path.name)
        shutil.copy2(label_path, dest_labels_dir / label_path.name)


def subsample_split(source_dir: Path, split_name: str, target_count: int, seed: int):
    """Find all valid pairs for a split and randomly sample target_count of them."""
    images_dir = source_dir / split_name / "images"
    labels_dir = source_dir / split_name / "labels"

    pairs = find_image_label_pairs(images_dir, labels_dir)
    available = len(pairs)

    if available == 0:
        print(f"  [ERROR] No image/label pairs found for split '{split_name}' in {images_dir}")
        return []

    if target_count > available:
        print(f"  [WARN] Requested {target_count} for '{split_name}' but only {available} available. "
              f"Using all {available}.")
        target_count = available

    rng = random.Random(seed)
    sampled = rng.sample(pairs, target_count)
    return sampled


def main(source: str, dest: str, total: int, seed: int):
    source_dir = Path(source).resolve()
    dest_dir = Path(dest).resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    # compute target counts per split from the desired total
    targets = {
        split: round(total * ratio)
        for split, ratio in SPLIT_RATIOS.items()
    }

    print("\n" + "=" * 55)
    print("  DATASET SUBSAMPLING")
    print("=" * 55)
    print(f"  Source       : {source_dir}")
    print(f"  Destination  : {dest_dir}")
    print(f"  Target total : {total}")
    for split, count in targets.items():
        print(f"    {split:<8}: {count} images ({SPLIT_RATIOS[split]*100:.0f}%)")
    print("=" * 55 + "\n")

    summary = {}

    for split_name, target_count in targets.items():
        print(f"Processing split: {split_name} (target: {target_count})")
        sampled_pairs = subsample_split(source_dir, split_name, target_count, seed)

        dest_images_dir = dest_dir / split_name / "images"
        dest_labels_dir = dest_dir / split_name / "labels"
        copy_pairs(sampled_pairs, dest_images_dir, dest_labels_dir)

        summary[split_name] = len(sampled_pairs)
        print(f"  Copied {len(sampled_pairs)} image/label pairs to {dest_dir / split_name}\n")

    # copy data.yaml as-is — paths inside are relative (train/images, valid/images, test/images)
    # so they remain valid in the new directory without edits
    source_yaml = source_dir / "data.yaml"
    if source_yaml.exists():
        shutil.copy2(source_yaml, dest_dir / "data.yaml")
        print(f"Copied data.yaml to {dest_dir / 'data.yaml'}")
    else:
        print(f"[WARN] No data.yaml found at {source_yaml} — you'll need to create one manually in {dest_dir}")

    actual_total = sum(summary.values())
    print("\n" + "=" * 55)
    print("  SUBSAMPLING COMPLETE")
    print("=" * 55)
    for split, count in summary.items():
        pct = (count / actual_total * 100) if actual_total else 0
        print(f"  {split:<8}: {count} images ({pct:.1f}%)")
    print(f"  TOTAL   : {actual_total} images")
    print(f"  Saved to: {dest_dir}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample a YOLO dataset while preserving train/test/valid ratio")
    parser.add_argument("--source", required=True, help="Path to source dataset folder (contains train/test/valid/data.yaml)")
    parser.add_argument("--dest", required=True, help="Path to destination folder for the subsampled dataset")
    parser.add_argument("--total", type=int, required=True, help="Target total number of images (e.g. 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    args = parser.parse_args()

    main(args.source, args.dest, args.total, args.seed)
