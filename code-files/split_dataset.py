"""
Split a Flat YOLO Dataset into Train / Valid / Test Folders
-----------------------------------------------------------------
Converts:
    dataset/
    ├── images/
    ├── labels/
    └── data.yaml

into:
    dataset_split/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── data.yaml

Usage:
  python split_dataset.py --source dataset --dest dataset_split
  python split_dataset.py --source dataset --dest dataset_split --train 0.8 --valid 0.1 --test 0.1
"""

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_image_label_pairs(images_dir: Path, labels_dir: Path):
    """Return list of (image_path, label_path) for every image that has a matching label file."""
    pairs = []
    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    for img_path in sorted(images_dir.iterdir()):
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


def main(source: str, dest: str, train_ratio: float, valid_ratio: float, test_ratio: float, seed: int):
    total_ratio = train_ratio + valid_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"train + valid + test ratios must sum to 1.0 (got {train_ratio} + {valid_ratio} + "
            f"{test_ratio} = {total_ratio})"
        )

    source_dir = Path(source).resolve()
    dest_dir = Path(dest).resolve()

    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    data_yaml = source_dir / "data.yaml"

    if not images_dir.exists():
        raise FileNotFoundError(f"Expected images folder not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Expected labels folder not found: {labels_dir}")

    print("\n" + "=" * 55)
    print("  DATASET SPLITTING")
    print("=" * 55)
    print(f"  Source      : {source_dir}")
    print(f"  Destination : {dest_dir}")
    print(f"  Split ratio : train={train_ratio:.2f}  valid={valid_ratio:.2f}  test={test_ratio:.2f}")
    print("=" * 55 + "\n")

    pairs = find_image_label_pairs(images_dir, labels_dir)
    total = len(pairs)
    if total == 0:
        raise RuntimeError("No matched image/label pairs found — nothing to split.")

    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    train_count = round(total * train_ratio)
    valid_count = round(total * valid_ratio)
    # give test whatever's left, so rounding never loses/gains an image
    test_count = total - train_count - valid_count

    train_pairs = shuffled[:train_count]
    valid_pairs = shuffled[train_count:train_count + valid_count]
    test_pairs = shuffled[train_count + valid_count:]

    splits = {
        "train": train_pairs,
        "valid": valid_pairs,
        "test": test_pairs,
    }

    dest_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_pairs in splits.items():
        dest_images_dir = dest_dir / split_name / "images"
        dest_labels_dir = dest_dir / split_name / "labels"
        copy_pairs(split_pairs, dest_images_dir, dest_labels_dir)
        print(f"  {split_name:<6}: {len(split_pairs)} image/label pairs → {dest_dir / split_name}")

    # copy data.yaml, updating the train/val/test path fields to the new layout
    if data_yaml.exists():
        import yaml
        with open(data_yaml, "r") as f:
            cfg = yaml.safe_load(f)

        cfg["train"] = "train/images"
        cfg["val"] = "valid/images"
        cfg["test"] = "test/images"

        with open(dest_dir / "data.yaml", "w") as f:
            yaml.dump(cfg, f, sort_keys=False)
        print(f"\n  Updated data.yaml written to {dest_dir / 'data.yaml'}")
    else:
        print(f"\n  [WARN] No data.yaml found at {data_yaml} — you'll need to create one manually in {dest_dir}")

    print("\n" + "=" * 55)
    print("  SPLIT COMPLETE")
    print("=" * 55)
    print(f"  Total images : {total}")
    for split_name, split_pairs in splits.items():
        pct = len(split_pairs) / total * 100
        print(f"    {split_name:<6}: {len(split_pairs)} ({pct:.1f}%)")
    print(f"  Saved to     : {dest_dir}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a flat images/labels YOLO dataset into train/valid/test folders")
    parser.add_argument("--source", required=True, help="Path to source dataset folder (contains images/, labels/, data.yaml)")
    parser.add_argument("--dest", required=True, help="Path to destination folder for the split dataset")
    parser.add_argument("--train", type=float, default=0.70, help="Train split ratio (default: 0.70)")
    parser.add_argument("--valid", type=float, default=0.20, help="Valid split ratio (default: 0.20)")
    parser.add_argument("--test", type=float, default=0.10, help="Test split ratio (default: 0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splitting")
    args = parser.parse_args()

    main(args.source, args.dest, args.train, args.valid, args.test, args.seed)
