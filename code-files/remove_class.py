"""
Remove a Class from YOLO Label Files
-----------------------------------------
Deletes every line belonging to a specified class ID from all .txt label
files in a folder (recursively), leaving every other line untouched.

Example:
  Input line:  5 0.332933 0.453012 0.040865 0.062651
  --remove 5   →  line is deleted entirely from the file

Usage:
  python remove_class.py --labels path/to/labels --remove 5
  python remove_class.py --labels path/to/dataset --remove 5 --recursive
"""

import argparse
from pathlib import Path


def process_file(label_path: Path, remove_class: int) -> tuple[int, int]:
    """Rewrite a single label file, dropping lines for remove_class.
    Returns (lines_kept, lines_removed)."""
    with open(label_path, "r") as f:
        lines = f.readlines()

    kept_lines = []
    removed_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        cls = int(parts[0])
        if cls == remove_class:
            removed_count += 1
        else:
            kept_lines.append(stripped)

    with open(label_path, "w") as f:
        for line in kept_lines:
            f.write(line + "\n")

    return len(kept_lines), removed_count


def main(labels_dir: str, remove_class: int, recursive: bool, dry_run: bool):
    labels_dir = Path(labels_dir).resolve()
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    pattern = "**/*.txt" if recursive else "*.txt"
    label_files = sorted(labels_dir.glob(pattern))

    if not label_files:
        print(f"No .txt label files found in {labels_dir} (recursive={recursive})")
        return

    print("\n" + "=" * 55)
    print("  REMOVE CLASS FROM LABELS")
    print("=" * 55)
    print(f"  Labels dir   : {labels_dir}")
    print(f"  Class to remove : {remove_class}")
    print(f"  Recursive    : {recursive}")
    print(f"  Dry run      : {dry_run}")
    print(f"  Files found  : {len(label_files)}")
    print("=" * 55 + "\n")

    total_removed = 0
    files_changed = 0

    for label_path in label_files:
        if dry_run:
            with open(label_path, "r") as f:
                lines = f.readlines()
            removed = sum(1 for line in lines if line.strip() and int(line.strip().split()[0]) == remove_class)
            if removed > 0:
                print(f"  [DRY RUN] {label_path.name}: would remove {removed} instance(s)")
                total_removed += removed
                files_changed += 1
            continue

        kept, removed = process_file(label_path, remove_class)
        if removed > 0:
            print(f"  {label_path.name}: removed {removed} instance(s), {kept} remaining")
            total_removed += removed
            files_changed += 1

    print("\n" + "=" * 55)
    print("  DONE")
    print("=" * 55)
    print(f"  Files affected      : {files_changed} / {len(label_files)}")
    print(f"  Total instances removed : {total_removed}")
    if dry_run:
        print("  (dry run — no files were actually modified)")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove all instances of a class from YOLO label files")
    parser.add_argument("--labels", required=True, help="Path to folder containing .txt label files")
    parser.add_argument("--remove", type=int, required=True, help="Class ID to remove (e.g. 5)")
    parser.add_argument("--recursive", action="store_true", help="Search subfolders too (e.g. train/labels, valid/labels, test/labels)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without modifying any files")
    args = parser.parse_args()

    main(args.labels, args.remove, args.recursive, args.dry_run)
