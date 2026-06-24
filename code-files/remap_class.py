"""
Remap a Class ID in YOLO Label Files
-----------------------------------------
Changes every line's class ID from one value to another across all .txt
label files in a folder (recursively). Coordinates are left untouched —
only the leading class index is rewritten.

Example:
  Input line:   2 0.341346 0.621687 0.278846 0.53494
  --from 2 --to 5   →   5 0.341346 0.621687 0.278846 0.53494

Usage:
  python remap_class.py --labels path/to/labels --from 2 --to 5
  python remap_class.py --labels path/to/dataset --from 2 --to 5 --recursive
"""

import argparse
from pathlib import Path


def process_file(label_path: Path, from_class: int, to_class: int) -> int:
    """Rewrite a single label file, remapping from_class -> to_class.
    Lines that don't start with an integer class ID (blank lines, comments,
    stray non-label files) are left untouched rather than crashing.
    Returns the number of lines changed."""
    with open(label_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    changed_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        try:
            cls = int(parts[0])
        except ValueError:
            # Not a valid YOLO label line (e.g. a comment starting with '#').
            # Keep the line exactly as it was instead of crashing.
            new_lines.append(stripped)
            continue
        if cls == from_class:
            parts[0] = str(to_class)
            changed_count += 1
        new_lines.append(" ".join(parts))

    with open(label_path, "w") as f:
        for line in new_lines:
            f.write(line + "\n")

    return changed_count


def main(labels_dir: str, from_class: int, to_class: int, recursive: bool, dry_run: bool):
    labels_dir = Path(labels_dir).resolve()
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    pattern = "**/*.txt" if recursive else "*.txt"
    label_files = sorted(labels_dir.glob(pattern))

    if not label_files:
        print(f"No .txt label files found in {labels_dir} (recursive={recursive})")
        return

    print("\n" + "=" * 55)
    print("  REMAP CLASS IN LABELS")
    print("=" * 55)
    print(f"  Labels dir   : {labels_dir}")
    print(f"  Remap        : class {from_class} → class {to_class}")
    print(f"  Recursive    : {recursive}")
    print(f"  Dry run      : {dry_run}")
    print(f"  Files found  : {len(label_files)}")
    print("=" * 55 + "\n")

    total_changed = 0
    files_changed = 0

    for label_path in label_files:
        if dry_run:
            with open(label_path, "r") as f:
                lines = f.readlines()
            changed = 0
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    if int(stripped.split()[0]) == from_class:
                        changed += 1
                except ValueError:
                    continue
            if changed > 0:
                print(f"  [DRY RUN] {label_path.name}: would remap {changed} instance(s)")
                total_changed += changed
                files_changed += 1
            continue

        changed = process_file(label_path, from_class, to_class)
        if changed > 0:
            print(f"  {label_path.name}: remapped {changed} instance(s)")
            total_changed += changed
            files_changed += 1

    print("\n" + "=" * 55)
    print("  DONE")
    print("=" * 55)
    print(f"  Files affected     : {files_changed} / {len(label_files)}")
    print(f"  Total instances remapped : {total_changed}")
    if dry_run:
        print("  (dry run — no files were actually modified)")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remap one class ID to another in YOLO label files")
    parser.add_argument("--labels", required=True, help="Path to folder containing .txt label files")
    parser.add_argument("--from", dest="from_class", type=int, required=True, help="Class ID to convert from (e.g. 2)")
    parser.add_argument("--to", dest="to_class", type=int, required=True, help="Class ID to convert to (e.g. 5)")
    parser.add_argument("--recursive", action="store_true", help="Search subfolders too (e.g. train/labels, valid/labels, test/labels)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without modifying any files")
    args = parser.parse_args()

    main(args.labels, args.from_class, args.to_class, args.recursive, args.dry_run)