
"""
YOLOv8 Detection Dataset Checker
--------------------------------
Checks YOLO bounding-box datasets before training.

Usage:
python check_dataset.py --data path/to/data.yaml
"""

import sys
import yaml
import argparse
from pathlib import Path
from collections import defaultdict

# ── optional visualization ──────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    VISUAL = True
except ImportError:
    VISUAL = False
    print("[WARN] cv2 / matplotlib not found – skipping visualisation.\n")


# ───────────────────────────── helpers ──────────────────────────────────────

COLORS = [
    (255, 56, 56),   # red
    (56, 255, 56),   # green
    (56, 56, 255),   # blue
    (255, 255, 56),  # yellow
    (255, 56, 255),  # magenta
]


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(yaml_dir: Path, rel: str) -> Path:
    p = Path(rel)

    if not p.is_absolute():
        p = (yaml_dir / p).resolve()

    return p


def count_files(folder: Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    if not folder.exists():
        return [], False

    files = [
        f for f in folder.iterdir()
        if f.suffix.lower() in exts
    ]

    return files, True


def parse_label(txt_path: Path):
    """
    YOLO Detection format:

    class x_center y_center width height

    Returns:
        [(class_id, (xc, yc, w, h)), ...]
    """

    objects = []

    try:
        with open(txt_path, "r") as f:

            for line in f:
                parts = line.strip().split()

                if not parts:
                    continue

                try:
                    cls = int(parts[0])

                    if len(parts) != 5:
                        objects.append((cls, None))
                        continue

                    xc, yc, bw, bh = map(float, parts[1:])

                    objects.append(
                        (
                            cls,
                            (xc, yc, bw, bh)
                        )
                    )

                except Exception:
                    objects.append((-1, None))

    except Exception:
        pass

    return objects


def draw_bboxes(img_path: Path, label_path: Path, class_names: list):

    img = cv2.imread(str(img_path))

    if img is None:
        return None

    h, w = img.shape[:2]

    if label_path.exists():

        objects = parse_label(label_path)

        for cls, bbox in objects:

            if bbox is None:
                continue

            xc, yc, bw, bh = bbox

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)

            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            color = COLORS[cls % len(COLORS)]

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = (
                class_names[cls]
                if cls < len(class_names)
                else str(cls)
            )

            cv2.putText(
                img,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    return img


# ───────────────────────────── main checker ─────────────────────────────────


def check(yaml_path: str, preview: int = 4):

    yaml_path = Path(yaml_path).resolve()

    if not yaml_path.exists():
        print(f"[ERROR] data.yaml not found: {yaml_path}")
        sys.exit(1)

    cfg = load_yaml(yaml_path)

    yaml_dir = yaml_path.parent

    nc = cfg.get("nc", 0)
    names = cfg.get("names", [])

    print("=" * 70)
    print("YOLO DETECTION DATASET CHECKER")
    print("=" * 70)

    print(f"YAML    : {yaml_path}")
    print(f"Classes : {nc}")
    print(f"Names   : {names}")
    print()

    splits = {
        "train": cfg.get("train"),
        "val": cfg.get("val"),
        "test": cfg.get("test"),
    }

    total_images = 0
    total_labels = 0

    class_counts = defaultdict(int)
    issues = []

    preview_data = []

    for split, rel in splits.items():

        if rel is None:
            print(f"[{split}] Not defined in YAML")
            continue

        img_dir = resolve_path(yaml_dir, rel)

        lbl_dir_sibling = img_dir.parent / "labels"
        lbl_dir_legacy = (
            img_dir.parent.parent /
            "labels" /
            img_dir.parent.name
        )

        lbl_dir = (
            lbl_dir_sibling
            if lbl_dir_sibling.exists()
            else lbl_dir_legacy
        )

        images, img_ok = count_files(img_dir)
        labels, lbl_ok = count_files(lbl_dir, exts=(".txt",))

        if not img_ok:
            issues.append(
                f"Missing images directory for split '{split}'"
            )
            continue

        print(
            f"[{split}] Images: {len(images):5d} "
            f"Labels: {len(labels):5d}"
        )

        label_names = {f.stem for f in labels}
        image_names = {f.stem for f in images}

        # Check images -> labels
        for img in images:

            total_images += 1

            lbl_path = lbl_dir / f"{img.stem}.txt"

            if img.stem not in label_names:
                issues.append(
                    f"[{split}] Missing label: {img.name}"
                )
                continue

            total_labels += 1

            objects = parse_label(lbl_path)

            if not objects:
                issues.append(
                    f"[{split}] Empty label file: {lbl_path.name}"
                )
                continue

            for cls, bbox in objects:

                if cls < 0:
                    issues.append(
                        f"[{split}] Invalid class ID in "
                        f"{lbl_path.name}"
                    )
                    continue

                if cls >= nc:
                    issues.append(
                        f"[{split}] Class ID {cls} >= nc({nc}) "
                        f"in {lbl_path.name}"
                    )

                class_counts[cls] += 1

                if bbox is None:
                    issues.append(
                        f"[{split}] Malformed annotation in "
                        f"{lbl_path.name}"
                    )
                    continue

                xc, yc, bw, bh = bbox

                if not (
                    0 <= xc <= 1 and
                    0 <= yc <= 1 and
                    0 < bw <= 1 and
                    0 < bh <= 1
                ):
                    issues.append(
                        f"[{split}] Invalid bbox values in "
                        f"{lbl_path.name}"
                    )

            if len(preview_data) < preview:
                preview_data.append((img, lbl_path))

        # Check labels -> images
        extra_labels = label_names - image_names

        for stem in extra_labels:
            issues.append(
                f"[{split}] Label exists but image missing: {stem}.txt"
            )

        print()

    # ───────────────────────── summary ─────────────────────────

    print("-" * 70)

    print(f"Total Images : {total_images}")
    print(f"Total Labels : {total_labels}")

    print()

    print("Class Distribution")

    total_objects = sum(class_counts.values())

    for cls_id, cnt in sorted(class_counts.items()):

        cls_name = (
            names[cls_id]
            if cls_id < len(names)
            else f"cls_{cls_id}"
        )

        pct = (
            (cnt / total_objects) * 100
            if total_objects > 0
            else 0
        )

        bar = "█" * min(40, int(cnt / max(1, total_objects) * 100))

        print(
            f"[{cls_id}] "
            f"{cls_name:<15} "
            f"{cnt:>8} "
            f"({pct:5.2f}%) "
            f"{bar}"
        )

    print()

    # Class imbalance warnings
    for cls_id, cnt in class_counts.items():

        if total_objects == 0:
            continue

        pct = cnt / total_objects

        if pct < 0.01:
            cls_name = (
                names[cls_id]
                if cls_id < len(names)
                else str(cls_id)
            )

            print(
                f"[WARN] Class '{cls_name}' "
                f"represents only {pct:.2%} of annotations"
            )

    print()

    if issues:

        print(f"FOUND {len(issues)} ISSUE(S)\n")

        for issue in issues[:50]:
            print(f"• {issue}")

        if len(issues) > 50:
            print(
                f"\n... and {len(issues) - 50} more issues"
            )

    else:
        print("DATASET LOOKS CLEAN")

    print("=" * 70)

    # ───────────────────── visualization ──────────────────────

    if VISUAL and preview_data:

        cols = min(len(preview_data), 4)

        fig, axes = plt.subplots(
            1,
            cols,
            figsize=(5 * cols, 5)
        )

        if cols == 1:
            axes = [axes]

        for ax, (img_path, lbl_path) in zip(
            axes,
            preview_data
        ):

            annotated = draw_bboxes(
                img_path,
                lbl_path,
                names
            )

            if annotated is None:
                ax.axis("off")
                continue

            annotated_rgb = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            ax.imshow(annotated_rgb)
            ax.set_title(img_path.name, fontsize=8)
            ax.axis("off")

        patches = [
            mpatches.Patch(
                color=[c / 255 for c in COLORS[i % len(COLORS)]],
                label=names[i]
                if i < len(names)
                else f"cls_{i}"
            )
            for i in range(nc)
        ]

        fig.legend(
            handles=patches,
            loc="lower center",
            ncol=max(1, nc),
            fontsize=10,
            frameon=False
        )

        plt.suptitle(
            "Sample Bounding Box Annotations",
            fontsize=12
        )

        plt.tight_layout()
        plt.show()

    elif not VISUAL:
        print(
            "\nInstall opencv-python and matplotlib "
            "to view annotation previews."
        )


# ───────────────────────── entry point ─────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="YOLO Detection Dataset Checker"
    )

    parser.add_argument(
        "--data",
        default="data.yaml",
        help="Path to data.yaml"
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=4,
        help="Number of preview images"
    )

    args = parser.parse_args()

    check(args.data, args.preview)
