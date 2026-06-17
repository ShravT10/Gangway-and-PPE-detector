"""
Side-by-side comparison: Ground Truth vs Model Prediction
-------------------------------------------------------------
Usage:
  python compare.py --weights runs/detect/ppe-detector/weights/best.pt \
                     --data dataset/data.yaml --split test --num 20
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


# distinct colors per class (BGR) — adjust if you have more/fewer classes
COLORS = {
    0: (60, 60, 230),    # head    - red-ish
    1: (60, 200, 60),    # helmet  - green
    2: (230, 160, 30),   # person  - blue-ish
}


def load_yolo_labels(label_path: Path, img_w: int, img_h: int):
    """Read a YOLO-format label file and convert to pixel-space boxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = map(float, parts)
            cls = int(cls)
            x1 = int((xc - w / 2) * img_w)
            y1 = int((yc - h / 2) * img_h)
            x2 = int((xc + w / 2) * img_w)
            y2 = int((yc + h / 2) * img_h)
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def draw_boxes(img, boxes, names, is_pred=False, confs=None):
    img = img.copy()
    for i, box in enumerate(boxes):
        cls, x1, y1, x2, y2 = box
        color = COLORS.get(cls, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = names[cls]
        if is_pred and confs is not None:
            label += f" {confs[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main(weights, data_yaml, split, num_images, conf, out_dir):
    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)

    data_root = Path(data_yaml).parent
    base_path = data_cfg.get("path")
    if base_path:
        data_root = Path(base_path)

    img_dir = data_root / data_cfg[split]
    # labels dir mirrors images dir, swapping "images" -> "labels"
    label_dir = Path(str(img_dir).replace("images", "labels"))

    names = data_cfg["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    model = YOLO(weights)

    img_paths = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not img_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    random.seed(42)
    sample = random.sample(img_paths, min(num_images, len(img_paths)))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Comparing {len(sample)} images from {img_dir} ...")

    for img_path in sample:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # ── ground truth ──
        label_path = label_dir / (img_path.stem + ".txt")
        gt_boxes = load_yolo_labels(label_path, w, h)
        gt_img = draw_boxes(img, gt_boxes, names, is_pred=False)
        cv2.putText(gt_img, "GROUND TRUTH", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(gt_img, "GROUND TRUTH", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # ── prediction ──
        result = model.predict(source=str(img_path), conf=conf, verbose=False)[0]
        pred_boxes = []
        confs = []
        for box in result.boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pred_boxes.append((cls, x1, y1, x2, y2))
            confs.append(float(box.conf))
        pred_img = draw_boxes(img, pred_boxes, names, is_pred=True, confs=confs)
        cv2.putText(pred_img, "PREDICTION", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(pred_img, "PREDICTION", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # ── combine side by side ──
        combined = np.hstack([gt_img, pred_img])
        out_path = out_dir / f"compare_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), combined)

    print(f"\nDone. {len(sample)} side-by-side comparisons saved to: {out_dir}")
    print("Each image: LEFT = ground truth, RIGHT = model prediction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Side-by-side GT vs prediction comparison")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which split to sample from")
    parser.add_argument("--num", type=int, default=20, help="Number of images to compare")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions")
    parser.add_argument("--out", default="runs/compare", help="Output directory")
    args = parser.parse_args()

    main(args.weights, args.data, args.split, args.num, args.conf, args.out)