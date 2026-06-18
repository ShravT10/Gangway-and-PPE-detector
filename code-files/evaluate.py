"""
Evaluate on labeled test split — gives mAP, precision, recall, confusion matrix
---------------------------------------------------------------------------------
Usage:
  python evaluate.py --weights best.pt --data path/to/data.yaml
  python evaluate.py --weights best.pt --data path/to/data.yaml --name dataset2-run1
"""
import os
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["YOLO_OFFLINE"] = "True"

import argparse
from ultralytics import YOLO


def evaluate(weights: str, data_yaml: str, imgsz: int = 640, conf: float = 0.001,
             project: str = "evaluation_res", name: str = None):
    model = YOLO(weights)

    metrics = model.val(
        data=data_yaml,
        split="test",      # evaluates on the 'test' split defined in data.yaml
        imgsz=imgsz,
        conf=conf,          # low conf threshold for proper mAP calculation
        plots=True,         # saves confusion matrix, PR curves, etc.
        project=project,    # root folder for evaluation output (default: evaluation_res)
        name=name,          # subfolder for this run; auto-incremented if None/exists
    )

    print("\n" + "=" * 50)
    print("  TEST SET RESULTS")
    print("=" * 50)
    print(f"  Results saved to : {metrics.save_dir}")
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")

    print("\n  Per-class mAP50-95:")
    for i, name_ in metrics.names.items():
        print(f"    {name_:<10}: {metrics.box.maps[i]:.4f}")

    print("\n  Confusion matrix and plots saved alongside val results.")
    print("=" * 50)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model on labeled test set")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=str, default="evaluation_res",
                         help="Root folder to save evaluation results (default: evaluation_res)")
    parser.add_argument("--name", type=str, default=None,
                         help="Subfolder name for this evaluation run (auto-generated if omitted)")
    args = parser.parse_args()

    evaluate(args.weights, args.data, imgsz=args.imgsz, project=args.project, name=args.name)