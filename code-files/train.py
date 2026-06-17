"""
Train YOLOv8 — Person / Helmet / Head Detection
-------------------------------------------------
Dataset: ~1000 real images, bounding-box labels (YOLO format), 3 classes
Usage: python train.py --data path/to/data.yaml
"""

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


DEFAULT_CFG = {
    # ── model ──
    "model"        : "yolov8s.pt",   # small detection model, pretrained on COCO
    "epochs"       : 150,
    "imgsz"        : 640,

    # ── batch & hardware ──
    "batch"        : 16,             # adjusted automatically below based on VRAM
    "device"       : "0",
    "workers"      : 8,

    # ── optimiser (use YOLO's battle-tested defaults) ──
    "optimizer"    : "auto",         # lets Ultralytics pick SGD w/ tuned schedule
    "lr0"          : 0.01,
    "lrf"          : 0.01,
    "momentum"     : 0.937,
    "weight_decay" : 0.0005,
    "warmup_epochs": 3,

    # ── augmentation (moderate — real, non-augmented dataset, fixed-orientation objects) ──
    "hsv_h"        : 0.015,
    "hsv_s"        : 0.7,
    "hsv_v"        : 0.4,
    "degrees"      : 0.0,            # people/helmets are upright — skip rotation
    "translate"    : 0.1,
    "scale"        : 0.5,
    "shear"        : 0.0,
    "perspective"  : 0.0,
    "flipud"       : 0.0,
    "fliplr"       : 0.5,            # horizontal flip is fine and useful
    "mosaic"       : 1.0,            # still helpful at 1000 images
    "mixup"        : 0.0,            # skip — not needed with real, decent-sized dataset
    "copy_paste"   : 0.0,            # detection task, not segmentation — leave off

    # ── training behaviour ──
    "patience"     : 30,
    "save_period"  : 10,
    "val"          : True,
    "plots"        : True,
    "project"      : "runs/detect",
    "name"         : "ppe-detector",
}


def check_gpu():
    print("\n" + "=" * 55)
    print("  HARDWARE CHECK")
    print("=" * 55)
    if torch.cuda.is_available():
        idx  = 0
        name = torch.cuda.get_device_name(idx)
        vram = torch.cuda.get_device_properties(idx).total_memory / 1e9
        print(f"  GPU found : {name}")
        print(f"  VRAM      : {vram:.1f} GB")
        print(f"  CUDA      : {torch.version.cuda}")

        if vram < 6:
            DEFAULT_CFG["batch"] = 8
            print("  [INFO] Limited VRAM — batch size set to 8")
        elif vram < 10:
            DEFAULT_CFG["batch"] = 16
            print("  [INFO] Moderate VRAM — batch size set to 16")
        else:
            DEFAULT_CFG["batch"] = 32
            print("  [INFO] Plenty of VRAM — batch size set to 32")
    else:
        print("  No GPU detected — falling back to CPU")
        print("  Training will be significantly slower")
        DEFAULT_CFG["device"] = "cpu"
        DEFAULT_CFG["batch"]  = 8
        DEFAULT_CFG["imgsz"]  = 416
    print("=" * 55 + "\n")


def train(data_yaml: str, resume: bool = False, overrides: dict = None):
    check_gpu()

    data_yaml = Path(data_yaml).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    cfg = {**DEFAULT_CFG, **(overrides or {})}
    cfg["data"] = str(data_yaml)

    print("  TRAINING CONFIG")
    print("=" * 55)
    for k, v in cfg.items():
        print(f"  {k:<20} {v}")
    print("=" * 55 + "\n")

    if resume:
        last_ckpt = Path(cfg["project"]) / cfg["name"] / "weights" / "last.pt"
        if not last_ckpt.exists():
            print("[WARN] No checkpoint found to resume from. Starting fresh.")
            model = YOLO(cfg["model"])
        else:
            print(f"[INFO] Resuming from {last_ckpt}")
            model = YOLO(str(last_ckpt))
    else:
        model = YOLO(cfg["model"])

    results = model.train(
        data         = cfg["data"],
        epochs       = cfg["epochs"],
        imgsz        = cfg["imgsz"],
        batch        = cfg["batch"],
        device       = cfg["device"],
        workers      = cfg["workers"],
        optimizer    = cfg["optimizer"],
        lr0          = cfg["lr0"],
        lrf          = cfg["lrf"],
        momentum     = cfg["momentum"],
        weight_decay = cfg["weight_decay"],
        warmup_epochs= cfg["warmup_epochs"],
        hsv_h        = cfg["hsv_h"],
        hsv_s        = cfg["hsv_s"],
        hsv_v        = cfg["hsv_v"],
        degrees      = cfg["degrees"],
        translate    = cfg["translate"],
        scale        = cfg["scale"],
        shear        = cfg["shear"],
        perspective  = cfg["perspective"],
        flipud       = cfg["flipud"],
        fliplr       = cfg["fliplr"],
        mosaic       = cfg["mosaic"],
        mixup        = cfg["mixup"],
        copy_paste   = cfg["copy_paste"],
        patience     = cfg["patience"],
        save_period  = cfg["save_period"],
        val          = cfg["val"],
        plots        = cfg["plots"],
        project      = cfg["project"],
        name         = cfg["name"],
        exist_ok     = resume,
    )

    save_dir = Path(results.save_dir)
    best_pt  = save_dir / "weights" / "best.pt"

    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)
    print(f"  Results saved to : {save_dir}")
    print(f"  Best weights     : {best_pt}")

    # final validation + per-class breakdown
    metrics = model.val()
    print("\n  Per-class mAP50-95:")
    for i, name in metrics.names.items():
        try:
            print(f"    {name:<10}: {metrics.box.maps[i]:.4f}")
        except Exception:
            pass

    print("\n  Next step → run inference with your trained weights:")
    print(f"    yolo predict model={best_pt} source=path/to/images")
    print("=" * 55)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 detector — person/helmet/head")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--model", type=str, default=None, help="Override base model (e.g. yolov8m.pt)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args = parser.parse_args()

    overrides = {}
    if args.epochs: overrides["epochs"] = args.epochs
    if args.batch:  overrides["batch"]  = args.batch
    if args.model:  overrides["model"]  = args.model

    train(args.data, resume=args.resume, overrides=overrides)