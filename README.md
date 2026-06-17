# Camera-Project — PPE (Helmet) Compliance Detector

A computer vision system that detects **person**, **helmet**, and **head** (i.e. an unprotected head — a policy violation) using a custom-trained YOLOv8 model. Includes scripts for training, evaluating, visually comparing predictions against ground truth, and a live webcam monitor that flags helmet violations in real time and emails a summary with screenshots when the session ends.

## Project Structure

```
Camera-Project/
├── code-files/
│   ├── check_dataset.py     # dataset/label sanity checks
│   ├── train.py              # train the YOLOv8 detector
│   ├── evaluate.py           # compute mAP/precision/recall on a labeled split
│   ├── compare.py            # side-by-side ground-truth vs prediction images
│   └── webcam_monitor.py     # live detection + violation screenshots + email alert
├── dataset/                  # YOLO-format dataset (images/ + labels/ + data.yaml)
├── runs/
│   ├── detect/ppe-detector/  # training output (weights, plots, logs)
│   ├── val/                  # evaluation output (confusion matrix, PR curves)
│   └── compare/              # side-by-side comparison images
├── violations/                # screenshots saved when "head" (no helmet) is detected
├── yolov8s.pt                 # pretrained base checkpoint used to start training
├── .env                       # your real email credentials (not committed)
├── .env_example                # template showing required environment variables
├── requirements.txt
└── README.md
```

## Classes

| Index | Class    | Meaning                                  |
|-------|----------|-------------------------------------------|
| 0     | head     | Unprotected head detected — **violation** |
| 1     | helmet   | Helmet worn correctly — compliant         |
| 2     | person   | A person detected (general presence)      |

Class order is defined in `dataset/data.yaml` and must stay consistent across training, evaluation, and inference.

## Requirements

- Python 3.11
- NVIDIA GPU with CUDA support (tested on RTX A3000 Laptop GPU)
- A working webcam (for `webcam_monitor.py`)

Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `ultralytics`, `torch` (CUDA build), `opencv-python`, `pyyaml`.

> **GPU note:** Make sure `torch` is installed with CUDA support, not the CPU-only build. Verify with:
> ```python
> import torch
> print(torch.cuda.is_available())  # should print True
> ```

## Environment Setup (Email Alerts)

`webcam_monitor.py` sends an email summary with violation screenshots attached when the session ends. Copy the example file and fill in your own credentials:

```bash
cp .env_example .env
```

Then edit `.env`:

```
SENDER_EMAIL=youraddress@gmail.com
APP_PASSWORD=xxxx xxxx xxxx xxxx
RECIPIENT_EMAIL=youraddress@gmail.com
```

`APP_PASSWORD` must be a Gmail **App Password**, not your normal Google account password:

1. Enable 2-Step Verification on your Google account.
2. Generate an App Password at https://myaccount.google.com/apppasswords.
3. Paste the 16-character code into `.env`.

`.env` is git-ignored and should never be committed. Only `.env_example` (with placeholder values) belongs in version control.

## Dataset

Dataset follows standard YOLOv8 format:

```
dataset/
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
```

`data.yaml` defines paths and class names:

```yaml
train: train/images
val: valid/images
test: test/images

nc: 3
names: ['head', 'helmet', 'person']
```

Run a sanity check on labels and images before training:

```bash
python code-files/check_dataset.py --data dataset/data.yaml
```

## Training

Trains a YOLOv8s detection model on the dataset, with hardware-aware batch sizing and an augmentation profile suited to a real (non-synthetic), few-hundred-to-thousand-image dataset.

```bash
python code-files/train.py --data dataset/data.yaml
```

Optional overrides:

```bash
python code-files/train.py --data dataset/data.yaml --epochs 100 --batch 8 --model yolov8m.pt
python code-files/train.py --data dataset/data.yaml --resume
```

Output (weights, training curves, confusion matrix) is saved to `runs/detect/ppe-detector/`. Best weights end up at:

```
runs/detect/ppe-detector/weights/best.pt
```

## Evaluation

Computes mAP50, mAP50-95, precision, recall, and a per-class breakdown on a labeled split (defaults to `test`):

```bash
python code-files/evaluate.py --weights runs/detect/ppe-detector/weights/best.pt --data dataset/data.yaml
```

Confusion matrix and PR-curve plots are saved to `runs/val/`. Use the per-class mAP output to spot classes that need more training data or label cleanup.

## Visual Comparison (Ground Truth vs Prediction)

Generates side-by-side images — ground truth on the left, model prediction on the right — for a random sample of images from a chosen split:

```bash
python code-files/compare.py --weights runs/detect/ppe-detector/weights/best.pt --data dataset/data.yaml --split test --num 20
```

Output is saved to `runs/compare/`. Useful for spotting systematic errors (e.g. consistent confusion between two classes) that aggregate metrics alone don't reveal.

## Live Webcam Monitoring

Runs real-time detection on your laptop camera. Draws bounding boxes for all three classes, and whenever a **head** (no-helmet) violation is detected, saves a screenshot to `violations/` with a 5-second cooldown to avoid duplicate captures of the same event. When you quit the session (`q`), all violation screenshots from that run are emailed as attachments to the configured recipient.

```bash
python code-files/webcam_monitor.py --weights runs/detect/ppe-detector/weights/best.pt
```

Optional flags:

```bash
python code-files/webcam_monitor.py --weights runs/detect/ppe-detector/weights/best.pt \
    --camera 0 --conf 0.4 --cooldown 5 --out violations --imgsz 640
```

| Flag         | Default        | Description                                  |
|--------------|----------------|-----------------------------------------------|
| `--weights`  | required       | Path to trained `best.pt`                      |
| `--camera`   | `0`            | Camera index (`0` = default laptop webcam)     |
| `--conf`     | `0.4`          | Confidence threshold for detections             |
| `--cooldown` | `5.0`          | Seconds between violation screenshots           |
| `--out`      | `violations`   | Folder to save violation screenshots            |
| `--imgsz`    | `640`          | Inference image size                            |

Press `q` in the camera window to stop the session and trigger the email alert.

## Notes & Known Limitations

- The model is most reliable on **head** and **helmet** detection; **person** detection has historically underperformed in testing due to limited labeled instances in the dataset — treat person-class output with caution until retrained on more balanced data.
- The violation cooldown is global (one timer for the whole session), not per-individual — multiple different people triggering violations within the cooldown window will only produce one screenshot.
- Email sending requires outbound access to `smtp.gmail.com:465`; this may be blocked on restrictive corporate networks.

## Next Steps / Ideas

- Add more labeled "person" instances to the training set to fix the underperforming class.
- Require violations to persist across N consecutive frames before saving a screenshot, to reduce false positives from single noisy frames.
- Add per-person tracking (e.g. with a tracker like ByteTrack) for more accurate individual-level violation logging instead of a single global cooldown.