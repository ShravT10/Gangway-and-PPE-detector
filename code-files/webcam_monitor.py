"""
Live Webcam PPE Monitor — Person / Helmet / Head Detection
---------------------------------------------------------------
Detects in real time from the laptop camera. Saves a screenshot to a
violations folder whenever a "no helmet" condition is detected (matched
by class NAME, not a hardcoded index — works across datasets with
different class orderings), with a 5-second cooldown. On quitting the
camera session, emails a summary with all violation images attached.

Usage:
  python webcam_monitor.py --weights runs/detect/ppe-detector/weights/best.pt
"""
import os
from dotenv import load_dotenv
import argparse
import time
import smtplib
import ssl
from pathlib import Path
from datetime import datetime
from email.message import EmailMessage

import cv2
from ultralytics import YOLO

load_dotenv()
email = os.getenv('EMAIL')
email_code = os.getenv('EMAIL_CODE')

# Names (lowercased) that count as a "violation" — i.e. no helmet detected.
# Matched by name, not hardcoded index, so this works across datasets where
# class order differs (e.g. dataset 1: ['head','helmet','person'] vs
# dataset 2: ['Helmet','No Helmet','Worker']).
VIOLATION_CLASS_NAMES = {"head", "no helmet", "no_helmet", "nohelmet"}

# fallback color palette by index, used to assign a consistent color per class
FALLBACK_COLORS = [
    (60, 60, 230), (60, 200, 60), (230, 160, 30), (200, 200, 60), (160, 60, 200),
]


def build_violation_index_set(names: dict) -> set:
    """Given model.names ({idx: 'ClassName'}), return the set of indices that count as violations."""
    violation_indices = set()
    for idx, name in names.items():
        if name.strip().lower() in VIOLATION_CLASS_NAMES:
            violation_indices.add(idx)
    return violation_indices


def build_color_map(names: dict) -> dict:
    """Assign a consistent color per class index, regardless of class name/order."""
    return {idx: FALLBACK_COLORS[i % len(FALLBACK_COLORS)] for i, idx in enumerate(sorted(names.keys()))}


# ──────────────────────────── EMAIL CONFIG ────────────────────────────
# Fill these in via .env (EMAIL, EMAIL_CODE). Use a Gmail "App Password",
# not your real password.
EMAIL_CONFIG = {
    "sender_email":    email,
    "app_password":    email_code,
    "recipient_email": email,
    "smtp_server":     "smtp.gmail.com",
    "smtp_port":        465,
}
# ────────────────────────────────────────────────────────────────────────


def draw_detections(frame, boxes, names, colors):
    for box in boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = colors.get(cls, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{names[cls]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def send_violation_email(config, violation_paths):
    """Send one email with all violation screenshots from this session attached."""
    if not violation_paths:
        print("No violations this session — skipping email.")
        return

    msg = EmailMessage()
    msg["Subject"] = f"Helmet Policy Violation Alert — {len(violation_paths)} incident(s)"
    msg["From"] = config["sender_email"]
    msg["To"] = config["recipient_email"]

    body_lines = [
        "Safety Monitoring System Alert",
        "",
        f"This person violated the helmet policy {len(violation_paths)} time(s) during this session.",
        "",
        "Timestamps:",
    ]
    for p in violation_paths:
        body_lines.append(f"  - {p.name}")
    body_lines.append("\nSee attached images for details.")
    msg.set_content("\n".join(body_lines))

    for p in violation_paths:
        with open(p, "rb") as f:
            img_data = f.read()
        msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename=p.name)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(config["smtp_server"], config["smtp_port"], context=context) as server:
            server.login(config["sender_email"], config["app_password"])
            server.send_message(msg)
        print(f"Email sent to {config['recipient_email']} with {len(violation_paths)} attachment(s).")
    except Exception as e:
        print(f"Failed to send email: {e}")


def main(weights, camera_index, conf_thresh, cooldown_seconds, violations_dir, imgsz, email_config):
    model = YOLO(weights)
    names = model.names  # dict: {0: 'Helmet', 1: 'No Helmet', 2: 'Worker'} or similar
    print(f"Classes: {names}")

    violation_indices = build_violation_index_set(names)
    if not violation_indices:
        print("[WARN] Could not match any class name to a known violation label "
              f"(looked for {VIOLATION_CLASS_NAMES}). No violations will ever be flagged. "
              "Check VIOLATION_CLASS_NAMES against your model's actual class names above.")
    else:
        flagged = {names[i] for i in violation_indices}
        print(f"Violation classes detected in this model: {flagged} (indices: {violation_indices})")

    colors = build_color_map(names)

    violations_dir = Path(violations_dir)
    violations_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    last_violation_time = 0.0
    session_violations = []  # paths saved during this run, emailed at the end

    print("Starting live monitor. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera.")
                break

            results = model.predict(source=frame, conf=conf_thresh, imgsz=imgsz, verbose=False)[0]
            boxes = results.boxes

            violation_detected = any(int(b.cls) in violation_indices for b in boxes)

            annotated = draw_detections(frame.copy(), boxes, names, colors)

            now = time.time()
            time_since_last = now - last_violation_time

            if violation_detected and time_since_last >= cooldown_seconds:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = violations_dir / f"violation_{timestamp}.jpg"
                cv2.imwrite(str(save_path), annotated)
                session_violations.append(save_path)
                print(f"[VIOLATION] No-helmet detected — saved {save_path}")
                last_violation_time = now

            status_text = "VIOLATION DETECTED" if violation_detected else "OK"
            status_color = (0, 0, 255) if violation_detected else (0, 200, 0)
            cv2.putText(annotated, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, status_color, 2, cv2.LINE_AA)

            if time_since_last < cooldown_seconds:
                remaining = cooldown_seconds - time_since_last
                cv2.putText(annotated, f"Cooldown: {remaining:.1f}s",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("PPE Monitor — press 'q' to quit", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released.")

        print(f"\nSession ended. Total violations: {len(session_violations)}")
        send_violation_email(email_config, session_violations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live webcam PPE violation monitor with email alert")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Seconds between violation screenshots")
    parser.add_argument("--out", default="violations", help="Folder to save violation screenshots")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    args = parser.parse_args()

    main(
        weights=args.weights,
        camera_index=args.camera,
        conf_thresh=args.conf,
        cooldown_seconds=args.cooldown,
        violations_dir=args.out,
        imgsz=args.imgsz,
        email_config=EMAIL_CONFIG,
    )