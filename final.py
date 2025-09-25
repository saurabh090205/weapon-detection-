
import os
import time
import threading
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from keras.models import load_model
import smtplib
from email.message import EmailMessage

# ----------------------
# Config
# ----------------------
YOLO_MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.35
MAX_WIDTH = 1280
SAVE_OUTPUT = False
OUTPUT_PATH = "output/output.mp4"
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

# Email config
FROM_EMAIL = "ethangade96@gmail.com"
FROM_PASSWORD = "lzoe yzqu urrl rsdk"
TO_EMAIL = "saurabh.gangurde24@vit.edu"

# Screenshot directory
SCREENSHOT_DIR = "weapon_screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Violence detection
CLASSES_LIST = ["non_violence", "violence"]
VIOLENCE_DETECTION_FRAMES_SKIP = 5

# ----------------------
# Shared state and locks
# ----------------------
alert_lock = threading.Lock()
last_violence_time = 0.0
violence_detected_flag = False
weapon_email_sent = False  # For email sent flag

def send_email_alert(subject, body, to_email, from_email, from_password):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, from_password)
        server.send_message(msg)
        server.quit()
        print("[EMAIL] Alert email sent successfully.")
    except Exception as e:
        print(f"[EMAIL][ERR] Failed to send email: {e}")

def set_violence_alert():
    global last_violence_time, violence_detected_flag
    with alert_lock:
        last_violence_time = time.time()
        violence_detected_flag = True

def violence_alert_active():
    with alert_lock:
        return violence_detected_flag and ((time.time() - last_violence_time) <= 3.0)

def draw_boxes(frame, boxes, confidences, classes, names):
    for (x1, y1, x2, y2), conf, cls in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{names.get(int(cls), str(cls))}: {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

def load_violence_model(path):
    try:
        return load_model(path)
    except Exception as e:
        print(f"[VIOLENCE][ERR] Failed to load violence model: {e}")
        return None

def violence_detect(frame, model):
    try:
        img = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        img = img / 960.0
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)
        class_index = np.argmax(pred)
        confidence = pred[0][class_index]
        if class_index == 1 and confidence > 0.5:
            print(f"[VIOLENCE] Suspected violence detected with confidence {confidence:.2f}")
            set_violence_alert()
            return True
        return False
    except Exception as e:
        print(f"[VIOLENCE][ERR] Inference error: {e}")
        return False

def video_loop(args):
    global weapon_email_sent

    src = args.source
    try:
        source = int(src)
    except Exception:
        source = src

    yolo_model = YOLO(args.model)
    violence_model = load_violence_model(args.violence_model)

    if torch.cuda.is_available():
        try:
            yolo_model.to("cuda")
            print("[VIDEO] Using GPU (CUDA) for YOLO")
        except Exception:
            print("[VIDEO][WARN] Could not move YOLO model to CUDA; using CPU")
    else:
        print("[VIDEO] Using CPU for YOLO")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: Unable to open video source: {src}")

    writer = None
    if args.save:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
        print(f"[VIDEO] Saving to {args.out}")

    frame_count = 0
    print("[VIDEO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[VIDEO] End of stream.")
            break

        frame_count += 1

        if args.width and frame.shape[1] > args.width:
            scale = args.width / float(frame.shape[1])
            frame = cv2.resize(frame, (args.width, int(frame.shape[0] * scale)))

        results = yolo_model(frame, imgsz=args.imgsz)
        r = results[0]

        boxes, confs, clss = [], [], []
        for box, conf, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            class_name = yolo_model.names[int(cls)].lower()
            # Show boxes >=35%
            if conf >= 0.35 and ("gun" in class_name or "knife" in class_name or "weapon" in class_name):
                boxes.append(box)
                confs.append(float(conf))
                clss.append(int(cls))
            # Send email once for first >=90% & save screenshot each time >=90%
            if conf >= 0.9 and ("gun" in class_name or "knife" in class_name or "weapon" in class_name):
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                screenshot_path = os.path.join(SCREENSHOT_DIR, f"weapon_{timestamp}.jpg")
                cv2.imwrite(screenshot_path, frame)
                print(f"[INFO] Screenshot saved to {screenshot_path}")

                if not weapon_email_sent:
                    print("[ALERT] Weapon detected with confidence >= 90%. Sending email alert.")
                    send_email_alert(
                        "Weapon Detected!",
                        f"A weapon ({class_name}) with confidence {conf:.2f} has been detected.",
                        TO_EMAIL,
                        FROM_EMAIL,
                        FROM_PASSWORD
                    )
                    weapon_email_sent = True

        draw_boxes(frame, boxes, confs, clss, yolo_model.names)

        if violence_model and (frame_count % VIOLENCE_DETECTION_FRAMES_SKIP) == 0:
            violence_detect(frame, violence_model)

        if violence_alert_active():
            banner = "Suspected Violence Detected"
            frame_height = frame.shape[0]
            (w, h), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            y_pos = frame_height - 30
            cv2.rectangle(frame, (10, y_pos), (10 + w + 10, y_pos + h + 10), (0, 0, 255), -1)
            cv2.putText(frame, banner, (20, y_pos + h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        fps = frame_count / max(1e-3, (time.time() - args.start_time))
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Weapon + Violence Detection", frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", default=YOLO_MODEL_PATH, help="YOLO weapon detection model path (.pt)")
    p.add_argument("--source", "-s", default="0", help="Video source (default=0 for webcam)")
    p.add_argument("--conf", "-c", type=float, default=CONF_THRESHOLD, help="Confidence threshold for YOLO")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size for YOLO")
    p.add_argument("--width", type=int, default=MAX_WIDTH, help="Max video width to resize")
    p.add_argument("--save", action="store_true", help="Save output video")
    p.add_argument("--out", type=str, default=OUTPUT_PATH, help="Output video path")
    p.add_argument("--violence_model", type=str, default="MobileNetV2_CNN_model.h5", help="Violence detection model (.h5)")
    return p.parse_args()

def main():
    global weapon_email_sent
    weapon_email_sent = False
    args = parse_args()
    args.start_time = time.time()
    video_loop(args)

if __name__ == "__main__":
    main()
