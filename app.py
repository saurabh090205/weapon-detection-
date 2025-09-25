import os
import time
import threading
from flask import Flask, render_template, Response, request, session, redirect, url_for, jsonify
import cv2
import torch
from ultralytics import YOLO
from keras.models import load_model
import smtplib
from email.message import EmailMessage
import numpy as np

app = Flask(__name__)
app.secret_key = '348056b0d47a7f909e4fff7f58846e3d02035bf36cd470f161416ff1a3829dad'

# Config
YOLO_MODEL_PATH = "best.pt"
VIOLENCE_MODEL_PATH = "MobileNetV2_CNN_model.h5"
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
CONF_THRESHOLD = 0.35
MAX_WIDTH = 1280

FROM_EMAIL = "ethangade96@gmail.com"
FROM_PASSWORD = "lzoe yzqu urrl rsdk"
SCREENSHOT_DIR = "static/screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Shared state
weapon_email_sent = False
alert_lock = threading.Lock()
last_violence_time = 0.0
violence_detected_flag = False

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
        print(f"[EMAIL] Alert email sent successfully to {to_email}.")
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

@app.route('/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        if email:
            session['user_email'] = email
            return redirect(url_for('webcam'))
    return render_template('register.html')

@app.route('/webcam')
def webcam():
    if 'user_email' not in session:
        return redirect(url_for('register'))
    return render_template('webcam.html')

def gen_frames():
    global weapon_email_sent
    weapon_email_sent = False

    yolo_model = YOLO(YOLO_MODEL_PATH)
    violence_model = load_violence_model(VIOLENCE_MODEL_PATH)
    if torch.cuda.is_available():
        yolo_model.to("cuda")
        print("[VIDEO] Using GPU for YOLO")
    else:
        print("[VIDEO] Using CPU for YOLO")

    cap = cv2.VideoCapture(0)
    frame_count = 0

    user_email = session.get('user_email', None)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame.shape[1] > MAX_WIDTH:
            scale = MAX_WIDTH / float(frame.shape[1])
            frame = cv2.resize(frame, (MAX_WIDTH, int(frame.shape[0] * scale)))

        results = yolo_model(frame, imgsz=640)
        r = results[0]

        boxes, confs, clss = [], [], []
        for box, conf, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                  r.boxes.conf.cpu().numpy(),
                                  r.boxes.cls.cpu().numpy()):
            class_name = yolo_model.names[int(cls)].lower()
            if conf >= 0.35 and ("gun" in class_name or "knife" in class_name or "weapon" in class_name):
                boxes.append(box)
                confs.append(float(conf))
                clss.append(int(cls))

            if conf >= 0.9 and ("gun" in class_name or "knife" in class_name or "weapon" in class_name):
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                screenshot_path = os.path.join('static/screenshots', f"weapon_{timestamp}.jpg")
                cv2.imwrite(screenshot_path, frame)
                print(f"[INFO] Screenshot saved to {screenshot_path}")
                if not weapon_email_sent and user_email:
                    print("[ALERT] Weapon detected with confidence >= 90%. Sending email alert.")
                    send_email_alert(
                        "Weapon Detected!",
                        f"A weapon ({class_name}) with confidence {conf:.2f} has been detected.",
                        user_email,
                        FROM_EMAIL,
                        FROM_PASSWORD
                    )
                    weapon_email_sent = True

        draw_boxes(frame, boxes, confs, clss, yolo_model.names)

        if violence_model and (frame_count % VIOLENCE_DETECTION_FRAMES_SKIP) == 0:
            violence_detect(frame, violence_model)

        if violence_alert_active():
            banner = "Suspected Violence Detected"
            (w, h), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            y_pos = frame.shape[0] - 30
            cv2.rectangle(frame, (10, y_pos), (10 + w + 10, y_pos + h + 10), (0,0,255), -1)
            cv2.putText(frame, banner, (20, y_pos + h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    if 'user_email' not in session:
        return redirect(url_for('register'))
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/screenshots')
def screenshots():
    files = os.listdir('static/screenshots')
    files = sorted(files, reverse=True)
    files = [url_for('static', filename='screenshots/' + f) for f in files]
    return jsonify(files)


if __name__ == "__main__":
    app.run(debug=True)
