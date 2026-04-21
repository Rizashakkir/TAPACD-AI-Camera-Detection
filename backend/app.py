from flask import Flask, render_template, Response, send_from_directory
import cv2
from ultralytics import YOLO
import time
import os
import winsound
import csv
from datetime import datetime

app = Flask(__name__)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO("runs/detect/train2/weights/best.pt")

# -----------------------------
# WEBCAM
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# -----------------------------
# VARIABLES
# -----------------------------
phone_detect_start = None
alert_triggered = False

CAPTURE_DIR = "captures"
CSV_FILE = "alerts_log.csv"

# -----------------------------
# CREATE FOLDERS & CSV IF NOT EXISTS
# -----------------------------
os.makedirs(CAPTURE_DIR, exist_ok=True)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Time", "Confidence", "Image"])

# -----------------------------
# FRAME GENERATOR
# -----------------------------
def generate_frames():
    global phone_detect_start, alert_triggered

    while True:
        success, frame = cap.read()
        if not success:
            break

        # YOLO DETECTION
        results = model(frame, conf=0.25)

        # DRAW BOUNDING BOX + CONFIDENCE
        annotated = results[0].plot(labels=True, conf=True)

        phone_detected = False
        confidence = 0.0

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if "phone" in model.names[cls].lower():
                    phone_detected = True
                    confidence = conf
                    break

        if phone_detected:
            if phone_detect_start is None:
                phone_detect_start = time.time()

            elapsed = time.time() - phone_detect_start

            # DISPLAY TIMER + CONFIDENCE ON FRAME
            cv2.putText(
                annotated,
                f"PHONE HELD: {int(elapsed)}s | CONF: {confidence:.2f}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            if elapsed >= 7 and not alert_triggered:
                alert_triggered = True

                timestamp = datetime.now()
                filename = os.path.join(
                    CAPTURE_DIR,
                    f"ALERT_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                )

                # ✅ SAVE ANNOTATED IMAGE (IMPORTANT)
                cv2.imwrite(filename, annotated)

                # CSV LOG
                with open(CSV_FILE, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp.strftime("%Y-%m-%d"),
                        timestamp.strftime("%H:%M:%S"),
                        round(confidence, 2),
                        filename
                    ])

                print("Saved:", filename)

                # BEEP ALERT
                winsound.Beep(1500, 1200)

        else:
            phone_detect_start = None
            alert_triggered = False

        # STREAM VIDEO
        ret, buffer = cv2.imencode(".jpg", annotated)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def index():
    images = sorted(os.listdir(CAPTURE_DIR), reverse=True)
    return render_template("index.html", images=images)

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/captures/<filename>")
def get_capture(filename):
    return send_from_directory(CAPTURE_DIR, filename)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)