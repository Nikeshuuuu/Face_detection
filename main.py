import cv2
import torch
import time
from ultralytics import YOLO

# --------------------------------------------------
# Device
# --------------------------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on: {device}")

# --------------------------------------------------
# Load Models
# --------------------------------------------------
object_model = YOLO("yolo11l.pt")
face_model = YOLO("yolo11l-face.pt")

# --------------------------------------------------
# Webcam
# --------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------------------------
    # Object Detection + Tracking
    # -----------------------------------------------
    object_results = object_model.track(
        frame,
        device=device,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.45,
        verbose=False,
    )

    # -----------------------------------------------
    # Face Detection
    # -----------------------------------------------
    face_results = face_model(
        frame,
        device=device,
        conf=0.5,
        verbose=False,
    )

    output = frame.copy()
    person_count = 0
    face_count = 0

    # -----------------------------------------------
    # Draw Objects
    # -----------------------------------------------
    if object_results[0].boxes is not None:
        for box in object_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = object_model.names[cls]

            if label == "person":
                person_count += 1

            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                output,
                f"{label}  ID:{track_id}  {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

    # -----------------------------------------------
    # Draw Faces
    # -----------------------------------------------
    if face_results[0].boxes is not None:
        for box in face_results[0].boxes:
            face_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                output,
                f"Face {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2,
            )

    # -----------------------------------------------
    # FPS
    # -----------------------------------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        output,
        f"FPS : {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        output,
        f"Persons : {person_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        output,
        f"Faces : {face_count}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
    )

    cv2.imshow("YOLOv8 Object + Face Detection", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("x"):
        break

cap.release()
cv2.destroyAllWindows()
