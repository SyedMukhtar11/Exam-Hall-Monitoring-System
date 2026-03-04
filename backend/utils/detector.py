import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("backend/model/best.pt")

import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

def process_frame(frame):
    results = model(frame, conf=0.25)

    suspicious = 0
    labels_detected = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            #Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            labels_detected.append(class_name)

            # 🚨 CHEATING LOGIC
            if class_name.lower() in ["cheating", "phone", "talking", "looking_left", "looking_right"]:
             
                suspicious += 1
    print("Detected:", labels_detected)
    return frame, suspicious, labels_detected


def generate_webcam_stream():
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    suspicious_count = 0

    while True:
        success, frame = cap.read()
        frame=cv2.flip(frame,1)
        if not success:
            break

        frame, suspicious, labels = process_frame(frame)

        if suspicious > 0:
            suspicious_count += 1

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


def generate_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)

    suspicious_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame, suspicious, labels = process_frame(frame)

        if suspicious > 0:
            suspicious_count += 1

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()