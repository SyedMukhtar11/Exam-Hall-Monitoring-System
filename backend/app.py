from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import shutil
import os

from backend.utils.detector import process_frame
from fastapi.middleware.cors import CORSMiddleware

import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CHEATING_THRESHOLD = 3

# GLOBAL STATE
alert_status = {
    "alert": False,
    "count": 0
}

# 🎥 PROCESS VIDEO STREAM
def generate_stream(source):

    if source == "webcam":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    suspicious_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame, suspicious, labels = process_frame(frame)

        if suspicious > 0:
            suspicious_count += 1

        # 🚨 ALERT LOGIC
        if suspicious_count > CHEATING_THRESHOLD:
            alert_status["alert"] = True
            alert_status["count"] = suspicious_count
        else:
            alert_status["alert"] = False

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


# 🎥 LIVE WEBCAM
@app.get("/webcam")
def webcam():
    return StreamingResponse(generate_stream("webcam"),
        media_type="multipart/x-mixed-replace; boundary=frame")


from fastapi.responses import StreamingResponse
import os



UPLOAD_FOLDER = "backend/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

last_uploaded_video = None


@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    global last_uploaded_video

    path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(path, "wb") as buffer:
        buffer.write(await file.read())

    last_uploaded_video = path

    return {"message": "Uploaded"}


# 🎥 PLAY UPLOADED VIDEO WITH DETECTION
@app.get("/video")
def video():
    if last_uploaded_video is None:
        return {"error": "No video uploaded"}

    return StreamingResponse(
        generate_stream(last_uploaded_video),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# 🚨 ALERT API

@app.get("/alert")
def get_alert():
    return alert_status