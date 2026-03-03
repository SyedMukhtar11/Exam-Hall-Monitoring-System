from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import shutil
import os

from backend.utils.detector import generate_webcam_stream, generate_video_stream

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CHEATING_THRESHOLD = 3

# Global state
alert_status = {
    "alert": False,
    "count": 0
}


@app.get("/")
def home():
    return {"message": "Exam Monitoring Backend Running"}


# 🎥 LIVE WEBCAM STREAM
@app.get("/webcam")
def webcam_stream():
    return StreamingResponse(generate_webcam_stream(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


# 📂 UPLOAD VIDEO
@app.post("/upload")
def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"video_path": file.filename}


# 🎥 STREAM UPLOADED VIDEO
@app.get("/video/{filename}")
def video_stream(filename: str):
    video_path = os.path.join(UPLOAD_FOLDER, filename)

    return StreamingResponse(generate_video_stream(video_path),
                             media_type="multipart/x-mixed-replace; boundary=frame")


# 🚨 ALERT STATUS API
@app.get("/alert")
def get_alert():
    return alert_status