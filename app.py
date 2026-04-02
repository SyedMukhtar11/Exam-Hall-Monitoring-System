"""
app.py
------
FastAPI backend.
Fixes applied:
  • _stream() now receives 0 (int) for webcam, not the string "webcam"
  • generate_stream uses consistent boundary string ("frame") everywhere
  • /webcam and /video return correct multipart media_type
  • upload saves file then immediately confirms; /video uses the saved path
  • All 6 return values from process_frame unpacked correctly
  • Incidents are built from already-resolved track data — no second
    call to _match_or_create_track (fixes ghost-track / name-mismatch bug)
  • /reset endpoint clears session state between tests
  • /export_attendance_csv exports student name + roll number
  • /export_csv (violations) now includes roll_no column
  • _ROLL_MAP includes Mukhtar (1604-24-733-011) and Azeem (1604-24-733-018)
    Student1-20 assigned unique random roll numbers from the same series
    (001-030, excluding 011 and 018)
"""

import cv2
import os
import csv
import io
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.utils.detector import process_frame, SUSPICIOUS_LABELS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER    = "uploads"
CHEAT_IMG_FOLDER = "backend/cheating_images"
os.makedirs(UPLOAD_FOLDER,    exist_ok=True)
os.makedirs(CHEAT_IMG_FOLDER, exist_ok=True)

app.mount("/backend", StaticFiles(directory="backend"), name="backend")

# ── Roll number map ───────────────────────────────────────────────────────────
# Mukhtar and Azeem have fixed roll numbers.
# Student1-20 are assigned unique random numbers from 1604-24-733-001 to -030,
# excluding 011 (Mukhtar) and 018 (Azeem).
# Seed=42 ensures consistent assignment across server restarts.
_ROLL_MAP: dict[str, str] = {
    "Mukhtar": "1604-24-733-011",
    "Azeem":   "1604-24-733-018",
    # Student1-20 random unique rolls (generated with seed=42)
    "Student1":  "1604-24-733-023",
    "Student2":  "1604-24-733-004",
    "Student3":  "1604-24-733-001",
    "Student4":  "1604-24-733-026",
    "Student5":  "1604-24-733-009",
    "Student6":  "1604-24-733-008",
    "Student7":  "1604-24-733-025",
    "Student8":  "1604-24-733-005",
    "Student9":  "1604-24-733-029",
    "Student10": "1604-24-733-020",
    "Student11": "1604-24-733-003",
    "Student12": "1604-24-733-015",
    "Student13": "1604-24-733-002",
    "Student14": "1604-24-733-028",
    "Student15": "1604-24-733-017",
    "Student16": "1604-24-733-022",
    "Student17": "1604-24-733-014",
    "Student18": "1604-24-733-027",
    "Student19": "1604-24-733-010",
    "Student20": "1604-24-733-016",
}

# ── Session state ────────────────────────────────────────────────────────────
last_uploaded_video:  str | None     = None
alert_status:         dict           = {"alert": False, "msg": "No cheating"}
attendance_list:      list[str]      = []
cheat_image_files:    list[str]      = []
violation_log:        list[dict]     = []
student_cheat_count:  dict[str, int] = {}


def _record_incident(incident: dict, frame):
    ts      = datetime.now().strftime("%H:%M:%S")
    student = incident["student"]
    student_cheat_count[student] = student_cheat_count.get(student, 0) + 1

    violation_log.append({
        "timestamp":    ts,
        "student":      student,
        "roll_no":      _ROLL_MAP.get(student, "N/A"),
        "action":       incident["action"],
        "from_student": incident.get("from_student") or "—",
        "severity":     incident["severity"],
        "count":        student_cheat_count[student],
    })

    if len(cheat_image_files) < 100:
        fname = f"cheat_{len(cheat_image_files):04d}.jpg"
        cv2.imwrite(os.path.join(CHEAT_IMG_FOLDER, fname), frame)
        cheat_image_files.append(fname)


# ── Core stream generator ────────────────────────────────────────────────────
def _stream(source):
    """
    Yields MJPEG frames.
    source: 0 (int) for webcam, or a file path string for uploaded video.
    Boundary string MUST match the media_type header — using 'frame'.
    """
    global alert_status

    if source == 0:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source!r}")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, _, labels, cheating, students, incidents = process_frame(frame)

            for s in students:
                if s not in attendance_list:
                    attendance_list.append(s)

            if cheating and incidents:
                detected     = list(set(labels) & set(SUSPICIOUS_LABELS))
                alert_status = {"alert": True, "msg": f"Detected: {', '.join(detected)}"}
                for inc in incidents:
                    _record_incident(inc, frame)
            else:
                alert_status = {"alert": False, "msg": "No cheating"}

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )
    finally:
        cap.release()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/webcam")
def webcam_feed():
    return StreamingResponse(
        _stream(0),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control":               "no-cache, no-store, must-revalidate",
            "Pragma":                      "no-cache",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    global last_uploaded_video
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    last_uploaded_video = path
    return {"status": "Uploaded", "filename": file.filename}


@app.get("/video")
def video_feed():
    if not last_uploaded_video:
        return JSONResponse({"error": "No video uploaded yet"}, status_code=400)
    return StreamingResponse(
        _stream(last_uploaded_video),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control":               "no-cache, no-store, must-revalidate",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/alert")
def get_alert():
    return alert_status


@app.get("/attendance")
def get_attendance():
    return attendance_list


@app.get("/cheat_images")
def get_cheat_images():
    return cheat_image_files


@app.get("/violations")
def get_violations():
    return violation_log


@app.get("/student_counts")
def get_student_counts():
    return student_cheat_count


@app.get("/export_csv")
def export_csv():
    """
    Export full violation log as CSV.
    Columns: timestamp, student, roll_no, action, from_student, severity, count
    """
    out = io.StringIO()
    w   = csv.DictWriter(
        out,
        fieldnames=[
            "timestamp", "student", "roll_no",
            "action", "from_student", "severity", "count"
        ]
    )
    w.writeheader()
    w.writerows(violation_log)
    out.seek(0)
    return StreamingResponse(
        iter([out.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=violations.csv"},
    )


@app.get("/export_attendance_csv")
def export_attendance_csv():
    """
    Export attendance list as CSV.
    Columns: Student Name, Roll Number
    Includes all students detected during the session.
    Mukhtar → 1604-24-733-011, Azeem → 1604-24-733-018,
    Student1-20 → unique random rolls from 001-030 (excl. 011, 018).
    """
    out = io.StringIO()
    w   = csv.writer(out)
    w.writerow(["Student Name", "Roll Number"])
    for student in attendance_list:
        roll = _ROLL_MAP.get(student, "N/A")
        w.writerow([student, roll])
    out.seek(0)
    return StreamingResponse(
        iter([out.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=attendance.csv"},
    )


@app.post("/reset")
def reset_session():
    """Clear all session data — call between exam sessions."""
    global alert_status, attendance_list, cheat_image_files, violation_log, student_cheat_count
    alert_status        = {"alert": False, "msg": "No cheating"}
    attendance_list     = []
    cheat_image_files   = []
    violation_log       = []
    student_cheat_count = {}
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)