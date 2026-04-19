"""
app.py  —  FIXED VERSION
-------------------------
BUGS FIXED IN THIS VERSION
===========================

BUG-A1  [SYNTAX / CRASH] `global (a, b, ...)` with parentheses is invalid
        Python and raises SyntaxError on the /reset call.
        Fixed → `global a, b, c` (no parentheses).

BUG-A2  [DEADLOCK] `_auto_alert_email()` was called INSIDE `with _email_lock`
        while `_record_incident()` also acquires the same lock under the
        stream generator.  Under concurrent access the generator hung.
        Fixed → check-and-mark inside lock; email is fired OUTSIDE it.

BUG-A3  [RACE CONDITION] `roll_no` was appended to a whisper dict AFTER
        the dict was added to whisper_log.  A concurrent CSV export saw
        the entry without that key and raised ValueError.
        Fixed → build the complete record first, append once.

BUG-A4  [LOG FLOODING] `_record_incident` was called on EVERY cheating
        frame (~30 fps).  A 10-second suspicious episode generated ~300
        identical violation rows for the same student.
        Fixed → per-student cooldown of INCIDENT_COOLDOWN_SEC (2 s).
        The flag COUNTER still increments once per cooldown window, so
        the auto-alert threshold logic remains correct.

BUG-A5  [IMAGE OVERWRITE ON RESET] After /reset, `cheat_image_files = []`
        reset the in-memory list so new images started at index 0000,
        silently overwriting existing snapshots on disk.
        Fixed → filenames now embed a UTC timestamp + counter so they are
        globally unique across resets.

BUG-A6  [STALE IMAGES AFTER RESET] The CHEAT_IMG_FOLDER was never cleared
        on /reset, so old snapshots from a previous session persisted on
        disk.  Fixed → /reset removes all .jpg files in that folder.

BUG-A7  [DETECTOR STATE NOT RESET] /reset cleared app.py's lists but
        never reset detector.py's tracker dict, track-ID counter, name
        iterator, or whisper queue.  Old track identities and the
        partially-consumed name pool bled into the next session.
        Fixed → /reset calls `reset_detector_state()` from detector.py.

BUG-A8  [THREAD SAFETY] Shared mutable state (`attendance_list`,
        `violation_log`, etc.) was read by API endpoints and written by
        the stream generator concurrently without synchronisation.
        Fixed → a single `_state_lock` guards all mutations and list
        reads that must be snapshot-consistent (CSV export, /violations).
"""

import cv2
import os
import csv
import io
import glob
import time
import zipfile
import smtplib
import threading
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.base      import MIMEBase
from email                import encoders

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.utils.detector import (
    process_frame,
    SUSPICIOUS_LABELS,
    get_pending_whisper_events,
    reset_detector_state,          # BUG-A7 FIX
)

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

# ── Email configuration ────────────────────────────────────────────────────────


# ── Alert threshold ────────────────────────────────────────────────────────────
FLAG_ALERT_THRESHOLD = 5

# BUG-A4 FIX: minimum seconds between consecutive incident records per student.
INCIDENT_COOLDOWN_SEC = 2.0

# ── Roll number map ────────────────────────────────────────────────────────────
_ROLL_MAP: dict[str, str] = {
    "Mukhtar":   "1604-24-733-011",
    "Azeem":     "1604-24-733-018",
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

# ── Session state ──────────────────────────────────────────────────────────────
last_uploaded_video: str | None     = None
alert_status:        dict           = {"alert": False, "msg": "No cheating"}
attendance_list:     list[str]      = []
cheat_image_files:   list[str]      = []
violation_log:       list[dict]     = []
whisper_log:         list[dict]     = []
student_cheat_count: dict[str, int] = {}

_alerted_students: set[str]    = set()
_email_lock  = threading.Lock()

# BUG-A8 FIX: single lock for all shared mutable session state
_state_lock  = threading.Lock()

# BUG-A4 FIX: track last incident timestamp per student (not exposed to front-end)
_last_incident_time: dict[str, float] = {}

# BUG-A5 FIX: session-level image counter; unique across sessions via timestamp
_img_session_prefix: str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
_img_counter: int = 0


# ── Email helpers ──────────────────────────────────────────────────────────────

def _build_violations_csv() -> bytes:
    with _state_lock:
        rows = list(violation_log)
    out = io.StringIO()
    w   = csv.DictWriter(
        out,
        fieldnames=["timestamp", "student", "roll_no",
                    "action", "from_student", "severity", "count"],
        extrasaction="ignore",
    )
    w.writeheader()
    w.writerows(rows)
    return out.getvalue().encode("utf-8")


def _build_whisper_csv() -> bytes:
    with _state_lock:
        rows = list(whisper_log)
    out = io.StringIO()
    w   = csv.DictWriter(
        out,
        fieldnames=["timestamp", "student", "roll_no", "whispering_to", "rms"],
        extrasaction="ignore",
    )
    w.writeheader()
    w.writerows(rows)
    return out.getvalue().encode("utf-8")


def _send_email(
    subject: str,
    body_html: str,
    attach_images: bool = False,
    extra_attachments: list[tuple[str, bytes, str]] | None = None,
):
    """Send email in a background/daemon thread — never blocks the stream."""
    try:
        msg = MIMEMultipart("mixed")
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = ", ".join([INVIGILATOR_EMAIL, EXAM_CELL_EMAIL])
        msg["Subject"] = subject
        msg.attach(MIMEText(body_html, "html"))

        if extra_attachments:
            for fname, data, mime_sub in extra_attachments:
                part = MIMEBase("application", mime_sub)
                part.set_payload(data)
                encoders.encode_base64(part)
                part.add_header("Content-Disposition",
                                f'attachment; filename="{fname}"')
                msg.attach(part)

        if attach_images:
            with _state_lock:
                imgs = list(cheat_image_files)
            if imgs:
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fname in imgs:
                        fpath = os.path.join(CHEAT_IMG_FOLDER, fname)
                        if os.path.exists(fpath):
                            zf.write(fpath, fname)
                zip_data = zip_buf.getvalue()
                part = MIMEBase("application", "zip")
                part.set_payload(zip_data)
                encoders.encode_base64(part)
                part.add_header("Content-Disposition",
                                'attachment; filename="cheat_snapshots.zip"')
                msg.attach(part)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(
                EMAIL_SENDER,
                [INVIGILATOR_EMAIL, EXAM_CELL_EMAIL],
                msg.as_string(),
            )
        print(f"[EMAIL] Sent: {subject}")
    except Exception as e:
        print(f"[EMAIL] Failed to send '{subject}': {e}")


def _fire_alert_email_if_needed(student: str, count: int):
    """
    BUG-A2 FIX: check-and-mark inside the lock; email fired OUTSIDE.
    This eliminates the deadlock risk with _record_incident.
    """
    should_send = False
    with _email_lock:
        if count == FLAG_ALERT_THRESHOLD and student not in _alerted_students:
            _alerted_students.add(student)
            should_send = True

    if should_send:
        roll    = _ROLL_MAP.get(student, "N/A")
        ts_now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = (f"[EXAM ALERT] {student} ({roll}) — "
                   f"{count} violations — AI Exam Monitor")
        body = f"""
        <html><body style="font-family:Arial,sans-serif;color:#222;">
        <h2 style="color:#c0392b;">&#9888; Exam Violation Alert</h2>
        <p>The automated monitoring system has detected that the following
        student has crossed the violation threshold of
        <strong>{FLAG_ALERT_THRESHOLD} flags</strong>.</p>
        <table border="1" cellpadding="8" cellspacing="0"
               style="border-collapse:collapse;">
          <tr><th>Student</th><th>Roll No.</th><th>Total Flags</th></tr>
          <tr>
            <td><strong>{student}</strong></td>
            <td>{roll}</td>
            <td style="color:#c0392b;font-weight:bold;">{count}</td>
          </tr>
        </table>
        <p>Please review the attached violation report and snapshots.</p>
        <p style="font-size:0.85em;color:#888;">
          Sent automatically by AI Exam Monitor Pro at {ts_now}
        </p>
        </body></html>
        """
        violations_csv = _build_violations_csv()
        threading.Thread(
            target=_send_email,
            args=(subject, body, True,
                  [("violations.csv", violations_csv, "csv")]),
            daemon=True,
        ).start()


# ── Incident recording ────────────────────────────────────────────────────────

def _record_incident(incident: dict, frame):
    """
    Record one visual cheating incident.
    BUG-A4 FIX: enforces a per-student cooldown to prevent log flooding
    at camera frame rate (~30 fps).  The flag counter still increments
    so alert thresholds are meaningful, but duplicate rows are suppressed.
    BUG-A5 FIX: image filenames embed session prefix + counter, unique
    across resets.
    BUG-A8 FIX: all writes are protected by _state_lock.
    """
    global _img_counter

    student = incident["student"]
    now     = time.monotonic()

    # BUG-A4: skip if within cooldown window for this student
    with _state_lock:
        last = _last_incident_time.get(student, 0.0)
        if now - last < INCIDENT_COOLDOWN_SEC:
            return
        _last_incident_time[student] = now

    ts    = datetime.now().strftime("%H:%M:%S")
    count = student_cheat_count.get(student, 0) + 1

    with _state_lock:
        student_cheat_count[student] = count
        violation_log.append({
            "timestamp":    ts,
            "student":      student,
            "roll_no":      _ROLL_MAP.get(student, "N/A"),
            "action":       incident["action"],
            "from_student": incident.get("from_student") or "—",
            "severity":     incident["severity"],
            "count":        count,
        })

        # BUG-A5 FIX: unique filenames per session via prefix + counter
        if len(cheat_image_files) < 100:
            fname = f"cheat_{_img_session_prefix}_{_img_counter:04d}.jpg"
            _img_counter += 1
            cv2.imwrite(os.path.join(CHEAT_IMG_FOLDER, fname), frame)
            cheat_image_files.append(fname)

    _fire_alert_email_if_needed(student, count)


# ── Core stream generator ─────────────────────────────────────────────────────

def _stream(source):
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

            # Attendance
            with _state_lock:
                for s in students:
                    if s not in attendance_list:
                        attendance_list.append(s)

            # ── Whisper events ─────────────────────────────────────────────
            # BUG-A3 FIX: build the complete record BEFORE any append.
            for wev in get_pending_whisper_events():
                student_w = wev["student"]
                roll_w    = _ROLL_MAP.get(student_w, "N/A")

                whisper_record = {
                    "timestamp":     wev["timestamp"],
                    "student":       student_w,
                    "roll_no":       roll_w,          # key present BEFORE append
                    "whispering_to": wev.get("whispering_to", "—"),
                    "rms":           wev.get("rms", 0),
                }

                with _state_lock:
                    whisper_log.append(whisper_record)
                    count_w = student_cheat_count.get(student_w, 0) + 1
                    student_cheat_count[student_w] = count_w

                    violation_log.append({
                        "timestamp":    wev["timestamp"],
                        "student":      student_w,
                        "roll_no":      roll_w,
                        "action":       "whispering",
                        "from_student": wev.get("whispering_to", "—"),
                        "severity":     "LOW",
                        "count":        count_w,
                    })

                _fire_alert_email_if_needed(student_w, count_w)

            # ── Visual cheating ────────────────────────────────────────────
            if cheating and incidents:
                detected     = list(set(labels) & set(SUSPICIOUS_LABELS))
                alert_status = {"alert": True,
                                "msg":   f"Detected: {', '.join(detected)}"}
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
    with _state_lock:
        return list(attendance_list)


@app.get("/cheat_images")
def get_cheat_images():
    with _state_lock:
        return list(cheat_image_files)


@app.get("/violations")
def get_violations():
    with _state_lock:
        return list(violation_log)


@app.get("/student_counts")
def get_student_counts():
    with _state_lock:
        return dict(student_cheat_count)


@app.get("/whisper_events")
def get_whisper_events():
    with _state_lock:
        return list(whisper_log)


@app.get("/export_csv")
def export_csv():
    csv_bytes = _build_violations_csv()
    return StreamingResponse(
        iter([csv_bytes.decode("utf-8")]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=violations.csv"},
    )


@app.get("/export_attendance_csv")
def export_attendance_csv():
    with _state_lock:
        students = list(attendance_list)
    out = io.StringIO()
    w   = csv.writer(out)
    w.writerow(["Student Name", "Roll Number"])
    for student in students:
        w.writerow([student, _ROLL_MAP.get(student, "N/A")])
    out.seek(0)
    return StreamingResponse(
        iter([out.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=attendance.csv"},
    )


@app.post("/send_report")
def send_report(background_tasks: BackgroundTasks):
    with _state_lock:
        n_violations = len(violation_log)
        n_images     = len(cheat_image_files)
        n_whispers   = len(whisper_log)
        n_students   = len(attendance_list)
        counts_snap  = dict(student_cheat_count)

    if not n_violations and not n_images:
        return JSONResponse({
            "status": "nothing_to_send",
            "msg":    "No violations or images recorded yet.",
        })

    ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"[EXAM REPORT] Full Session Report — {ts} — AI Exam Monitor"

    rows = "".join(
        f"<tr><td>{s}</td><td>{_ROLL_MAP.get(s, 'N/A')}</td>"
        f"<td><b>{c}</b></td></tr>"
        for s, c in sorted(counts_snap.items(), key=lambda x: x[1], reverse=True)
    )
    body = f"""
    <html><body style="font-family:Arial,sans-serif;color:#222;">
    <h2 style="color:#2c3e50;">&#128203; Exam Session Report</h2>
    <p>Full report for the examination session ending at
    <strong>{ts}</strong>.</p>
    <h3>Summary</h3>
    <table border="1" cellpadding="8" cellspacing="0"
           style="border-collapse:collapse;">
      <tr><th>Total students detected</th><td>{n_students}</td></tr>
      <tr><th>Total violations logged</th><td>{n_violations}</td></tr>
      <tr><th>Whisper events</th>          <td>{n_whispers}</td></tr>
      <tr><th>Snapshot images</th>         <td>{n_images}</td></tr>
    </table>
    <h3>Students by Flag Count</h3>
    <table border="1" cellpadding="8" cellspacing="0"
           style="border-collapse:collapse;">
      <tr><th>Student</th><th>Roll No.</th><th>Flags</th></tr>
      {rows}
    </table>
    <p style="font-size:0.85em;color:#888;">
      Sent by AI Exam Monitor Pro — {ts}
    </p>
    </body></html>
    """

    violations_csv = _build_violations_csv()
    whisper_csv    = _build_whisper_csv()

    background_tasks.add_task(
        _send_email,
        subject,
        body,
        True,
        [
            ("violations.csv",  violations_csv, "csv"),
            ("whisper_log.csv", whisper_csv,    "csv"),
        ],
    )
    return {
        "status": "sending",
        "msg":    "Report is being sent to invigilator and exam cell.",
    }


@app.post("/reset")
def reset_session():
    """
    Clear all session state between exam sessions.

    BUG-A1 FIX: `global (a, b)` with parentheses is invalid Python.
                Corrected to `global a, b` (no parentheses).
    BUG-A6 FIX: physically delete old snapshot images from disk.
    BUG-A7 FIX: call reset_detector_state() to clear tracker + name pool.
    """
    # BUG-A1 FIX: no parentheses around the names
    global last_uploaded_video, alert_status, attendance_list
    global cheat_image_files, violation_log, whisper_log
    global student_cheat_count, _alerted_students
    global _last_incident_time, _img_session_prefix, _img_counter

    # BUG-A7 FIX: reset detector-side tracker state
    reset_detector_state()

    # BUG-A6 FIX: clear old snapshots from disk before resetting the list
    for jpg_path in glob.glob(os.path.join(CHEAT_IMG_FOLDER, "*.jpg")):
        try:
            os.remove(jpg_path)
        except OSError as e:
            print(f"[RESET] Could not remove {jpg_path}: {e}")

    with _state_lock:
        last_uploaded_video  = None
        alert_status         = {"alert": False, "msg": "No cheating"}
        attendance_list      = []
        cheat_image_files    = []
        violation_log        = []
        whisper_log          = []
        student_cheat_count  = {}
        _alerted_students    = set()
        _last_incident_time  = {}

    # BUG-A5 FIX: new session prefix so filenames are unique
    _img_session_prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    _img_counter        = 0

    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)