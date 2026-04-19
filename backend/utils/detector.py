"""
detector.py  —  FIXED VERSION
------------------------------
BUGS FIXED IN THIS VERSION
===========================

BUG-D1  [LOGIC ERROR] Cheating threshold used `>` instead of `>=`.
        `suspicious_count > CHEAT_COUNT_THRESHOLD` with threshold=1 required
        2+ simultaneous suspicious detections. A single suspicious student
        was silently ignored. Fixed → `>= CHEAT_COUNT_THRESHOLD`.

BUG-D2  [RACE CONDITION] `get_pending_whisper_events()` called
        `list(_whisper_queue)` then `_whisper_queue.clear()`.
        Events appended by the audio thread between those two lines were
        silently lost. Fixed → drain with `popleft()` loop (each pop is
        atomic under the GIL).

BUG-D3  [SESSION BLEED] No reset function existed. After app.py called
        /reset, the module-level tracker dict `_tracks`, `_next_track_id`,
        `_name_iter`, `_last_whisper_time`, and `_whisper_queue` kept their
        old values, leaking old track identities into the new session.
        Fixed → `reset_detector_state()` reinstates all to initial values.

BUG-D4  [ZERO-DIVISION] `_iou()`: when both boxes are degenerate (zero
        area — e.g. a model returns a single-pixel box), the union becomes 0
        and Python raises ZeroDivisionError, crashing the stream generator.
        Fixed → guard `union = area_a + area_b - inter; return inter/union
        if union else 0.0`.

BUG-D5  [NAME POOL EXHAUSTION] After all 20 pool names are consumed,
        `next(_name_iter, f"Student_{tid+1}")` returns fallback IDs that
        are not present in app.py's `_ROLL_MAP`, so those students always
        show "N/A" and cannot be emailed. Fixed → the pool now cycles via
        `itertools.cycle` so names are reused rather than producing
        unmapped fallbacks.  `reset_detector_state()` also resets the
        iterator so the full pool is available again each session.

DUAL-MODEL ARCHITECTURE
========================
Model 1  (behaviour_model) — best1.pt
    Detects behaviour classes: back_watching, side_watching, suspicious,
    normal, front_watching, invigilator.

Model 2  (face_model) — best2.pt
    Detects named faces: "Mukhtar", "Azeem".
    Runs BEFORE behaviour model. Real names override Student pool when
    face box overlaps behaviour box.

AUDIO WHISPER DETECTION
========================
A background thread continuously reads from the default microphone.
Each 0.5-second chunk is analysed:
  • RMS < WHISPER_FLOOR  → silence / background noise  → ignored
  • WHISPER_FLOOR ≤ RMS < WHISPER_CEIL → WHISPER detected
  • RMS ≥ WHISPER_CEIL   → normal speech / loud event  → ignored

STUDENT ATTRIBUTION (non-random, precision-first):
  Priority 1 — Exactly one flagged student in frame → that student.
  Priority 2 — Closest student to WHISPER_ZONE within radius.
  Priority 3 — "Unknown" — no false accusation.

CHEATING RULE (fixed)
=====================
suspicious_count >= CHEAT_COUNT_THRESHOLD (1) → cheating = True
"""

import cv2
import itertools
import random
import threading
import time
import collections
from ultralytics import YOLO

# ── Optional audio import ──────────────────────────────────────────────────
try:
    import pyaudio
    import numpy as np
    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False
    print("[WARN] pyaudio/numpy not installed — whisper detection disabled.")

# ── Model paths ────────────────────────────────────────────────────────────
_BEHAVIOUR_MODEL_PATH = r"C:\Users\syedm\OneDrive\Desktop\TechNova1\best1.pt"
_FACE_MODEL_PATH      = r"C:\Users\syedm\OneDrive\Desktop\TechNova1\best2.pt"

behaviour_model = YOLO(_BEHAVIOUR_MODEL_PATH)
face_model      = YOLO(_FACE_MODEL_PATH)

# ── Class config ───────────────────────────────────────────────────────────
SUSPICIOUS_LABELS = {"back_watching", "side_watching", "suspicious"}
IGNORED_CLASSES   = {"invigilator"}
HIGH_SEVERITY     = {"suspicious"}
KNOWN_FACES       = {"Mukhtar", "Azeem"}

LABEL_COLOR = {
    "suspicious":     (0,   0,   255),
    "back_watching":  (0,  165,  255),
    "side_watching":  (0,  255,  255),
    "normal":         (0,  255,    0),
    "front_watching": (255, 255,   0),
}
FACE_COLOR = {
    "Mukhtar": (180, 105, 255),
    "Azeem":   (255, 191,   0),
}
DEFAULT_COLOR = (200, 200, 200)

# ── Thresholds ─────────────────────────────────────────────────────────────
# BUG-D1: Was compared with `>`, changed to `>=` in process_frame below.
CHEAT_COUNT_THRESHOLD = 1
FACE_IOU_THRESHOLD    = 0.20
FACE_PROXIMITY_PX     = 120

# ── Audio / whisper config ─────────────────────────────────────────────────
AUDIO_RATE          = 16000
AUDIO_CHUNK         = 8000
AUDIO_CHANNELS      = 1
WHISPER_FLOOR       = 200
WHISPER_CEIL        = 1800
WHISPER_COOLDOWN    = 2.0
WHISPER_ZONE_FX     = 0.50
WHISPER_ZONE_FY     = 0.85
WHISPER_ZONE_RADIUS = 200

# ── Student name pool ──────────────────────────────────────────────────────
_NAME_POOL_BASE = [
    "Student1",  "Student2",  "Student3",  "Student4",  "Student5",
    "Student6",  "Student7",  "Student8",  "Student9",  "Student10",
    "Student11", "Student12", "Student13", "Student14", "Student15",
    "Student16", "Student17", "Student18", "Student19", "Student20",
]

# BUG-D5 FIX: Use itertools.cycle so the pool never runs dry, and store a
# mutable reference so reset_detector_state() can replace it.
_NAME_POOL = list(_NAME_POOL_BASE)
random.shuffle(_NAME_POOL)
_name_iter = itertools.cycle(_NAME_POOL)

# ── IoU tracker state ──────────────────────────────────────────────────────
_tracks: dict[int, dict] = {}
_next_track_id = 0
IOU_THRESHOLD  = 0.35

# ── Shared state between audio thread and process_frame ───────────────────
_meta_lock        = threading.Lock()
_current_frame_meta: dict = {
    "frame_shape":    (480, 640),
    "flagged_tracks": [],
    "all_tracks":     {},
}

# BUG-D2 FIX: use collections.deque; drain via popleft() which is atomic
_whisper_queue: collections.deque = collections.deque(maxlen=200)
_last_whisper_time = 0.0


# ──────────────────────────────────────────────────────────────────────────
# Session reset  (BUG-D3 FIX)
# ──────────────────────────────────────────────────────────────────────────
def reset_detector_state():
    """
    Reset all module-level mutable state so a new exam session starts clean.
    Called by app.py's /reset endpoint.
    """
    global _tracks, _next_track_id, _name_iter
    global _last_whisper_time, _current_frame_meta

    _tracks        = {}
    _next_track_id = 0

    # Rebuild a fresh shuffled cycle
    pool = list(_NAME_POOL_BASE)
    random.shuffle(pool)
    _name_iter = itertools.cycle(pool)

    _last_whisper_time = 0.0
    _whisper_queue.clear()

    with _meta_lock:
        _current_frame_meta["frame_shape"]    = (480, 640)
        _current_frame_meta["flagged_tracks"] = []
        _current_frame_meta["all_tracks"]     = {}

    print("[DETECTOR] State reset for new session.")


# ──────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────
def _iou(a, b):
    """Intersection-over-Union of two (x1,y1,x2,y2) boxes.
    BUG-D4 FIX: guard against zero-area (degenerate) boxes that
    previously caused ZeroDivisionError.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union else 0.0  # BUG-D4: was plain division


def _centre_dist(a, b):
    cx1 = (a[0] + a[2]) / 2;  cy1 = (a[1] + a[3]) / 2
    cx2 = (b[0] + b[2]) / 2;  cy2 = (b[1] + b[3]) / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def _box_centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


# ──────────────────────────────────────────────────────────────────────────
# Tracker helpers
# ──────────────────────────────────────────────────────────────────────────
def _match_or_create_track(box, forced_name: str | None = None):
    global _next_track_id
    best_tid, best_iou = None, 0.0
    for tid, info in _tracks.items():
        score = _iou(box, info["box"])
        if score > best_iou:
            best_iou = score
            best_tid = tid

    if best_iou >= IOU_THRESHOLD and best_tid is not None:
        _tracks[best_tid]["box"] = box
        _tracks[best_tid]["age"] = 0
        if forced_name and _tracks[best_tid]["name"].startswith("Student"):
            _tracks[best_tid]["name"] = forced_name
        return best_tid, _tracks[best_tid]["name"]
    else:
        tid  = _next_track_id
        _next_track_id += 1
        # BUG-D5 FIX: next() on a cycle never raises StopIteration
        name = forced_name if forced_name else next(_name_iter)
        _tracks[tid] = {"name": name, "box": box, "age": 0}
        return tid, name


def _age_tracks():
    to_delete = [tid for tid, info in _tracks.items() if info["age"] > 10]
    for tid in to_delete:
        del _tracks[tid]
    for tid in _tracks:
        _tracks[tid]["age"] += 1


def _nearest_other_name(current_tid: int, frame_tracks: dict) -> str | None:
    if len(frame_tracks) < 2:
        return None
    box = frame_tracks[current_tid]["box"]
    best_name, best_dist = None, float("inf")
    for tid, info in frame_tracks.items():
        if tid == current_tid:
            continue
        d = _centre_dist(box, info["box"])
        if d < best_dist:
            best_dist = d
            best_name = info["name"]
    return best_name


def _resolve_face_for_box(behaviour_box, face_detections: list[dict]) -> str | None:
    best_name, best_score = None, -1.0
    for fd in face_detections:
        score = _iou(behaviour_box, fd["box"])
        if score >= FACE_IOU_THRESHOLD and score > best_score:
            best_score = score
            best_name  = fd["name"]
    if best_name:
        return best_name
    best_dist = float("inf")
    for fd in face_detections:
        d = _centre_dist(behaviour_box, fd["box"])
        if d < FACE_PROXIMITY_PX and d < best_dist:
            best_dist = d
            best_name = fd["name"]
    return best_name


# ──────────────────────────────────────────────────────────────────────────
# Whisper attribution
# ──────────────────────────────────────────────────────────────────────────
def _attribute_whisper() -> tuple[str, str | None]:
    """
    Returns (whisperer_name, nearest_neighbour_or_None).
    Priority 1 → exactly one flagged student in frame.
    Priority 2 → closest to WHISPER_ZONE within radius.
    Priority 3 → "Unknown".
    """
    with _meta_lock:
        flagged  = list(_current_frame_meta["flagged_tracks"])
        all_trk  = dict(_current_frame_meta["all_tracks"])
        h, w     = _current_frame_meta["frame_shape"]

    # Priority 1
    if len(flagged) == 1:
        whisperer = flagged[0]
        wbox = next((info["box"] for info in all_trk.values()
                     if info["name"] == whisperer), None)
        neighbour = None
        if wbox:
            best_dist = float("inf")
            for info in all_trk.values():
                if info["name"] == whisperer:
                    continue
                d = _centre_dist(wbox, info["box"])
                if d < best_dist:
                    best_dist = d
                    neighbour = info["name"]
        return whisperer, neighbour

    # Priority 2
    zone_x = w * WHISPER_ZONE_FX
    zone_y = h * WHISPER_ZONE_FY
    best_name, best_dist = None, float("inf")
    for info in all_trk.values():
        cx, cy = _box_centre(info["box"])
        d = ((cx - zone_x) ** 2 + (cy - zone_y) ** 2) ** 0.5
        if d < WHISPER_ZONE_RADIUS and d < best_dist:
            best_dist = d
            best_name = info["name"]

    if best_name:
        wbox = next((info["box"] for info in all_trk.values()
                     if info["name"] == best_name), None)
        neighbour = None
        if wbox:
            bd2 = float("inf")
            for info in all_trk.values():
                if info["name"] == best_name:
                    continue
                d = _centre_dist(wbox, info["box"])
                if d < bd2:
                    bd2 = d
                    neighbour = info["name"]
        return best_name, neighbour

    return "Unknown", None


# ──────────────────────────────────────────────────────────────────────────
# Background audio thread
# ──────────────────────────────────────────────────────────────────────────
def _audio_worker():
    global _last_whisper_time
    pa     = pyaudio.PyAudio()
    stream = None
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK,
        )
        while True:
            try:
                raw  = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                rms  = float(np.sqrt(np.mean(data ** 2)))
            except Exception:
                time.sleep(0.1)
                continue

            now = time.time()
            if (WHISPER_FLOOR <= rms < WHISPER_CEIL
                    and now - _last_whisper_time >= WHISPER_COOLDOWN):
                _last_whisper_time = now
                whisperer, neighbour = _attribute_whisper()
                _whisper_queue.append({
                    "timestamp":     time.strftime("%H:%M:%S"),
                    "student":       whisperer,
                    "whispering_to": neighbour or "—",
                    "rms":           round(rms, 1),
                })
    except Exception as e:
        print(f"[AUDIO] Worker error: {e}")
    finally:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        pa.terminate()


def get_pending_whisper_events() -> list[dict]:
    """
    Drain and return all pending whisper events atomically.
    BUG-D2 FIX: Use popleft() loop instead of list() + clear() to avoid
    losing events appended between those two operations.
    """
    events = []
    while True:
        try:
            events.append(_whisper_queue.popleft())
        except IndexError:
            break
    return events


if _AUDIO_AVAILABLE:
    _audio_thread = threading.Thread(target=_audio_worker, daemon=True)
    _audio_thread.start()
    print("[AUDIO] Whisper detection thread started.")


# ──────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────
def process_frame(frame):
    """
    Returns:
        frame     - annotated frame
        _         - reserved 0
        labels    - list[str] of detected behaviour classes this frame
        cheating  - bool
        students  - list[str] of visible student names
        incidents - list[dict] with student/action/from_student/severity
    """
    labels:        list[str]  = []
    students:      list[str]  = []
    incidents:     list[dict] = []
    frame_tracks:  dict       = {}
    box_meta:      list       = []
    flagged_names: list[str]  = []

    _age_tracks()

    h, w = frame.shape[:2]

    # ── Face detection ─────────────────────────────────────────────────────
    face_results     = face_model(frame, conf=0.40, verbose=False)
    face_detections: list[dict] = []

    for r in face_results:
        for fbox in r.boxes:
            cls_id    = int(fbox.cls[0])
            face_name = face_model.names[cls_id]
            if face_name not in KNOWN_FACES:
                continue
            fx1, fy1, fx2, fy2 = map(int, fbox.xyxy[0])
            fconf = float(fbox.conf[0])
            face_detections.append({"name": face_name,
                                    "box":  (fx1, fy1, fx2, fy2),
                                    "conf": fconf})
            fc = FACE_COLOR.get(face_name, (255, 255, 255))
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), fc, 2)
            lbl = f"{face_name} {fconf:.2f}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (fx1, fy1 - th - 8), (fx1 + tw + 6, fy1), fc, -1)
            cv2.putText(frame, lbl, (fx1 + 3, fy1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # ── Behaviour detection ────────────────────────────────────────────────
    beh_results = behaviour_model(frame, conf=0.30, verbose=False)

    for r in beh_results:
        for box in r.boxes:
            cls        = int(box.cls[0])
            class_name = behaviour_model.names[cls]
            conf       = float(box.conf[0])

            if class_name in IGNORED_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = LABEL_COLOR.get(class_name, DEFAULT_COLOR)

            forced_name = _resolve_face_for_box((x1, y1, x2, y2), face_detections)
            tid, student = _match_or_create_track(
                (x1, y1, x2, y2), forced_name=forced_name)
            frame_tracks[tid] = _tracks[tid]
            box_meta.append((tid, student, class_name))

            labels.append(class_name)
            if student not in students:
                students.append(student)
            if class_name in SUSPICIOUS_LABELS and student not in flagged_names:
                flagged_names.append(student)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            lbl = f"{class_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, lbl, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Update shared meta for audio thread ────────────────────────────────
    with _meta_lock:
        _current_frame_meta["frame_shape"]    = (h, w)
        _current_frame_meta["flagged_tracks"] = flagged_names
        _current_frame_meta["all_tracks"]     = {
            tid: {"name": info["name"], "box": info["box"]}
            for tid, info in frame_tracks.items()
        }

    # BUG-D1 FIX: was `suspicious_count > CHEAT_COUNT_THRESHOLD` (required 2+).
    # Changed to `>=` so a single suspicious detection triggers cheating.
    suspicious_count = sum(1 for lbl in labels if lbl in SUSPICIOUS_LABELS)
    cheating = suspicious_count >= CHEAT_COUNT_THRESHOLD

    if cheating:
        unique_cheats = sorted(set(l for l in labels if l in SUSPICIOUS_LABELS))
        banner = "ALERT: " + " | ".join(unique_cheats).replace("_", " ").upper()
        cv2.rectangle(frame, (0, 0), (w, 45), (0, 0, 180), -1)
        cv2.putText(frame, banner, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Build incidents ────────────────────────────────────────────────────
    if cheating:
        for tid, student, class_name in box_meta:
            if class_name not in SUSPICIOUS_LABELS:
                continue
            from_s   = _nearest_other_name(tid, frame_tracks)
            severity = "HIGH" if class_name in HIGH_SEVERITY else "LOW"
            incidents.append({
                "student":      student,
                "action":       class_name,
                "from_student": from_s,
                "severity":     severity,
            })

    return frame, 0, labels, cheating, students, incidents