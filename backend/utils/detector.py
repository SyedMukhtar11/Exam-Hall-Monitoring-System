"""
detector.py
-----------
DUAL-MODEL ARCHITECTURE
========================
Model 1  (behaviour_model) — best1.pt
    Detects behaviour classes: back_watching, side_watching, suspicious,
    normal, front_watching, invigilator.
    Drives the cheating-threshold logic and incident recording.

Model 2  (face_model) — best2.pt
    Detects named faces: "Mukhtar", "Azeem".
    Runs on every frame BEFORE the behaviour model.
    If a face box overlaps a behaviour box (IoU >= FACE_IOU_THRESHOLD
    or nearest-centre fallback), that track is assigned the real person
    name instead of StudentN.
    If neither Mukhtar nor Azeem is detected, system falls back to the
    Student1-20 pool exactly as before.

IoU TRACKER
===========
Each physical person keeps the same name across frames via IoU-based
box tracking. _match_or_create_track() is called ONCE per box to prevent
ghost tracks and name mismatches.

CHEATING RULE
=============
suspicious_count = number of SUSPICIOUS_LABELS detections in this frame
cheating = True only when suspicious_count > CHEAT_COUNT_THRESHOLD (1)
i.e. 2+ suspicious detections in one frame trigger the alert.
"""

import cv2
import random
from ultralytics import YOLO

# ── Model paths ────────────────────────────────────────────────────────────
_BEHAVIOUR_MODEL_PATH = r"C:\Users\syedm\OneDrive\Desktop\TechNova1\best1.pt"
_FACE_MODEL_PATH      = r"C:\Users\syedm\OneDrive\Desktop\TechNova1\best2.pt"

behaviour_model = YOLO(_BEHAVIOUR_MODEL_PATH)
face_model      = YOLO(_FACE_MODEL_PATH)

# ── Class config ───────────────────────────────────────────────────────────
SUSPICIOUS_LABELS = {"back_watching", "side_watching", "suspicious"}
IGNORED_CLASSES   = {"invigilator"}
HIGH_SEVERITY     = {"suspicious"}

# Named students detected by face_model (best2.pt classes)
KNOWN_FACES = {"Mukhtar", "Azeem"}

# Color per behaviour label (BGR)
LABEL_COLOR = {
    "suspicious":     (0,   0,   255),   # Red
    "back_watching":  (0,  165,  255),   # Orange
    "side_watching":  (0,  255,  255),   # Yellow
    "normal":         (0,  255,    0),   # Green
    "front_watching": (255, 255,   0),   # Cyan
}
# Distinct colors for named-face boxes (BGR)
FACE_COLOR = {
    "Mukhtar": (180, 105, 255),   # Purple-pink — easy to distinguish
    "Azeem":   (255, 191,   0),   # Deep sky blue
}
DEFAULT_COLOR = (200, 200, 200)

# ── Thresholds ─────────────────────────────────────────────────────────────
CHEAT_COUNT_THRESHOLD = 1    # cheating = True when suspicious count > 1
FACE_IOU_THRESHOLD    = 0.20  # relaxed — face box is smaller than body box
FACE_PROXIMITY_PX     = 120   # fallback: max centre-to-centre distance (px)

# ── Student name pool (used when no named faces are detected) ──────────────
_NAME_POOL = [
    "Student1","Student2","Student3","Student4","Student5",
    "Student6","Student7","Student8","Student9","Student10",
    "Student11","Student12","Student13","Student14","Student15",
    "Student16","Student17","Student18","Student19","Student20"
]
random.shuffle(_NAME_POOL)
_name_iter = iter(_NAME_POOL)

# ── IoU tracker state ──────────────────────────────────────────────────────
# track_id -> { "name": str, "box": (x1,y1,x2,y2), "age": int }
_tracks: dict[int, dict] = {}
_next_track_id = 0
IOU_THRESHOLD  = 0.35   # body-to-body IoU to match same person across frames


# ── Geometry helpers ───────────────────────────────────────────────────────
def _iou(a, b):
    """Intersection-over-Union for boxes (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _centre_dist(a, b):
    """Euclidean distance between centres of two boxes."""
    cx1 = (a[0] + a[2]) / 2;  cy1 = (a[1] + a[3]) / 2
    cx2 = (b[0] + b[2]) / 2;  cy2 = (b[1] + b[3]) / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


# ── Tracker helpers ────────────────────────────────────────────────────────
def _match_or_create_track(box, forced_name: str | None = None):
    """
    Return (track_id, student_name) for a behaviour box.

    forced_name: real person name resolved by face_model; if provided and
                 the matched track currently holds a generic StudentN name,
                 the track is upgraded to the real name.
    """
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
        # Upgrade generic name to real name when face is now recognised
        if forced_name and _tracks[best_tid]["name"].startswith("Student"):
            _tracks[best_tid]["name"] = forced_name
        return best_tid, _tracks[best_tid]["name"]
    else:
        tid  = _next_track_id
        _next_track_id += 1
        name = forced_name if forced_name else next(_name_iter, f"Student_{tid + 1}")
        _tracks[tid] = {"name": name, "box": box, "age": 0}
        return tid, name


def _age_tracks():
    """Increment age of all tracks; remove any unseen for >10 frames."""
    to_delete = [tid for tid, info in _tracks.items() if info["age"] > 10]
    for tid in to_delete:
        del _tracks[tid]
    for tid in _tracks:
        _tracks[tid]["age"] += 1


def _nearest_other_name(current_tid: int, frame_tracks: dict) -> str | None:
    """Return name of the spatially closest other tracked student."""
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


# ── Face-to-behaviour box resolver ────────────────────────────────────────
def _resolve_face_for_box(behaviour_box, face_detections: list[dict]) -> str | None:
    """
    Given a behaviour bounding box and the list of face detections from
    best2.pt, return the best-matching face name, or None.

    Two-step strategy:
      1. IoU >= FACE_IOU_THRESHOLD  (strict geometric overlap).
      2. Nearest-centre within FACE_PROXIMITY_PX  (handles face-inside-body
         containment where IoU is low despite full overlap).
    """
    best_name  = None
    best_score = -1.0

    # Step 1 — IoU overlap
    for fd in face_detections:
        score = _iou(behaviour_box, fd["box"])
        if score >= FACE_IOU_THRESHOLD and score > best_score:
            best_score = score
            best_name  = fd["name"]

    if best_name:
        return best_name

    # Step 2 — proximity fallback
    best_dist = float("inf")
    for fd in face_detections:
        d = _centre_dist(behaviour_box, fd["box"])
        if d < FACE_PROXIMITY_PX and d < best_dist:
            best_dist = d
            best_name = fd["name"]

    return best_name


# ── Main entry point ───────────────────────────────────────────────────────
def process_frame(frame):
    """
    Returns:
        frame     - annotated frame (face boxes + behaviour label boxes)
        _         - reserved 0
        labels    - list[str] of detected behaviour class names this frame
        cheating  - bool (True only when suspicious count > CHEAT_COUNT_THRESHOLD)
        students  - list[str] of student names visible this frame
        incidents - list[dict] with student/action/from_student/severity

    Pipeline:
        1. Run face_model (best2.pt) — collect Mukhtar/Azeem detections.
        2. Run behaviour_model (best1.pt) — detect behaviour classes.
        3. For each behaviour box, resolve a real name via face overlap.
           Fall back to Student pool when no face matches.
        4. Apply cheating threshold.
        5. Build incidents only when cheating = True.
    """
    labels:    list[str]   = []
    students:  list[str]   = []
    incidents: list[dict]  = []

    frame_tracks: dict[int, dict] = {}
    box_meta:     list[tuple]     = []   # (tid, student_name, class_name)

    _age_tracks()

    # ── Step 1: Face detection ─────────────────────────────────────────────
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
            face_detections.append({
                "name": face_name,
                "box":  (fx1, fy1, fx2, fy2),
                "conf": fconf,
            })
            # Draw named-face bounding box
            fc = FACE_COLOR.get(face_name, (255, 255, 255))
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), fc, 2)
            label_text = f"{face_name} {fconf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame,
                          (fx1, fy1 - th - 8), (fx1 + tw + 6, fy1),
                          fc, -1)
            cv2.putText(frame, label_text, (fx1 + 3, fy1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 1, cv2.LINE_AA)

    # ── Step 2 & 3: Behaviour detection + name resolution ─────────────────
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

            # Resolve real name from face detections (if any)
            forced_name = _resolve_face_for_box((x1, y1, x2, y2), face_detections)

            # Single tracker call per box — ghost-track fix
            tid, student = _match_or_create_track(
                (x1, y1, x2, y2), forced_name=forced_name
            )
            frame_tracks[tid] = _tracks[tid]
            box_meta.append((tid, student, class_name))

            labels.append(class_name)
            if student not in students:
                students.append(student)

            # Draw behaviour box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{class_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame,
                          (x1, y1 - th - 8), (x1 + tw + 6, y1),
                          color, -1)
            cv2.putText(frame, label_text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)

    # ── Step 4: Cheating decision ──────────────────────────────────────────
    suspicious_count = sum(1 for lbl in labels if lbl in SUSPICIOUS_LABELS)
    cheating = suspicious_count > CHEAT_COUNT_THRESHOLD

    if cheating:
        unique_cheats = sorted(
            set(lbl for lbl in labels if lbl in SUSPICIOUS_LABELS))
        banner_text = "ALERT: " + " | ".join(unique_cheats).replace("_", " ").upper()
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (0, 0, 180), -1)
        cv2.putText(frame, banner_text, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # ── Step 5: Build incidents ────────────────────────────────────────────
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