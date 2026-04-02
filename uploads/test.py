from ultralytics import YOLO
import cv2
import os

# --- 1. SETUP & MODEL LOADING ---
# Use a raw string (r"") to handle Windows backslashes correctly
MODEL_PATH = r"C:\Users\syedm\OneDrive\Desktop\TechNova1\best1.pt"
VIDEO_PATH = "test6.mp4"

# Load the model
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()

model = YOLO(MODEL_PATH)

# --- 2. CONFIGURATION ---
CHEATING_CLASSES = {"back_watching", "side_watching", "suspicious"}
IGNORED_CLASSES = {"invigilator"}

CLASS_COLORS = {
    "suspicious":     (0, 0, 255),    # Red
    "back_watching":  (0, 165, 255),  # Orange
    "side_watching":  (0, 255, 255),  # Yellow
    "normal":         (0, 255, 0),    # Green
    "front_watching": (255, 255, 0),  # Cyan
}

# --- 3. VIDEO PROCESSING ---
cap = cv2.VideoCapture("test8.mp4")

if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    exit()

print("Processing... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO inference
    results = model(frame, conf=0.30, verbose=False)
    cheating_found = []

    for r in results:
        for box in r.boxes:
            # Extract class and confidence
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])

            if class_name in IGNORED_CLASSES:
                continue

            # Bounding Box Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = CLASS_COLORS.get(class_name, (200, 200, 200))

            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label Background
            label = f"{class_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            
            # Draw Label Text
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            if class_name in CHEATING_CLASSES:
                cheating_found.append(class_name)

    # --- 4. UI OVERLAY ---
    # Draw top banner if suspicious behavior is detected
    if cheating_found:
        # Create unique list of detected behaviors
        unique_cheats = sorted(set(cheating_found))
        banner_text = "ALERT: " + " | ".join(unique_cheats).replace("_", " ").upper()
        
        # Draw semi-transparent red rectangle at top
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (0, 0, 180), -1)
        cv2.putText(frame, banner_text, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the result
    cv2.imshow("Exam Monitor - Press Q to quit", frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- 5. CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("Monitoring finished.")