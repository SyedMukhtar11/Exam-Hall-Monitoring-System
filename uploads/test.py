from ultralytics import YOLO
import cv2

# Load your custom model
model = YOLO("C:/Users/syedm/OneDrive/Desktop/TechNova1/backend/model/best.pt")

# Use VideoCapture for video files
cap = cv2.VideoCapture("test2.mp4")

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        # Run YOLO inference on the current frame
        results = model(frame)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the frame
        cv2.imshow("YOLO Detection", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # End of video reached
        break

cap.release()
cv2.destroyAllWindows()
