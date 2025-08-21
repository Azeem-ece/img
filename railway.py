from ultralytics import YOLO
import cv2

# Load YOLOv8 model (small, faster for live detection)
model = YOLO("yolov8n.pt")

# Open webcam (0 = default camera, replace with video path if needed)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Count number of people detected
    people_count = 0
    for box in results[0].boxes:
        cls_id = int(box.cls[0])  # class ID
        if model.names[cls_id] == "person":  # check if detected object = person
            people_count += 1

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Show crowd count on screen
    cv2.putText(annotated_frame, f"Crowd Count: {people_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Show frame
    cv2.imshow("Railway Crowd Detector", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
