from ultralytics import YOLO
import cv2

model = YOLO('yolov8n-pose.pt') 

# Open the camera
cap = cv2.VideoCapture(0)

# Read and process frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame is read correctly
    if not ret:
        break

    # Predict with the model
    results = model(frame, show=True)

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()