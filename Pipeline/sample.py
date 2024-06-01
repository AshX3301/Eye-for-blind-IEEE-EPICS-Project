import cv2
import numpy as np
from maintest import ObjectDetection
from detecttest import CurrencyDetection
import time

detector = ObjectDetection()
detector2 = CurrencyDetection()

# Open the video file or capture from a camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
font = cv2.FONT_HERSHEY_PLAIN

# Initialize FPS variables
starting_time = time.time()
frame_id = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    frame_id += 1
    if not ret:
        break

    # Detect objects in the frame
    confidence = detector.detect_objects(frame)

    if confidence is not None and confidence > 0.95:
        label = "Object Detection"
        #detector.draw_bounding_boxes(frame)
    else:
        confidence1 = detector2.detect_currencies(frame)
        if confidence1 is not None and confidence1 > 0.5:
            label = "Currency Detection"
            #detector2.draw_bounding_boxes(frame)
        else:
            continue

    # Display the image and label
    cv2.putText(frame, label, (10, 30), font, 2, (0, 255, 0), 2)
    cv2.imshow('Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()

