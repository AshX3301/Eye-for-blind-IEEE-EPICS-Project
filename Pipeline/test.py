import cv2
import numpy as np
from detecttest import CurrencyDetection
from maintest import ObjectDetection
import time

# Initialize the models
currency_model = CurrencyDetection()
object_detection_model = ObjectDetection()

# Initialize the video capture object for the webcam
cap = cv2.VideoCapture(0)

# Define the font for FPS display
font = cv2.FONT_HERSHEY_PLAIN

# Initialize FPS variables
starting_time = time.time()
frame_id = 0

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if ret:
        # Increment the frame ID
        frame_id += 1

        # Resize the frame
        frame = cv2.resize(frame, (400, 400))

        # Run the object detection model
        object_detection_output = object_detection_model.detect_objects(frame)

        if object_detection_output is None:
            # Run the currency detection model if objects are detected
            currency_value = currency_model.detect_currencies(frame)

            if currency_value is not None:
                # Output the currency value
                print(currency_value)

            else:
                # Output null value if no currency is detected
                print(None)

        else:
            # Output null value if no objects are detected
            print(None)

        # Display the output frame
        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 1, (0, 0, 255), 2)
        cv2.imshow("Output", frame)

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

