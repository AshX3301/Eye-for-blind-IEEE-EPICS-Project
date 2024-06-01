import cv2
import pytesseract
from pytesseract import Output
import pyttsx3

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Set the speech rate (change it if necessary)
engine.setProperty('rate', 150)


# Set the path to your Tesseract installation (change it if necessary)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale for better OCR performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform text detection using pytesseract
    results = pytesseract.image_to_data(gray, output_type=Output.DICT)

    for i, text_box in enumerate(results['text']):
        if int(results['conf'][i]) > 60:  # Filter out low-confidence detections
            x = results['left'][i]
            y = results['top'][i]
            w = results['width'][i]
            h = results['height'][i]

            # Draw a bounding box around the detected text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Get the detected text and display it
            text = results['text'][i]
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Speak the detected text
            print("Detected Text:", text)
            engine.say(text)
            engine.runAndWait()

    # Display the frame
    cv2.imshow('Text Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
