import cv2
import numpy as np
import time
import imutils
import pyttsx3

class CurrencyDetection:
    def __init__(self, weights_path="yolov4-tiny-custom_final2.weights", config_path="yolov4-tiny-custom2.cfg", names_path="classes2.names"):
        # Load Yolo
        print("checkpoint0")
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.classes = []
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.outputlayers = [self.layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
	
        # Initialize variables for sum calculation
        self.sum = 0
        self.prev_label = ""
        self.start_time = time.time()

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        #self.cap = cv2.VideoCapture("/dev/video0")

    def detect_currencies(self, frame):
        print("checkpoint1")
        frame = np.array(frame)
        font = cv2.FONT_HERSHEY_PLAIN
        starting_time = time.time()
        while True:
            frame = imutils.resize(frame, width=400, height=400)
            
            height, width, channels = frame.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            self.net.setInput(blob)
            outs = self.net.forward(self.outputlayers)

            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.96:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        print("checkpoint box")
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
            print("checkpoint2")
            label =""
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    confidence = confidences[i]
                    color = self.colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                  
                    cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)
                    # Add to sum if the same label is being constantly predicted for more than 10 seconds
                    if label == self.prev_label:
                        if time.time() - self.start_time > 10:
                            if len(label) > 0:
                                self.sum += int(label)
                                print("Added {} to sum, new sum is {}".format(label, self.sum))
                                text_to_say = "Added " + label + " to the sum " + ", new sum is " + str(sum)
                                self.engine.say(text_to_say)
                                self.engine.runAndWait()
                        # cv2.putText(frame, "Total Sum: " + str(sum), (10, 100), font, 4, (0, 0, 0), 3)
                                self.start_time = time.time()
                    else:
                        self.start_time = time.time()
                    self.prev_label = label
# display total sum
                                
            return confidence                
                                

