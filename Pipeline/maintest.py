import cv2
import numpy as np

class ObjectDetection:
    def __init__(self, weights_path="yolov4-tiny-custom_final.weights", config_path="yolov4-tiny-custom.cfg", classes_path="classes.names"):
        print("flagpoint0")
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.classes = []
        with open(classes_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def detect_objects(self, frame, confidence_threshold=0.8, nms_threshold=0.3):
        height, width, channels = frame.shape
        print("flagpoint1")
        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
	
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        font = cv2.FONT_HERSHEY_PLAIN
        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
               # print(confidence)
                if confidence > confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    print("flagpoint2")
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        #print("flagpoint2")
        output1 = confidence
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = self.colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)
                return confidence
          
                """output.append({
                    'label': label,
                    'confidence': confidence,
                    'box': (x, y, w, h)
                })
                """
        

