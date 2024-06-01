# Load the object detection model
obj_detect_net = cv2.dnn.readNet("yolov4-tiny-custom_final.weights", "yolov4-tiny-custom.cfg")

# Load the currency detection model
currency_detect_net = cv2.dnn.readNet("currency_detect.weights", "currency_detect.cfg")

# Load the class names
classes = []
with open("classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the image
img = cv2.imread("input.jpg")

# Perform object detection
obj_blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
obj_detect_net.setInput(obj_blob)
obj_outs = obj_detect_net.forward(obj_detect_net.getUnconnectedOutLayers())

# Extract the detected objects
obj_boxes = []
for obj_out in obj_outs:
    for detection in obj_out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.85:
            center_x = int(detection[0] * img.shape[1])
            center_y = int(detection[1] * img.shape[0])
            w = int(detection[2] * img.shape[1])
            h = int(detection[3] * img.shape[0])
            x = center_x - w // 2
            y = center_y - h // 2
            obj_boxes.append([x, y, w, h])

# Crop the detected objects and perform currency detection
for box in obj_boxes:
    x, y, w, h = box
    obj_crop = img[y:y+h, x:x+w, :]
    currency_blob = cv2.dnn.blobFromImage(obj_crop, 1/255., (416, 416), [0,0,0], 1, crop=False)
    currency_detect_net.setInput(currency_blob)
    currency_outs = currency_detect_net.forward(currency_detect_net.getUnconnectedOutLayers())
    

