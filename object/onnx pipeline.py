import cv2
import numpy as np

net = cv2.dnn.readNet("yolov4-tiny-custom_final.weights", "yolov4-tiny-custom.cfg")

# Specify the target output type as ONNX
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Get the names of the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Get the classes
classes = []
with open("classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set up the input blob
blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set the input blob for the network
net.setInput(blob)

# Get the output of the network
outputs = net.forward(output_layers)

# Save the ONNX model
cv2.dnn.writeTextGraph(net.getLayerNames(), 'yolov4-tiny-custom.onnx')

