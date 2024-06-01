import cv2
import numpy as np

# Load YOLOv4-tiny model
net = cv2.dnn.readNetFromDarknet('yolov4-tiny-custom.cfg', 'yolov4-tiny-custom_final.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Set input and output nodes
input_node = 'data'
output_nodes = ['output']

# Specify input shape
input_shape = (416, 416)

# Create dummy input blob
dummy_input = np.random.uniform(0, 1, size=(1, 3, *input_shape)).astype('float32')

# Convert model to ONNX format
onnx_model = cv2.dnn.writeNet(net, 'yolov4-tiny-custom.onnx')

# Verify the output model
onnx_net = cv2.dnn.readNetFromONNX('yolov4-tiny-custom.onnx')
output = onnx_net.forward(output_nodes)

