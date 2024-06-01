import torch
import onnx
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='detect.py', help='Path to input PyTorch model script')
parser.add_argument('--output', type=str, default='detect.onnx', help='Path to output ONNX model file')
args = parser.parse_args()

# Load PyTorch model
model = torch.load(args.input)

# Create input tensor
input_shape = (1, 3, 416, 416)  # Modify this according to your input shape
input_tensor = torch.randn(input_shape)

# Export model to ONNX
onnx.export(model, input_tensor, args.output)

print(f"Model converted to ONNX format and saved to '{args.output}'")

