import torch
import torchvision

	# Create a PyTorch model
	model = torchvision.models.resnet18()

	# Set the input shape for the model
	input_shape = (1, 3, 224, 224)

	# Define the file name for the ONNX model
	onnx_model_name = "main.onnx"

#	 Export the PyTorch model to ONNX format
	torch.onnx.export(model, torch.randn(input_shape), onnx_model_name, verbose=True)

