import torch
import torch.nn as nn
import torch.onnx

from training_exp_safe_mask import LyapunovNetworkV, TwoDimDocking

def single_model(file_1, file_2):
	x = torch.randn(1,4, requires_grad=True, dtype=torch.float32)

	V = torch.load(file_1, map_location=torch.device('cpu')).to(torch.float32)
	V.linear_relu_stack = V.linear_relu_stack.float()
    
	
	class SingleNetwork(nn.Module):
		def __init__(self, model):
			super().__init__()
			self.nn = model.linear_relu_stack

		def forward(self, x):
			output = self.nn(x)
			return output
    
	model = SingleNetwork(V)

	output = model(x)
	print(x, output)
	
        
	torch.onnx.export(model, x, file_2,export_params=True,opset_version=10,do_constant_folding=True,input_names = ['input'],output_names = ['output'])
