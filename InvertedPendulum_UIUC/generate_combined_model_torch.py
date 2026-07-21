import torch
import torch.nn as nn
import torch.onnx

from training_exp_mask import LyapunovNetworkV, InvertedPendulum
from attempt_conversion import LearnedController


def combined_model(file_1, file_2, file_3, device):
    V = torch.load(file_1, weights_only=False, map_location=device)
    # V = V.to(device="cpu")

    controller = torch.load(file_2, weights_only=False, map_location=device)
    # controller = controller.to(device="cpu")
    # Create the combined network with stacked A and two copies of B

    class CombinedNetwork(nn.Module):
        def __init__(self, model_A, model_B):
            super(CombinedNetwork, self).__init__()
            self.model_A = model_A
            self.model_B1 = model_B
            self.model_B2 = model_B

        def forward(self, x, y):
            input_A = x
            input_B1 = x
            input_B2 = y

            output_A = self.model_A(input_A)
            output_B1 = self.model_B1(input_B1)
            output_B2 = self.model_B2(input_B2)

            return output_A, output_B1, output_B2

    # Create an instance of the combined network
    combined_network = CombinedNetwork(controller, V)

    x = torch.randn(1, 2, requires_grad=True, device=device)
    y = torch.randn(1, 2, requires_grad=True, device=device)

    torch.onnx.export(combined_network, (x, y), file_3, export_params=True, opset_version=10, do_constant_folding=True, input_names=[
                      'input_1', 'input_2'], output_names=['output_1', 'output_2', 'output_3'])
