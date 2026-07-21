import torch
import torch.nn as nn
import torch.onnx

from training_exp_mask import LyapunovNetworkV, InvertedPendulum


def single_model(file_1, file_2, device):
    x = torch.randn(1, 2, requires_grad=True, device=device)

    model = torch.load(file_1, map_location=device, weights_only=False)
    torch.onnx.export(model, x, file_2, export_params=True, opset_version=10,
                      do_constant_folding=True, input_names=['input'], output_names=['output'])
