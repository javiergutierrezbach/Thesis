import torch
from attempt_conversion import LearnedController

# Load your trained controller

controller_name = 'controller_5'
controller = torch.load(f'controllers/{controller_name}.pt')
controller.eval()  # important: switch to eval mode!


# Dummy input for export — shape (1, 4)
x = torch.randn(1, 4, requires_grad=False)

# Export to ONNX
torch.onnx.export(
    controller,
    x,
    f'onnx_controllers/{controller_name}.onnx',  # output filename
    export_params=True,
    opset_version=10,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

print("Exported controller.onnx with one input!")

