import torch
from attempt_conversion import LearnedController

# Load your trained controller

controller_name = 'controller_3'
controller_path = f'../2D_Docking_Safety/controllers/{controller_name}.pt'
#controller_path = f'{controller_name}.pt'


# controller = LearnedController().float()
# controller.load_state_dict(torch.load(controller_path))

controller = torch.load(controller_path).float()


# Dummy input for export — shape (1, 4)
x = torch.randn(1, 4, requires_grad=False, dtype=torch.float32)

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

