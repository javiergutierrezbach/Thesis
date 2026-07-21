import torch
from attempt_conversion import LearnedController

# Load your trained controller

controller_name = 'controller_initial'
#controller = torch.load(f'{controller_name}.pt')
#controller.eval()  # important: switch to eval mode!


nn = LearnedController()
ckpt = torch.load(f'{controller_name}.pt')
nn.net.load_state_dict(ckpt["state_dict"])


# Dummy input for export — shape (1, 4)
x = torch.randn(1, 6, requires_grad=False)

# Export to ONNX
torch.onnx.export(
    nn,
    x,
    f'onnx_controllers/{controller_name}.onnx',  # output filename
    export_params=True,
    opset_version=10,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

print("Exported controller.onnx with one input!")

