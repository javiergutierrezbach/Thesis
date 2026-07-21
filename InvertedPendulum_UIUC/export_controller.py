import torch
from training_exp_mask import LyapunovNetworkV, InvertedPendulum

# Load your trained controller

controller_name = 'controller_2'
controller = torch.load(f'controllers_epsl_5e-3_lip3_4/{controller_name}.pt').float()
controller.eval()  # important: switch to eval mode!

# controller_name = 'cert_2'
# controller_path = f'/Users/javiergutierrez/Documents/Thesis/Thesis_Repo/InvertedPendulum_UIUC/models_epsl_5e-3_lip3_4/cert_2.pt'

# env = InvertedPendulum()
# controller = LyapunovNetworkV(env)
# controller = torch.load(controller_path)

# Dummy input for export — shape (1, 2)
x = torch.randn(1, 2, requires_grad=False, dtype=torch.float32)

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

