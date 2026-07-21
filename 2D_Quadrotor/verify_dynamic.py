import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from queries_mask import safe_descent_cond_check
from generate_combined_model_torch import combined_model
from attempt_conversion import LearnedController
import torch
from datetime import datetime



pos_limit = 1.2
safe_pos = 0.5
docking = 0.1

cert_file = "models/cert_0.pt"

controller_file = "controllers/controller_0.pt"

# net = LearnedController().double()
# net.load_state_dict(torch.load(controller_file))
# torch.save(net, "initial_controller.pt")

# controller_file = "initial_controller.pt"

comb_file = "combined/combined_verify_dyn.onnx"
model_onnx_file = "models/cert_0.onnx"

dynamic_file = "dynamics/dynamic_0.pt"

combined_model(cert_file, controller_file, dynamic_file, comb_file)

st_train_time = datetime.now()

ret, ret_ranges, failed = safe_descent_cond_check(comb_file, model_onnx_file, safe_pos = safe_pos, limit_pos = pos_limit, docking_pos = docking)

end_train_time = datetime.now()
diff = end_train_time - st_train_time

print("\nTotal verification time:", str(diff.seconds) + "\n")

print(ret)
print("Number of counterexamples: ", len(ret))
print(failed)