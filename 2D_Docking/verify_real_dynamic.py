import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from FRWA.queries_mask import safe_descent_cond_check
from generate_combined_model_torch import combined_model_real



pos_limit = 2
safe_pos = pos_limit - 1

cert_file = "models/cert_4.pt"
controller_file = "controllers/controller_4.pt"

comb_file = "combined/combined_verify.onnx"
model_onnx_file = "models/cert_4.onnx"

combined_model_real(cert_file, controller_file, comb_file)

ret, ret_ranges, failed = safe_descent_cond_check(comb_file, model_onnx_file, safe_pos = safe_pos, limit_pos = pos_limit, docking_pos = 0.35, vel_limit = 0.5)

print(ret)
print("Number of counterexamples: ", len(ret))
print(failed)