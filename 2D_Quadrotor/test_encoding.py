from Inverted_Pendulum.queries_mask_real import safe_descent_cond_check
from generate_combined_model_torch import combined_model
from training_exp_mask import Dynamic
import torch

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FRWA')))

from FRWA.queries_mask import safe_descent_cond_check as verify_real
from FRWA.generate_combined_model_torch import combined_model as combined_model_real

if __name__ == "__main__":

    cur_model_file = "models/cert_5.pt"
    cur_controller_file = "controllers/controller_5.pt"
    dynamic_file = "dynamic.pt"
    cur_comb_file = "combined/combined_5.onnx"

    combined_model_real(cur_model_file, cur_controller_file, cur_comb_file)

    pos_limit = 2
    safe_pos = pos_limit - 1

    dynamic = Dynamic(dynamic_file)

    #input_0 = torch.tensor([-0.35, -0.3766288009323993, 0.25, 0.25, -1.2654915953615826, -0.8080280828448365])
    ret, ret_ranges, failed = verify_real(cur_comb_file, cur_model_file, safe_pos = safe_pos, limit_pos = pos_limit, docking_pos = 0.35, vel_limit = 0.5)
    #print(dynamic(input_0))

    print("Number of verification counterexamples: " + str(len(ret)) + "\n")
    print("Number of failed cases: " + str(len(failed)) + "\n")
    print("Verification counterexamples: " + str(ret_ranges) + "\n")

# output 0 = -0.152284510040706
# output 1 = -0.1621769940336287
# output 2 = 0.14575167342364942
# output 3 = 0.18237295361463746