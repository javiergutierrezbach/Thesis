import torch
from sample_dynamics import next_state_batch
from training_exp_mask import Dynamic

dynamic = Dynamic("dynamics/dynamic_lip_4.pt")

input_state = torch.tensor([1.17, 1.0828123940978702, -0.14961872770695514, 0.14893211418090924, 0.23248938192288898, -1.0], dtype=torch.float64)

next_state_pred = dynamic(input_state)

input = input_state.unsqueeze(0)

true_next_state = next_state_batch(input).squeeze(0)

print(next_state_pred)
print(true_next_state)

error = next_state_pred - true_next_state

error_norm = torch.linalg.norm(error, ord=2)

squared_error = error ** 2
mse = squared_error.mean()  

print(error)
print(error_norm)
print(mse)