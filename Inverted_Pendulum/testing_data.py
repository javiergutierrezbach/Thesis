import torch
from sample_dynamics import next_state_batch

data = torch.load("dynamic_data/train_0.pt")

inputs = data["inputs"]  
outputs = data["outputs"]

torch.set_printoptions(precision=10)

total = 0

for input, output in zip(inputs, outputs):
    input = input.unsqueeze(0)
    real_next_state = next_state_batch(input).squeeze(0)

    # if not (torch.allclose(real_next_state, output, rtol=1e-5, atol=1e-8)):
    #     print("\nActual state: ")
    #     print(real_next_state)
    #     print("Output: ")
    #     print(output)
    #     total += 1

    if not (torch.equal(real_next_state, output)):
        # print("\nActual state: ")
        # print(real_next_state)
        # print("Output: ")
        # print(output)
        total += 1

print(total)
print(len(inputs))

