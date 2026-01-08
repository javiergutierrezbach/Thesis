from ast import literal_eval
from training_exp_mask import Controller, Dynamic
import torch
from sample_dynamics import next_state_batch

with open("counterexamples.txt", "r") as f:
    data = f.read()

blocks = data.split("\n\n")

dynamic = Dynamic("dynamic.pt")

finetune_outputs = []
finetune_inputs = []

for i, block in enumerate(blocks):

    print(f"Index: {i}\n")

    line = block.split("\n")[1][len("Counterexamples: "):].strip()
    counterexamples = torch.tensor(literal_eval(line))

    controller_file = f"controllers/controller_{i}.pt"

    controller = Controller(dynamic, file_name=controller_file)

    forces = controller.nn(counterexamples)
    #forces = torch.clip(forces,-1,1)

    inputs = torch.cat((counterexamples, forces), 1)

    
    real_next_states = next_state_batch(inputs)
    dyn_next_states = dynamic(inputs)

    # print("Initial inputs: ")
    # print(inp)
    # print("Real state: ")
    # print(real_next_state)
    # print("Dynamic predicted state: ")
    # print(dyn_next_state)
    # print("\n")

    finetune_outputs.append(real_next_states)
    finetune_inputs.append(inputs)


finetune_inputs = torch.cat(finetune_inputs, dim=0)
finetune_outputs = torch.cat(finetune_outputs, dim=0)


data_train = {
    "inputs": finetune_inputs,     # tensor of shape (N, 6)
    "outputs": finetune_outputs    # tensor of shape (N, 4)
}

print(data_train)
torch.save(data_train, "finetune_data.pt")
