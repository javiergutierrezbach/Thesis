from ast import literal_eval
from training_exp_mask import Controller, Dynamic
import torch
from sample_dynamics import next_state_batch

def parse_counterexamples(dynamic_file, controller_file, counterexamples): 

    dynamic = Dynamic(dynamic_file)

    finetune_inputs = []
    finetune_outputs = []

    controller = Controller(dynamic, file_name=controller_file)

    counterexamples = torch.tensor(counterexamples, dtype=torch.float64)
    forces = controller.nn(counterexamples)
    #forces = torch.clip(forces,-1,1)

    inputs = torch.cat((counterexamples, forces), 1)

    real_next_states = next_state_batch(inputs)
    
    #dyn_next_states = dynamic(inputs)

    # print("Initial inputs: ")
    # print(inp)
    # print("Real state: ")
    # print(real_next_state)
    # print("Dynamic predicted state: ")
    # print(dyn_next_state)
    # print("\n")

    finetune_inputs.append(inputs)
    finetune_outputs.append(real_next_states)

    assert all(t.dtype == torch.float64 for t in finetune_inputs), "Non-float64 tensor found!"
    assert all(t.dtype == torch.float64 for t in finetune_outputs), "Non-float64 tensor found!"

    finetune_inputs = torch.cat(finetune_inputs, dim=0)
    finetune_outputs = torch.cat(finetune_outputs, dim=0)


    data_train = {
        "inputs": finetune_inputs,     # tensor of shape (N, 6)
        "outputs": finetune_outputs    # tensor of shape (N, 4)
    }

    return data_train

if __name__ == "__main__":

    print("test")
