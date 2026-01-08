import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from training_exp_mask import Dynamic
from train_dynamics import StateTransitionDataset
from evaluate_dynamic import stat_eval, find_top200_error_fast, find_max_error
from datetime import datetime


def append_training_set(current_data, append_data, scale, out_file):

    # Load your finetuning data
    finetune_data = torch.load(append_data)
    inputs = finetune_data["inputs"]
    outputs = finetune_data["outputs"]

    old_set = torch.load(current_data)
    old_inputs = old_set["inputs"]
    old_outputs = old_set["outputs"]

    old_inputs = old_inputs.detach().clone()
    old_outputs = old_outputs.detach().clone()
    

    if scale:
        num_repeats = int(len(old_inputs) * scale // len(inputs)) # how many times to repeat the finetune data
    else:
        num_repeats = 1

    print(f"Repeating counterexamples {num_repeats} times.")

    # If inputs is a list of tensors -> stack into one tensor
    if isinstance(inputs, list):
        inputs = torch.stack(inputs)

    if isinstance(outputs, list):
        outputs = torch.stack(outputs)

    # Repeat inputs and outputs `num_repeats` times
    repeated_inputs = inputs.repeat((num_repeats, 1)).detach().clone()
    repeated_outputs = outputs.repeat((num_repeats, 1)).detach().clone()


    print(f"Counterexample entries: {len(repeated_inputs)}.")

    total_len = old_inputs.size(0) + repeated_inputs.size(0)

    new_inputs = torch.empty((total_len, old_inputs.size(1)), dtype=torch.float64)
    new_outputs = torch.empty((total_len, old_outputs.size(1)), dtype=torch.float64)

    new_inputs[:old_inputs.size(0)] = old_inputs
    new_inputs[old_inputs.size(0):] = repeated_inputs
    new_outputs[:old_outputs.size(0)] = old_outputs
    new_outputs[old_outputs.size(0):] = repeated_outputs


    # new_inputs = torch.cat([old_inputs, repeated_inputs], dim=0)
    # new_outputs = torch.cat([old_outputs, repeated_outputs], dim=0)

    random_indices = torch.randperm(len(new_inputs))

    new_inputs = new_inputs[random_indices]
    new_outputs = new_outputs[random_indices]

    finetune_data = {
        "inputs": new_inputs,     # tensor of shape (N, 6)
        "outputs": new_outputs    # tensor of shape (N, 4)
    }

    with open(out_file, "wb") as f:
        torch.save(finetune_data, f)

    return finetune_data

def finetune_dynamic(dynamic, data_file, threshold, out_file):

    data = torch.load(data_file)
    inputs = data["inputs"]
    outputs = data["outputs"]

    finetune_dataset = StateTransitionDataset(inputs, outputs)
    finetune_loader = DataLoader(finetune_dataset, batch_size=64, shuffle=True, pin_memory=False, num_workers=0)

    assert finetune_dataset.inputs.dtype == torch.float64, "inputs tensor is not float64!"
    assert finetune_dataset.outputs.dtype == torch.float64, "inputs tensor is not float64!"
    print(finetune_dataset.__len__())
    # Recreate your model architecture

    model = Dynamic(dynamic).double()

    # Set up training tools
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)  # slightly lower lr for finetuning

    # Fine-tune

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(
    #     optimizer, gamma = 0.97, verbose=True
    # )

    epoch_loss = float('inf')

    epoch = 0

    # patience = 20
    # best_loss = float('inf')
    # stagnant_epochs = 0

    model.train()

    avg_loss = float('inf')


    print(f"Threshold: {threshold:.4}\n")


    while avg_loss > threshold and epoch < 50:
        epoch_loss = 0.0

        if epoch == 10:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print("Updating Learning Rate")

        # if epoch == 5 or epoch == 70 or epoch == 75:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1
        #     print("Updating Learning Rate")

        i = 0
        for inputs_batch, targets_batch in finetune_loader:
    

           

            optimizer.zero_grad()
            preds = model(inputs_batch)
            loss = criterion(preds, targets_batch)
            st_train_time = datetime.now()
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            end_train_time = datetime.now()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: grad min={param.grad.min()}, max={param.grad.max()}, mean={param.grad.mean()}")


            
            diff = end_train_time - st_train_time
            # print("optimizer time: ", diff)

            epoch_loss += loss.item() * inputs_batch.size(0)
            i += 1

        avg_loss = epoch_loss / len(finetune_dataset)
        print(f"Finetune Epoch {epoch+1}, Loss: {avg_loss:.4}")

        # Adjust learning rate based on validation loss (here just using train loss)
        #scheduler.step()
        # if epoch % 50 == 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1  # Reduce LR by 10x at epoch 100

        epoch += 1

        # if avg_loss + 1e-6 >= best_loss:
        #     stagnant_epochs += 1
        # else:
        #     stagnant_epochs = 0
        #     best_loss = avg_loss

        # if stagnant_epochs >= patience:
        #     print("Early stopping.")
        #     break

    # Save the updated model
    torch.save(model.nn.state_dict(), out_file)

    return model

def save_dynamic(old_file, new_file):
    model = Dynamic(old_file).double()

    torch.save(model.nn.state_dict(), new_file)

def save_dynamic_data(old_file, new_file):
    data = torch.load(old_file)

    with open(new_file, "wb") as f:
        torch.save(data, f)
    


if __name__ == "__main__":

    inps, outs, errs = find_top200_error_fast("dynamics/dynamic_0.pt", "dynamic_data/train_0.pt", "top200/train_0.pt", k=1000)

    finetune_data = append_training_set("dynamic_data/train_0.pt", "top200/train_0.pt", 0.1, "dynamic_data/finetune_data.pt")
    # finetune_data_filename = "dynamic_data/finetune_data.pt"
    # torch.save(finetune_data, "finetune_data_filename")

    # length = len(finetune_data["inputs"])

    # print(f"Total Dataset: {length}.\n")

     # Load your finetuning data
    #finetune_data = torch.load("counterexamples_data.pt")

    #prev_split = "train_data.pt"

    # mse, mae = stat_eval("dynamic.pt", "finetune_data.pt")
    # print(f"Original dynamic counterexamples MSE: {mse:.8f}")

    # mse, mae = stat_eval("dynamic.pt", finetune_data_filename)
    # print(f"Original dynamic full data MSE: {mse:.8f}")

    model = finetune_dynamic("dynamics/dynamic_0.pt", "dynamic_data/finetune_data.pt", 0.00000000000000000000000000000000000000000000000000000000001, "dynamics/dynamic_0_test.pt")
    #model = finetune_dynamic("dynamics/dynamic_0.pt", "dynamic_data/train_0.pt", 0.00000000000000000000000000000000000000000000000000000000001, "dynamics/dynamic_0_test.pt")
    #model = finetune_dynamic("dynamics/dynamic_0.pt", "counterexamples/counterexample_0.pt", 0.00000000000000000000000000000000000000000000000000000000001, "dynamics/dynamic_0_test.pt")

    # Save the updated model
    torch.save(model.nn.state_dict(), "dynamics/dynamic_finetuned.pt")

    #mse, mae = stat_eval("dynamics/dynamic_finetuned.pt", "finetune_data.pt")
    #print(f"New dynamic finetune data MSE: {mse:.8f}")

    # # mse, mae = stat_eval("dynamics/dynamic_finetuned.pt", finetune_data_filename)
    # # print(f"New dynamic full data MSE: {mse:.8f}")

    mse, mae = stat_eval("dynamics/dynamic_finetuned.pt", "dynamic_data/train_0.pt")
    print(f"New dynamic training data MSE: {mse:.8f}")

    find_max_error("dynamics/dynamic_finetuned.pt", "dynamic_data/train_0.pt")
