import torch
import sys
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from sample_dynamics import sample_train_data



# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../FRWA')))

class StateTransitionDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        # also return idx so we can track per-sample loss
        return self.inputs[idx], self.outputs[idx], idx
    
    
def train_model(train_data, out_file):

    data = torch.load(train_data)

    inputs = data["inputs"]  
    outputs = data["outputs"]


    # num_repeats = 1  # how many times to repeat the finetune data

    # # Repeat inputs and outputs `num_repeats` times
    # repeated_inputs = torch.cat([inputs] * num_repeats, dim=0)
    # repeated_outputs = torch.cat([outputs] * num_repeats, dim=0)


    dataset = StateTransitionDataset(inputs, outputs)

    batch_size = 64
    #batch_size = len(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = nn.Sequential(
        nn.Linear(8, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 6), # output: next_state
    )


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)


    # --- Training loop
    num_epochs = 100
    epoch = 0
    #epoch_loss = 100 


    N = len(dataset)
    sample_losses = torch.zeros(N, dtype=torch.float32)  # on CPU is fine

    top_frac = 0.03  
    tail_weight = 0.5

    while epoch < num_epochs:

        if epoch == 30 or epoch == 60 or epoch == 90:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print("Updating Learning Rate")

        # -----------------------------
        # Phase 1: standard epoch over all data
        # -----------------------------
        sample_losses.zero_()
        epoch_loss = 0.0

        model.train()

        for inputs_batch, targets_batch, idx in loader:
            optimizer.zero_grad()

            preds = model(inputs_batch)

            # per-sample loss
            per_sample = ((preds - targets_batch) ** 2).mean(dim=1)  # [B]

            # record for global top-k computation (no grad needed here)
            sample_losses[idx] = per_sample.detach() # or .cpu().detach()

            # base MSE for this batch
            base_loss = (per_sample).mean()

            base_loss.backward()
            optimizer.step()

            epoch_loss += base_loss.item() * inputs_batch.size(0)

        avg_loss = epoch_loss / N

        # -----------------------------
        # Phase 2: extra training on global top-k
        # -----------------------------
        k_global = max(1, int(top_frac * N))  # e.g. top_frac = 0.05 for 5%
        topk_vals, topk_idx = torch.topk(sample_losses, k_global)

        # Subset dataset to only those indices
        topk_subset = torch.utils.data.Subset(dataset, topk_idx.tolist())
        topk_loader = torch.utils.data.DataLoader(
            topk_subset,
            batch_size=batch_size,
            shuffle=True
        )

        tail_epoch_loss = 0.0

        num_tail_passes = 1

        # extra passes over the global top-k samples

        alpha = 100

        for _ in range(num_tail_passes):

            for inputs_batch, targets_batch, _ in topk_loader:
                optimizer.zero_grad()
                preds = model(inputs_batch)

                tail_mse = ((preds - targets_batch) ** 2).mean(dim=1)  # mean over batch
                tail_loss = tail_mse.mean() + alpha * (tail_mse ** 2).mean()

                (tail_weight * (tail_loss)).backward()  # scale how hard you push tail

                optimizer.step()

                tail_epoch_loss += (tail_loss).item() * inputs_batch.size(0)

        tail_avg_loss = 0 if num_tail_passes == 0 else tail_epoch_loss / (num_tail_passes * k_global)

        total_objective = avg_loss + tail_weight * tail_avg_loss

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"mean_loss: {avg_loss:.4e}, "
            f"tail_loss: {(tail_weight * tail_avg_loss):.4e}, "
            f"combined: {total_objective:.4e}"
        )

        epoch += 1

    torch.save(model.state_dict(), out_file)


if __name__ == "__main__":

    #sample_train_data("dynamic_data/train_test.pt", 'dynamic_data/val_test.pt')

    train_model("dynamic_data/train_test.pt", "dynamics/test.pt")

    #find_max_error("dynamics/test.pt", "dynamic_data/train_test.pt")