import torch
import sys
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset


# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../FRWA')))

class StateTransitionDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
    
    
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
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = nn.Sequential(
        nn.Linear(6, 64),
        # nn.ReLU(),
        # nn.Linear(64, 64),
        # nn.ReLU(),
        nn.Linear(64, 4)  # output: next_state
    )

    model = model.double()  

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)


    # --- Training loop
    num_epochs = 50
    epoch = 0
    epoch_loss = 100 


    # patience = 3000
    # best_loss = float('inf')
    # stagnant_epochs = 0

    #model.train()
    #model.eval()

    while epoch < num_epochs:

        if epoch != 0 and epoch % 5 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print("Updating Learning Rate")

        epoch_loss = 0.0

        for inputs_batch, targets_batch in loader:

            optimizer.zero_grad()
            preds = model(inputs_batch)
            loss = criterion(preds, targets_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs_batch.size(0)

        avg_loss = epoch_loss / len(dataset)

        #scheduler.step()
        # if avg_loss + 1e-6 >= best_loss:
        #     stagnant_epochs += 1
        # else:
        #     stagnant_epochs = 0
        #     best_loss = avg_loss

        # if stagnant_epochs >= patience:
        #     print("Early stopping.")
        #     #break

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4}")
        epoch += 1

    torch.save(model.state_dict(), out_file)


if __name__ == "__main__":

    train_model("dynamic_data/train_0.pt", "test5.pt")