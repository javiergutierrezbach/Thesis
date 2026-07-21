import sys
import os
import numpy as np
import torch
import math

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../FRWA')))

from training_exp_mask import SampleData


def prepare_data(num_points = 1000000, dim = 2, val_split = 0.1, batch_size = 100, max_tries = 500):
        
        x = torch.empty((
                num_points, dim), dtype=torch.float32).uniform_(-1.0, 1.0)
        
        y = torch.empty((num_points, 1), dtype=torch.float32). uniform_(-1.5, 1.5)

        x = torch.cat((x, y), 1)
        
        random_indices = torch.randperm(len(x))

        x_train = x[random_indices]
        


        x = torch.empty((num_points//10, dim), dtype=torch.float32).uniform_(
            -1.0, 1.0
        )

        y = torch.empty((num_points//10, 1), dtype=torch.float32).uniform_(
            -1.5, 1.5
        )

        x = torch.concat((x, y), 1)

        random_indices = torch.randperm(len(x))
        x_val = x[random_indices]

        return x_train, x_val

def next_state_batch(xu):

    dt = 0.05
    g = 10
    m = 0.15
    l = 0.5
    b = 0.1
    
    th, thdot, action = xu[:, 0], xu[:, 1], xu[:, 2]

    u =  2 * torch.clamp(action, -1, 1).squeeze()

    newthdot = (1 - b) * thdot + (
            3 * g * 0.5 / (2 * l) * torch.sin(th) +
            3.0 / (m * l ** 2) * u
        ) * dt
    
    newth = th + newthdot * dt

    return torch.stack([newth, newthdot], dim=1)

def sample_train_data(out_file, out_val_file):
  

    num_points = 1000000
    dim = 2

    x_train, x_val = prepare_data(num_points, dim)

    outputs_train = next_state_batch(x_train)

    data_train = {
        "inputs": x_train,     # tensor of shape (N, 6)
        "outputs": outputs_train    # tensor of shape (N, 4)
    }


    torch.save(data_train, out_file)

    outputs_val = next_state_batch(x_val)

    data_val = {
        "inputs": x_val,     # tensor of shape (N, 6)
        "outputs": outputs_val    # tensor of shape (N, 4)
    }

    torch.save(data_val, out_val_file)

def sample_grid(out_file, spacing):
    pos_limit = 2.2
    vel_limit = 0.55
    force_limit = 1.5

    rows = []

    for x in np.arange(-pos_limit, pos_limit, spacing):
        for y in np.arange(-pos_limit, pos_limit, spacing):
            for vx in np.arange(-vel_limit, vel_limit, spacing):
                for vy in np.arange(-vel_limit, vel_limit, spacing):
                    for fx in np.arange(-force_limit, force_limit, spacing):
                        for fy in np.arange(-force_limit, force_limit, spacing):
                            point = torch.tensor([x, y, vx, vy, fx, fy], dtype=torch.float32)
                            rows.append(point)

    x_train = torch.stack(rows)

    print(x_train.shape)

    outputs_train = next_state_batch(x_train)

    data_train = {
        "inputs": x_train,     # tensor of shape (N, 6)
        "outputs": outputs_train    # tensor of shape (N, 4)
    }


    torch.save(data_train, out_file)


if __name__ == "__main__":
     sample_train_data("train_test.pt", "val_test.pt")

