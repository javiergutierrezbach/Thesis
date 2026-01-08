import sys
import os
import numpy as np
import torch
import math

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../FRWA')))

from training_exp_mask import SampleData, TwoDimDocking


def prepare_data(two_dim_docking, ranges, num_points = 1000000, dim = 4, val_split = 0.1, batch_size = 100, max_tries = 500):
        x = torch.empty((num_points, dim), dtype=torch.float64).uniform_(
             -1.0, 1.0
        )

        F_x = torch.empty((num_points, 2), dtype=torch.float64).uniform_(
            -2.0, 2.0
        )


        
        for i in range(dim):
            min_val, max_val = ranges[i]
            x[:, i] = x[:, i] * (max_val - min_val) + min_val

        x = torch.cat((x, F_x), 1)
        
    

        random_indices = torch.randperm(len(x))

        x_train = x[random_indices]
        


        x = torch.empty((num_points//10, dim), dtype=torch.float64).uniform_(
            0.0, 1.0
        )

        F_x = torch.empty((num_points//10, 2), dtype=torch.float64).uniform_(
            -2.0, 2.0
        )
        #F_x = torch.clip(F_x, -1, 1)

        for i in range(dim):
            min_val, max_val = ranges[i]
            x[:, i] = x[:, i] * (max_val - min_val) + min_val

    
        x = torch.concat((x, F_x), 1)

        random_indices = torch.randperm(len(x))
        x_val = x[random_indices]

        return x_train, x_val

def next_state_batch(xu):
    x, y, vx, vy, fx, fy = xu[:, 0], xu[:, 1], xu[:, 2], xu[:, 3], xu[:, 4], xu[:, 5]
    m = torch.tensor(12.0, dtype=xu.dtype)
    n = torch.tensor(0.001027, dtype=xu.dtype)
    T = torch.tensor(1.0, dtype=xu.dtype)

    angle = n * T

    xt = (2 * vy / n + 4 * x + fx / (m * n ** 2)) \
         + (2 * fy / (m * n)) * T \
         + (-fx / (m * n ** 2) - 2 * vy / n - 3 * x) * torch.cos(angle) \
         + (-2 * fy / (m * n ** 2) + vx / n) * torch.sin(angle)

    yt = (-2 * vx / n + y + 4 * fy / (m * n ** 2)) \
         + (-2 * fx / (m * n) - 3 * vy - 6 * n * x) * T \
         + (-3 * fy / (2 * m)) * T ** 2 \
         + (-4 * fy / (m * n ** 2) + 2 * vx / n) * torch.cos(angle) \
         + (2 * fx / (m * n ** 2) + 4 * vy / n + 6 * x) * torch.sin(angle)

    vxt = (2 * fy / (m * n)) \
          + (-2 * fy / (m * n) + vx) * torch.cos(angle) \
          + (fx / (m * n) + 2 * vy + 3 * n * x) * torch.sin(angle)

    vyt = (-2 * fx / (m * n) - 3 * vy - 6 * n * x) \
          + (-3 * fy / m) * T \
          + (2 * fx / (m * n) + 4 * vy + 6 * n * x) * torch.cos(angle) \
          + (4 * fy / (m * n) - 2 * vx) * torch.sin(angle)

    return torch.stack([xt, yt, vxt, vyt], dim=1)

def sample_train_data(out_file, out_val_file):
    pos_limit = 2
    safe_pos = pos_limit - 1
    vel_limit = 0.5

    ranges = []

    two_dim_docking = TwoDimDocking(safe_pos, pos_limit, vel_limit)

    for _ in range(2):
        ranges.append([-two_dim_docking.unsafe_pos-0.2, two_dim_docking.unsafe_pos+0.2])

    for _ in range(2):
        ranges.append([-two_dim_docking.vel_limit-0.05, two_dim_docking.vel_limit+0.05])

    num_points = 1000000
    dim = 4

    x_train, x_val = prepare_data(two_dim_docking, ranges, num_points, dim)


    m = 12
    n = 0.001027

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
                            point = torch.tensor([x, y, vx, vy, fx, fy], dtype=torch.float64)
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

