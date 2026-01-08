import sys
import os
import numpy as np
import torch

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../FRWA')))

from training_exp_mask import TwoDimDocking

def generate_clipped_pair():
    # Generate two random values in [-1, 1]
    pair = torch.rand(2) * 2 - 1  # → in [-1, 1]

    # Randomly choose one or both indices to clip to -1 or 1
    idx_to_clip = torch.randperm(2)[:torch.randint(1, 3, (1,))]  # choose 1 or 2 indices

    for i in idx_to_clip:
        pair[i] = torch.tensor([-1.0, 1.0])[torch.randint(0, 2, (1,))]

    return pair

def prepare_data(two_dim_docking, ranges, num_points = 100000, dim = 4, val_split = 0.1, batch_size = 100, max_tries = 500):
        x = torch.Tensor(num_points*4//5, dim).uniform_(
            0.0, 1.0
        )

        pairs = [generate_clipped_pair() for _ in range(num_points)]
        F_x = torch.stack(pairs)  
        

        for i in range(dim):
            min_val, max_val = ranges[i]
            x[:, i] = x[:, i] * (max_val - min_val) + min_val

        y = torch.Tensor(num_points//5, dim).uniform_(
            0.0, 1.0
        )

        for i in range(2):
            y[:, i] = y[:, i] * (two_dim_docking.st_pos - (-two_dim_docking.st_pos)) + (-two_dim_docking.st_pos)

        for j in range(2):
            y[:, j+2] = y[:, j+2] * (two_dim_docking.st_vel_limit - (-two_dim_docking.st_vel_limit)) + (-two_dim_docking.st_vel_limit)

        x = torch.cat((x,y))

        x = torch.concat((x, F_x), 1)
        
        '''
        #confirm data is safe to begin with
        for _ in range(self.max_tries):
            violations = torch.logical_not(self.safe_mask(x))
            
            if not violations.any():
                break

            new_samples = int(violations.sum().item())

            x_new = torch.Tensor(new_samples, self.dim).uniform_(
                0.0, 1.0
            )

            for i in range(self.dim):
                min_val, max_val = self.ranges[i]
                x_new[:, i] = x_new[:, i] * (max_val - min_val) + min_val

            
            x[violations] = x_new
        '''

        random_indices = torch.randperm(len(x))

        x_train = x[random_indices]
        


        x = torch.Tensor(num_points*4//50, dim).uniform_(
            0.0, 1.0
        )

        pairs = [generate_clipped_pair() for _ in range(num_points//10)]
        F_x = torch.stack(pairs)  

        for i in range(dim):
            min_val, max_val = ranges[i]
            x[:, i] = x[:, i] * (max_val - min_val) + min_val

        y = torch.Tensor(num_points//50, dim).uniform_(
            0.0, 1.0
        )

        for i in range(2):
            y[:, i] = y[:, i] * (two_dim_docking.st_pos - (-two_dim_docking.st_pos)) + (-two_dim_docking.st_pos)

        for j in range(2):
            y[:, j+2] = y[:, j+2] * (two_dim_docking.st_vel_limit - (-two_dim_docking.st_vel_limit)) + (-two_dim_docking.st_vel_limit)

        x = torch.cat((x,y))
        x = torch.concat((x, F_x), 1)

        random_indices = torch.randperm(len(x))
        x_val = x[random_indices]

        return x_train, x_val

def next_state(x_0, y_0, v_x_0, v_y_0, F_x, F_y):

    m = 12
    n = 0.001027
    T = 1

    x_t = (2 * v_y_0 / n + 4 * x_0 + F_x / (m * n ** 2)) \
        + (2 * F_y / (m * n)) * T \
        + (-F_x / (m * n ** 2) - 2 * v_y_0 / n - 3 * x_0) * np.cos(n * T) \
        + (-2 * F_y / (m * n ** 2) + v_x_0 / n) * np.sin(n * T)

    y_t = (-2 * v_x_0 / n + y_0 + 4 * F_y / (m * n ** 2)) \
        + (-2 * F_x / (m * n) - 3 * v_y_0 - 6 * n * x_0) * T \
        + (-3 * F_y / (2 * m)) * T ** 2 \
        + (-4 * F_y / (m * n ** 2) + 2 * v_x_0 / n) * np.cos(n * T) \
        + (2 * F_x / (m * n ** 2) + 4 * v_y_0 / n + 6 * x_0) * np.sin(n * T)

    v_x_t = (2 * F_y / (m * n)) \
        + (-2 * F_y / (m * n) + v_x_0) * np.cos(n * T) \
        + (F_x / (m * n) + 2 * v_y_0 + 3 * n * x_0) * np.sin(n * T)

    v_y_t = (-2 * F_x / (m * n) - 3 * v_y_0 - 6 * n * x_0) \
        + (-3 * F_y / m) * T \
        + (2 * F_x / (m * n) + 4 * v_y_0 + 6 * n * x_0) * np.cos(n * T) \
        + (4 * F_y / (m * n) - 2 * v_x_0) * np.sin(n * T)

    next_state = [x_t, y_t, v_x_t, v_y_t]

    return next_state



pos_limit = 2
safe_pos = pos_limit - 1
vel_limit = 0.5

ranges = []


two_dim_docking = TwoDimDocking(safe_pos, pos_limit, vel_limit)

for _ in range(2):
    ranges.append([-two_dim_docking.unsafe_pos-0.2, two_dim_docking.unsafe_pos+0.2])

for _ in range(2):
    ranges.append([-two_dim_docking.vel_limit-0.05, two_dim_docking.vel_limit+0.05])

num_points = 100000
dim = 4

x_train, x_val = prepare_data(two_dim_docking, ranges, num_points, dim)


m = 12
n = 0.001027

outputs_train = []


for i, row in enumerate(x_train):
    outputs_train.append(next_state(*row))

outputs_train = torch.tensor(outputs_train)

data_train = {
    "inputs": x_train,     # tensor of shape (N, 6)
    "outputs": outputs_train    # tensor of shape (N, 4)
}


torch.save(data_train, "train_edgecase_data.pt")

outputs_val = []

for i, row in enumerate(x_val):
    outputs_val.append(next_state(*row))

outputs_val = torch.tensor(outputs_val)

data_val = {
    "inputs": x_val,     # tensor of shape (N, 6)
    "outputs": outputs_val    # tensor of shape (N, 4)
}

torch.save(data_val, "val_edgecase_data.pt")


