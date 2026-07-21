import sys
import os
import numpy as np
import torch
import math
import scipy.integrate

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../FRWA')))

from training_exp_mask import SampleData


def prepare_data(num_points = 1000000, val_split = 0.1, batch_size = 100, max_tries = 500):
        

        # Training

        p = torch.empty((
                num_points, 2), dtype=torch.float32).uniform_(-0.21, 0.21)
        
        theta = torch.empty((
                num_points, 1), dtype=torch.float32).uniform_(-0.21 * np.pi, 0.21 * np.pi)
        
        v = torch.empty((
                num_points, 2), dtype=torch.float32).uniform_(-0.42, 0.42)
        
        omega = torch.empty((
                num_points, 1), dtype=torch.float32).uniform_(-0.42, 0.42)
        
        u = torch.empty((num_points, 2), dtype=torch.float32).uniform_(0, 6.2)

        x = torch.cat((p, theta, v, omega, u), 1)
        
        random_indices = torch.randperm(len(x))

        x_train = x[random_indices]

        # Validation

        p = torch.empty((
                num_points // 10, 2), dtype=torch.float32).uniform_(-0.21, 0.21)
        
        theta = torch.empty((
                num_points // 10, 1), dtype=torch.float32).uniform_(-0.21 * np.pi, 0.21 * np.pi)
        
        v = torch.empty((
                num_points // 10, 2), dtype=torch.float32).uniform_(-0.42, 0.42)
        
        omega = torch.empty((
                num_points // 10, 1), dtype=torch.float32).uniform_(-0.42, 0.42)
        
        u = torch.empty((num_points // 10, 2), dtype=torch.float32).uniform_(0,6.2)

        x = torch.cat((p, theta, v, omega, u), 1)
        
        random_indices = torch.randperm(len(x))

        x_val = x[random_indices]

        return x_train, x_val

def quad_next_state_batch(x: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
    """
    Vectorized next state computation using Euler integration.
    x: (batch, 8) -> [px, pz, theta, vx, vz, omega, u1, u2]
    Returns: (batch, 6) next state
    """
    # Physics parameters (matching your earlier code)
    length = 0.25
    mass = 0.486
    inertia = 0.00383
    gravity = 9.81

    # Split input
    state = x[:, :6]   # (batch, 6)
    u = x[:, 6:]       # (batch, 2)

    px, pz, theta, vx, vz, omega = state.unbind(dim=1)
    u1, u2 = u.unbind(dim=1)

    # Forces
    thrust_sum = u1 + u2
    thrust_diff = u1 - u2                   
    torque = length * thrust_diff

    ax = -(thrust_sum / mass) * torch.sin(theta)
    az = (thrust_sum / mass) * torch.cos(theta) - gravity
    alpha = torque / inertia

    # Euler integration
    px_next = px + dt * vx
    pz_next = pz + dt * vz
    theta_next = theta + dt * omega
    vx_next = vx + dt * ax
    vz_next = vz + dt * az
    omega_next = omega + dt * alpha

    next_state = torch.stack([
        px_next, pz_next, theta_next,
        vx_next, vz_next, omega_next
    ], dim=1)

    return next_state  # shape (batch, 6)

# def forward(t, x, u):
#         """
#         Compute the continuous-time dynamics (batched, pytorch).
#         This is the actual computation that will be bounded using auto_LiRPA.
#         """
#         length=0.25 
#         mass=0.486 
#         inertia=0.00383 
#         gravity=9.81

#         px, pz, theta, vx, vz, omega = x
#         u1, u2 = u

#         thrust_sum = u1 + u2
#         thrust_diff = u1 - u2

#         ax = -(np.sin(theta) / mass) * thrust_sum
#         az =  ((np.cos(theta) / mass) * thrust_sum) - gravity
#         alpha = (length / inertia) * thrust_diff

#         xdot = np.array([vx,       # ṗx
#                         vz,       # ṗz
#                         omega,    # θ̇
#                         ax,
#                         az,
#                         alpha],
#                         dtype=float)
        
#         return xdot

# def next_pose(x, u, dt: float):
#     """
#     Integrate continuous dynamics for time dt using solve_ivp.
#     x: (6,) torch.Tensor or np.ndarray
#     u: (2,) torch.Tensor or np.ndarray
#     returns: same type as x, shape (6,)
#     """
#     # Convert to numpy
#     if isinstance(x, torch.Tensor):
#         x_np = x.detach().cpu().numpy()
#     else:
#         x_np = np.asarray(x, dtype=float)

#     if isinstance(u, torch.Tensor):
#         u_np = u.detach().cpu().numpy()
#     else:
#         u_np = np.asarray(u, dtype=float)

#     sol = scipy.integrate.solve_ivp(
#         fun=lambda t, x_val: forward(t, x_val, u_np),
#         t_span=[0.0, dt],
#         y0=x_np,
#         method="RK45",
#     )

#     x_next_np = sol.y[:, -1]

#     if isinstance(x, torch.Tensor):
#         return torch.from_numpy(x_next_np).to(x.dtype).to(x.device)
#     else:
#         return x_next_np

def sample_train_data(out_file, out_val_file):
  

    num_points = 1000000
    dim = 6

    x_train, x_val = prepare_data(num_points)

    outputs_train = quad_next_state_batch(x_train)

    data_train = {
        "inputs": x_train,     # tensor of shape (N, 6)
        "outputs": outputs_train    # tensor of shape (N, 4)
    }


    torch.save(data_train, out_file)

    outputs_val = quad_next_state_batch(x_val)

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

