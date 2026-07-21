"""
File: inputCell.py
Author(s): Umberto Ravaioli, Andrew Wu, Tobey Shim

Numerical integration approach written by Umberto Ravaioli and Andrew Wu.
Exact/analytical approach derived and implemented by Tobey Shim, along with plotting functionality
& console output/printing.

This file implements an exact simulation of the control network's behavior. To run, set state
and state_exact to be the desired starting position, and run the run_environment.py script from terminal.
This script also doubles as a verification that the analytical system dynamics (see 2D_Docking_Dynamics.pdf)
are consistent with the result from numerical integration. If need be, replace "networkName"
with a path to the network.
"""

import math
import numpy as np
import scipy.integrate
import abc
import matplotlib.pyplot as plt
import torch
import onnxruntime
from training_exp_mask import InvertedPendulum
import statistics

# These classes & their superclasses are used for numerical integration.
class BaseDynamics(abc.ABC):
    @abc.abstractmethod
    def step(self, step_size, state, control):
        raise NotImplementedError


class BaseODESolverDynamics(BaseDynamics):
    def __init__(self, integration_method='Euler'):
        self.integration_method = integration_method
        super().__init__()

    @abc.abstractmethod
    def dx(self, t, state_vec, control):
        raise NotImplementedError

    def step(self, step_size, state, control):

        if self.integration_method == "RK45":
            sol = scipy.integrate.solve_ivp(self.dx, (0, step_size), state, args=(control,))

            state = sol.y[:, -1]  # save last timestep of integration solution
        elif self.integration_method == 'Euler':
            state_dot = self.dx(0, state, control)
            state = vector + step_size * state_dot
        else:
            raise ValueError("invalid integration method '{}'".format(self.integration_method))

        return state


class BaseLinearODESolverDynamics(BaseODESolverDynamics):

    def __init__(self, integration_method='Euler'):
        self.A, self.B = self.gen_dynamics_matrices()
        super().__init__(integration_method=integration_method)

    @abc.abstractmethod
    def gen_dynamics_matrices(self):
        raise NotImplementedError

    def update_dynamics_matrices(self, state_vec):
        pass

    def dx(self, t, state_vec, control):
        self.update_dynamics_matrices(state_vec)
        dx = np.matmul(self.A, state_vec) + np.matmul(self.B, control)
        return dx

    def step(self, step_size, state, control):
        return super().step(step_size, state, control)

class CWH2dDynamics(BaseLinearODESolverDynamics):
    def __init__(self, m=12, n=0.001027, integration_method='Euler'):
        self.m = m  # kg
        self.n = n  # rads/s

        super().__init__(integration_method=integration_method)

    def gen_dynamics_matrices(self):
        m = self.m
        n = self.n

        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [3 * n ** 2, 0, 0, 2 * n],
            [0, 0, -2 * n, 0],
        ], dtype=np.float64)

        B = np.array([
            [0, 0],
            [0, 0],
            [1 / m, 0],
            [0, 1 / m],
        ], dtype=np.float64)

        return A, B
    

def simulate_traj(networkName, state, state_exact, docking_radius, max_dist, certificate, pert_rad, timeout):
    DOCK_RAD = docking_radius  # Successful docking distance
    UNSAFE_DIST = max_dist  # Maximum distance before failure
    max_speed = 5
    dt = 0.05
    g = 10
    m = 0.15
    l = 0.5
    b = 0.1


    TIMEOUT = timeout

    state = state.detach().cpu().numpy().astype(np.float32)
    state_exact = state_exact.detach().cpu().numpy().astype(np.float32)


    x_val_numerical = []
    y_val_numerical = []

    x_val_exact = []
    y_val_exact = []

    dockStatus = False

    safe = True

    steps = 0

    for i in range(TIMEOUT):
        print("Exact state is", state_exact)
        print("cert: ", certificate(torch.Tensor(state_exact)))
        #print("Numerical state is", state)

        #dist = math.sqrt(state_exact[0] ** 2 + state_exact[1] ** 2)
        #maxVel = V_0 + V_1 * dist
        #curVel = math.sqrt(state[2] ** 2 + state[3] ** 2)
        #if curVel >= maxVel:
            #print("FAILURE: VELOCITY CONSTRAINT")
            #dockStatus = False
            #break
        # Distance-based termination conditions, docking or out of bounds
        if abs(state_exact[0]) <= DOCK_RAD and abs(state_exact[1]) <= DOCK_RAD:
            print("DOCKING SUCCESSFUL")
            dockStatus = True
            break

        left_unsafe = (
            (state_exact[0] <= -0.6) &  # th condition
            (state_exact[1] <= 0.0)      # thdot condition
        )

        right_unsafe = (
            (state_exact[0] >= 0.6) &    # th condition
            (state_exact[1] >= 0.0)   # thdot condition
        )
        if left_unsafe or right_unsafe:
            print("FAILURE: UNSAFE STATE")
            safe = False
            
            

        # session = onnxruntime.InferenceSession(networkName, None)


        # input1_name = session.get_inputs()[0].name
        # #input2_name = session.get_inputs()[1].name
        # output_name = session.get_outputs()[0].name

        # Dummy input_2 (zeros)
        #input2_data = np.zeros((1, 4), dtype=np.float32)

        # Run model with both inputs
        # control = session.run(
        #     [output_name],
        #     {
        #         input1_name: state[None, :]
        #     }
        # )[0][0]


        session_exact = onnxruntime.InferenceSession(networkName, None)

        input1_exact_name = session_exact.get_inputs()[0].name
        #input2_exact_name = session_exact.get_inputs()[1].name
        output_exact_name = session_exact.get_outputs()[0].name
        
        # Dummy input_2 for exact as well
        #input2_exact_data = np.zeros((1, 4), dtype=np.float32)
        
        control_exact = session_exact.run(
            [output_exact_name],
            {
                input1_exact_name: state_exact[None, :]
            }
        )[0][0]

        #perturbation = torch.rand(1) * (2 * pert_rad) - pert_rad
        perturbation = torch.Tensor([pert_rad])

        
        # x_val_numerical.append(state[0])
        # y_val_numerical.append(state[1])
        x_val_exact.append(state_exact[0])
        y_val_exact.append(state_exact[1])

        # Numerical integration of next state
        #dynamics = CWH2dDynamics(integration_method='RK45')
        # state_next = dynamics.step(dt, state, control)
        # state = state_next.astype(np.float32)

        # Exact method of next state
        th_0, thdot_0 = state_exact[0], state_exact[1]

        u = torch.tensor(control_exact)
        u = u + perturbation
        u = 2 * torch.clip(u, -1, 1)


        newthdot = (1 - b) * thdot_0 + (
            3 * g * 0.5 / (2 * l) * np.sin(th_0) +
            3.0 / (m * l ** 2) * u
        ) * dt

        newth = th_0 + newthdot * dt

        print(control_exact)
        print(perturbation)
        print(u)

        state_exact = np.array([to_scalar(newth), to_scalar(newthdot)], dtype=np.float32)

        steps += 1

        print(i)

    return dockStatus, steps, safe

def to_scalar(z):
    # torch tensor
    if hasattr(z, "detach"):
        z = z.detach().cpu().numpy()
    # numpy scalar or numpy array
    if isinstance(z, np.ndarray):
        return float(z.reshape(-1)[0])
    # python float/int already
    return float(z)

def sample_initial_points_2d_docking(
    env,
    cert,
    N=1000,
    pos_min=0.25,
    device="cpu",
    dtype=torch.float32,
    max_tries=200000
):
    """
    Sample N points with:
      - NOT in goal
      - NOT unsafe
      - NOT too close to goal (origin), controlled by pos_min and mode

    State x: (N,4) = [px, py, vx, vy]
    """
    lo = torch.tensor(
        [-env.unsafe_pos, -env.unsafe_pos],
        device=device, dtype=dtype
    )
    hi = torch.tensor(
        [env.unsafe_pos,  env.unsafe_pos],
        device=device, dtype=dtype
    )

    out = []
    remaining = N
    tries = 0

    while remaining > 0 and tries < max_tries:
        tries += 1
        M = max(remaining * 5, 2000)

        x = torch.rand((M, 2), device=device, dtype=dtype) * (hi - lo) + lo
        th, thdot = x[:, 0], x[:, 1]

        # "not near goal" condition
        far_from_goal = (th.abs() >= pos_min) 

        good = (~env.goal_mask(x)) & (~env.unsafe_mask(x)) & far_from_goal
        safe = (cert(x).squeeze(-1) < 1.0)

        x_good = x[good & safe]
        if x_good.numel() == 0:
            continue

        take = min(remaining, x_good.shape[0])
        out.append(x_good[:take])
        remaining -= take

    if remaining > 0:
        raise RuntimeError(
            f"Only sampled {N-remaining}/{N} points after {tries} tries. "
            "Increase max_tries or relax pos_min / masks."
        )

    return torch.cat(out, dim=0)


if __name__ == "__main__":


    cert_name = "models/cert_0_lip7.pt"
    cert = torch.load(cert_name, weights_only=False, map_location=torch.device("cpu"))
    cert.eval()
    # Replace with a path to controller network

    controller_name_init = 'pend_100_ppo_128'
    controller_name = 'controller_0_lip7'
    networkName_true = '/Users/javiergutierrez/Documents/Thesis/Thesis_Repo/InvertedPendulum_UIUC/onnx_controllers/controller_2.onnx'
    networkName_init = f"onnx_controllers/{controller_name_init}.onnx"
    networkName = f"onnx_controllers/{controller_name}.onnx"

    numpoints = 5000

    env = InvertedPendulum()

    x0 = sample_initial_points_2d_docking(env, cert, N=numpoints)

    #x0 = torch.Tensor([[-0.35, -1.18, -0.25, 0.0]])

    docking_radius = 0.2
    max_dist = 0.6
    pert_rad = 0.0
    timeout_epochs = 2000

    
    num_docks_init = 0
    num_unsafes_init = 0
    num_timeouts_init = 0
    unsafes_init = []

    steps_list_init = []
    total_steps_init = 0

    for i in range(len(x0)):
      
        dockStatus, steps, safe = simulate_traj(networkName_init, x0[i], x0[i], docking_radius, max_dist, cert, pert_rad, timeout_epochs)

        print(dockStatus)

        if dockStatus is True:
            num_docks_init += 1

            steps_list_init.append(steps)
            total_steps_init += steps
        else:
            num_timeouts_init += 1

        if safe is False:
            num_unsafes_init += 1
            unsafes_init.append(i)

    num_docks_true = 0
    num_unsafes_true = 0
    num_timeouts_true = 0
    unsafes_true = []

    steps_list_true = []
    total_steps_true = 0

    for i in range(len(x0)):
      
        dockStatus, steps, safe = simulate_traj(networkName_true, x0[i], x0[i], docking_radius, max_dist, cert, pert_rad, timeout_epochs)

        print(dockStatus)

        if dockStatus is True:
            num_docks_true += 1

            steps_list_true.append(steps)
            total_steps_true += steps
        else:
            num_timeouts_true += 1

        if safe is False:
            num_unsafes_true += 1
            unsafes_true.append(i)

    
    num_docks = 0
    num_unsafes = 0
    num_timeouts = 0
    unsafes = []

    steps_list = []
    total_steps = 0

    for i in range(len(x0)):
      
        dockStatus, steps, safe = simulate_traj(networkName, x0[i], x0[i], docking_radius, max_dist, cert, pert_rad, timeout_epochs)

        print(dockStatus)

        if dockStatus is True:
            num_docks += 1


            steps_list.append(steps)
            total_steps += steps

        else:
            num_timeouts += 1

        if safe is False:
            num_unsafes += 1
            unsafes.append(i)

        

    print("num docks initial: ", num_docks_init)
    print("num unsafes initial: ", num_unsafes_init)
    print("num timeouts initial: ", num_timeouts_init)

    avg_steps_init = total_steps_init / num_docks_init
    print("avg steps initial: ", avg_steps_init)
    print("max steps initial: ", max(steps_list_init))

    #print(steps_list_init)
    print("std dev steps initial: ", statistics.stdev(steps_list_init))

    print("\nnum docks true: ", num_docks_true)
    print("num unsafes true: ", num_unsafes_true)
    print("num timeouts true: ", num_timeouts_true)

    avg_steps_true = total_steps_true / num_docks_true
    print("avg steps true: ", avg_steps_true)
    print("max steps true: ", max(steps_list_true))

    #print(steps_list_init)
    print("std dev steps true: ", statistics.stdev(steps_list_true))

    print("\nnum docks: ", num_docks)
    print("num unsafes: ", num_unsafes)
    print("num timeouts: ", num_timeouts)

    avg_steps = total_steps / num_docks
    print("avg steps: ", avg_steps)
    print("max steps: ", max(steps_list))

    #print(steps_list)
    print("std dev steps: ", statistics.stdev(steps_list))

     


    # plt.plot(x_val_exact, y_val_exact, color='b', label='closed-form')
    # plt.show()    

    #Exact state is [ 0.5222759  -1.408922   -0.11554945  0.24840283]
# cert:  tensor([0.9979], dtype=torch.float64, grad_fn=<AddBackward0>)
# [ 0.3433984  -0.61038846]