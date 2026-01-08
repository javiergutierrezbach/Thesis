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

# Starting states.
state = np.array([5.0, 5.0, -0.14, -0.14], dtype=np.float32)
state_exact = np.array([5.0, 5.0, -0.14, -0.14], dtype=np.float32)


# Replace with a path to controller network
networkName = "onnx_controllers/controller_5.onnx"

def simulate_traj(networkName, state, state_exact):
    DOCK_RAD = 0.5  # Successful docking distance
    MAX_DIST = 40000.0  # Maximum distance before failure
    m = 12
    n = 0.001027
    V_0 = 0.2
    V_1 = 2.0 * n  # 2 * n
    TIMESTEP = 1

    TIMEOUT = 2000

    x_val_numerical = []
    y_val_numerical = []

    x_val_exact = []
    y_val_exact = []

    dockStatus = None

    for i in range(TIMEOUT):
        print("Exact state is", state_exact)
        print("Numerical state is", state)

        dist = math.sqrt(state[0] ** 2 + state[1] ** 2)
        maxVel = V_0 + V_1 * dist
        curVel = math.sqrt(state[2] ** 2 + state[3] ** 2)
        #if curVel >= maxVel:
            #print("FAILURE: VELOCITY CONSTRAINT")
            #dockStatus = False
            #break
        # Distance-based termination conditions, docking or out of bounds
        if dist <= DOCK_RAD:
            print("DOCKING SUCCESSFUL")
            dockStatus = True
            break
        if dist >= MAX_DIST:
            print("FAILURE: OUT OF BOUNDS")
            dockStatus = False
            break

        session = onnxruntime.InferenceSession(networkName, None)


        input1_name = session.get_inputs()[0].name
        #input2_name = session.get_inputs()[1].name
        output_name = session.get_outputs()[0].name

        # Dummy input_2 (zeros)
        #input2_data = np.zeros((1, 4), dtype=np.float32)

        # Run model with both inputs
        control = session.run(
            [output_name],
            {
                input1_name: state[None, :]
            }
        )[0][0]


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

    
        
        x_val_numerical.append(state[0])
        y_val_numerical.append(state[1])
        x_val_exact.append(state_exact[0])
        y_val_exact.append(state_exact[1])

        # Numerical integration of next state
        dynamics = CWH2dDynamics(integration_method='RK45')
        state_next = dynamics.step(TIMESTEP, state, control)
        state = state_next.astype(np.float32)

        # Exact method of next state
        x_0, y_0, v_x_0, v_y_0 = state_exact[0], state_exact[1], state_exact[2], state_exact[3]

        forces = torch.clip(torch.tensor(control_exact), -1, 1)
        F_x, F_y = forces

        t = TIMESTEP

        x_t = (2 * v_y_0 / n + 4 * x_0 + F_x / (m * n ** 2)) + (2 * F_y / (m * n)) * t + (-F_x / (m * n ** 2) - 2 * v_y_0 / n - 3 * x_0) * np.cos(n * t) + \
            (-2 * F_y / (m * n ** 2) + v_x_0 / n) * np.sin(n * t)
        y_t = (-2 * v_x_0 / n + y_0 + 4 * F_y / (m * n ** 2)) + (-2 * F_x / (m * n) - 3 * v_y_0 - 6 * n * x_0) * t + (-3 * F_y / (2 * m)) * (t ** 2) \
            + (-4 * F_y / (m * n ** 2) + 2 * v_x_0 / n) * np.cos(n * t) + (2 * F_x / (m * n ** 2) + 4 * v_y_0 / n + 6 * x_0) * np.sin(n * t)
        v_x_t = (2 * F_y / (m * n)) + (-2 * F_y / (m * n) + v_x_0) * np.cos(n * t) + (F_x / (m * n) + 2 * v_y_0 + 3 * n * x_0) * np.sin(n * t)
        v_y_t = (-2 * F_x / (m * n) - 3 * v_y_0 - 6 * n * x_0) + (-3 * F_y / m) * t + (2 * F_x / (m * n) + 4 * v_y_0 + 6 * n * x_0) * np.cos(n * t) + (4 * F_y / (m * n) - 2 * v_x_0) * np.sin(n * t)

        print(control_exact)
        print(forces)



        state_exact = np.array([x_t, y_t, v_x_t, v_y_t], dtype=np.float32)

        print(i)

    return dockStatus, steps
        
        
dockStatus, steps = simulate_traj(networkName, state, state_exact)

# No resolution reached, therefore timeout
if dockStatus is None:
    print("FAILURE: TIMEOUT")


# plt.plot(x_val_exact, y_val_exact, color='b', label='closed-form')
# plt.show()