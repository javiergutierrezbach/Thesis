import random
import torch
import torch.onnx
#from onnx2torch import convertk
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import onnxruntime
import lightning.pytorch as pl
import math
from attempt_conversion import LearnedController
#from create_points import read_in_points
#from generate_combined_model_torch import combined_model
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import os

#from experimental_rollouts import Rollout

#trying easier (larger docking regions) at a time
#try further away (50-100)
#try tanh
class LyapunovNetworkV(nn.Module):
    def __init__(self):
        super().__init__()
        
        #dim = 196
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 128),   # Wider input layer
            nn.ReLU(),
            nn.Linear(128, 64), # Deeper/wider middle
            nn.ReLU(),
            nn.Linear(64, 64), # Deeper/wider middle
            nn.ReLU(),
            nn.Linear(64, 32), # Deeper/wider middle
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        #logits_goal_mask = self.two_dim_docking.goal_mask(x)
        #logits[logits_goal_mask] = -0.1
        #logits_unsafe_mask = self.two_dim_docking.unsafe_mask(x)
        #logits[logits_unsafe_mask] = 1.1
        return logits
    
class Quadrator2D():
    def __init__(self):
        self.st_pos = 0.1
        self.unsafe_pos = 0.2

        self.dt = 0.05
        self.length=0.25 
        self.mass=0.486 
        self.inertia=0.00383 
        self.gravity=9.81

        self.u_hover = self.mass * self.gravity / 2.0  # scalar

        # Overall state-space limits (similar to your env / training setup)
        self.pos_limit = 0.2           # |px|, |pz| <= 0.3
        self.theta_limit = 0.2 * math.pi  # |theta| <= 0.35*pi
        self.vel_limit = 0.4             # |vx|, |vz| <= 2
        self.omega_limit = 0.4          # |omega| <= ~1.2

        self.pos_init = 0.1             # |px|, |pz| <= 0.3
        self.theta_init = 0.1 * math.pi  # |theta| <= 0.35*pi
        self.vel_init = 0.2             # |vx|, |vz| <= 2
        self.omega_init = 0.2         # |omega| <= ~1.2

        # Goal region (small box around hover at origin)
        self.goal_pos = 0.03              # |px|, |pz| < 0.1
        self.goal_theta = 0.04 * math.pi            # |theta| < 0.1 rad
        self.goal_vel = 0.05            # |vx|, |vz| < 0.2
        self.goal_omega = 0.05            # |omega| < 0.2
 

    def space_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overall domain:
          |px|, |pz| <= pos_limit
          |theta| <= theta_limit
          |vx|, |vz| <= vel_limit
          |omega| <= omega_limit
        """
        px, pz, theta, vx, vz, omega = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

        mask = (px.abs() <= self.pos_limit)
        mask.logical_and_(pz.abs() <= self.pos_limit)
        mask.logical_and_(theta.abs() <= self.theta_limit)
        mask.logical_and_(vx.abs() <= self.vel_limit)
        mask.logical_and_(vz.abs() <= self.vel_limit)
        mask.logical_and_(omega.abs() <= self.omega_limit)
        return mask

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Goal = near hover:
          |px|, |pz| < goal_pos
          |theta| < goal_theta
          |vx|, |vz| < goal_vel
          |omega| < goal_omega
        and inside space_mask.
        """
        px, pz, theta, vx, vz, omega = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

        mask = (px.abs() < self.goal_pos)
        mask.logical_and_(pz.abs() < self.goal_pos)
        mask.logical_and_(theta.abs() < self.goal_theta)
        mask.logical_and_(vx.abs() < self.goal_vel)
        mask.logical_and_(vz.abs() < self.goal_vel)
        mask.logical_and_(omega.abs() < self.goal_omega)
        mask.logical_and_(self.space_mask(x))
        return mask

    def goal_mask_smaller(self, x: torch.Tensor) -> torch.Tensor:
        """
        Slightly smaller inner goal region for margin.
        """
        eps = 0.001
        px, pz, theta, vx, vz, omega = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

        mask = (px.abs() < self.goal_pos - eps)
        mask.logical_and_(pz.abs() < self.goal_pos - eps)
        mask.logical_and_(theta.abs() < self.goal_theta - eps)
        mask.logical_and_(vx.abs() < self.goal_vel - eps)
        mask.logical_and_(vz.abs() < self.goal_vel - eps)
        mask.logical_and_(omega.abs() < self.goal_omega - eps)
        mask.logical_and_(self.space_mask(x))
        return mask

    def nongoal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        States inside space but not in goal (like the pendulum version).
        """
        px, pz, theta, vx, vz, omega = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

        # Outside at least one goal coordinate
        mask = (px.abs() >= self.goal_pos)
        mask.logical_or_(pz.abs() >= self.goal_pos)
        mask.logical_or_(theta.abs() >= self.goal_theta)
        mask.logical_or_(vx.abs() >= self.goal_vel)
        mask.logical_or_(vz.abs() >= self.goal_vel)
        mask.logical_or_(omega.abs() >= self.goal_omega)

        # But still in the overall space
        mask.logical_and_(self.space_mask(x))
        return mask

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define 'unsafe' regions near the boundaries where the quadrotor is
        close to leaving the domain and moving outward.
        """

        px, pz, theta, vx, vz, omega = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

        margin = 0.02   # width of near-edge unsafe band

        # Near left/right boundary and moving further outward
        left_unsafe = (
            (px >= -self.pos_limit) &
            (px <= -self.pos_limit + margin) &
            (vx < 0)                     # moving further left
            )

        right_unsafe = (
            (px <= self.pos_limit) &
            (px >= self.pos_limit - margin) &
            (vx > 0)                     # moving further right
            )

        # Near bottom boundary (low altitude) and moving downward
        bottom_unsafe = (
            (pz >= -self.pos_limit) & (pz <= -self.pos_limit + margin) &   # near bottom edge
            (vz < 0)                                                   # falling further down
        )

        top_unsafe = (
            (pz <= self.pos_limit) & (pz >= self.pos_limit - margin) &   # near top edge
            (vz > 0)                                                   # rising further up
        )

        # Large tilt near theta limit with angular velocity pushing it further
        left_tilt_unsafe = (
            (theta >= -self.theta_limit) & (theta <= -self.theta_limit + (0.02*math.pi)) &
            (omega < 0.0)
        )
        right_tilt_unsafe = (
            (theta <= self.theta_limit) & (theta >= self.theta_limit - (0.02*math.pi)) &
            (omega > 0.0)
        )

        unsafe_mask = left_unsafe | right_unsafe | bottom_unsafe | top_unsafe | left_tilt_unsafe | right_tilt_unsafe

        # unsafe_mask = abs(px) >= self.unsafe_pos - margin
        # unsafe_mask.logical_or_(abs(pz) >= self.unsafe_pos - margin)
        # unsafe_mask.logical_or_(abs(theta) >= self.theta_limit - margin)
        # unsafe_mask.logical_or_(abs(vx) >= self.vel_limit - margin)
        # unsafe_mask.logical_or_(abs(vz) >= self.vel_limit - margin)
        # unsafe_mask.logical_or_(abs(omega) >= self.omega_limit - margin)

        unsafe_mask.logical_and_(self.space_mask(x))

        return unsafe_mask

    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Safe = inside space and not in unsafe regions.
        """
        mask = ~self.unsafe_mask(x)
        mask.logical_and_(self.space_mask(x))
        return mask

    def init_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Initial set: moderate box around origin, not in goal,
        inside space, and safe.
        """
        px, pz, theta, vx, vz, omega = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]


        mask = (px.abs() <= self.pos_init)
        mask.logical_and_(pz.abs() <= self.pos_init)
        mask.logical_and_(theta.abs() <= self.theta_init)
        mask.logical_and_(vx.abs() <= self.vel_init)
        mask.logical_and_(vz.abs() <= self.vel_init)
        mask.logical_and_(omega.abs() <= self.omega_init)

        mask.logical_and_(self.nongoal_mask(x))
        mask.logical_and_(self.safe_mask(x))
        mask.logical_and_(self.space_mask(x))
        return mask

class Controller():
    def __init__(self, dynamic, model, file_name = "init_controller.pth", isInitial=False, t=0.05, device="cpu"):
        self.file_name = file_name
        self.device = device
        self.dynamic = dynamic
        self.model = model

        u_hover = self.model.mass * self.model.gravity / 2.0
        self.u_hover_vec = torch.tensor([u_hover, u_hover], dtype=torch.float64).unsqueeze(0)  # (1,2)

        self.t = t
        
        if (isInitial):
            self.nn = LearnedController().to(self.device)
            #self.nn.load_state_dict(torch.load(file_name, weights_only=False, map_location=self.device))
            # ckpt = torch.load(file_name, map_location=self.device)
            # self.nn.net.load_state_dict(ckpt["state_dict"])
        else:
            self.nn = torch.load(file_name, weights_only=False, map_location=self.device)
            self.nn = self.nn
        #self.nn = self.nn.to(device="cuda")

        # checking a more elaborate inductive property holds (closer or velocity decreases in appropriate direction)

        # velocity expansion is arbitrary


    def next_step(self, states, actions=None):


        states = states

        if actions is None:
            actions = self.nn(states)

        u_hover_vec = self.u_hover_vec.to(device=states.device, dtype=states.dtype)

        
        # delta-around-hover controller
        u = torch.clamp(u_hover_vec + actions, 0.0, 6.0)
        
        input = torch.cat((states, u), 1)

        next_step = self.dynamic(input)

        return next_step
    
class Dynamic(nn.Module):
    def __init__(self, file_name = "dynamic.pt"):

        super().__init__()

        self.nn = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6), # output: next_state
        )
        
        self.nn.load_state_dict(torch.load(file_name))

    def forward(self, x):

        next_state = self.nn(x)

        return next_state

#num_points = 10000000
class SampleData(pl.LightningDataModule):
    def __init__(self, train_file_name, val_file_name, env, device="cpu", num_points = 10000000, dim = 6, val_split = 0.1, batch_size = 10000, max_tries = 5000):
        super().__init__()
        self.num_points = num_points
        self.env = env
        self.dim = dim
        self.val_split = val_split
        self.batch_size = batch_size
        self.max_tries = max_tries
        self.ranges = []
        self.train_file_name = train_file_name
        self.val_file_name = val_file_name
        self.device = device

        for _ in range(2):
            self.ranges.append([-self.env.unsafe_pos-0.02, self.env.unsafe_pos+0.02])

        self.ranges.append([-self.env.theta_limit-0.02, self.env.theta_limit+0.02])

        for _ in range(2):
            #ranges.append([-self.vel_limit,self.vel_limit])
            self.ranges.append([-self.env.vel_limit-0.02,self. env.vel_limit+0.02])

        self.ranges.append([-self.env.omega_limit-0.02, self.env.omega_limit+0.02])
        
    '''
    def safe_mask(self, x):
        safe_mask = x[:, 3:].norm(dim=-1, p=2) <= self.v0+self.v1*(x[:, :2].norm(dim=-1, p=2))
        return safe_mask
    '''

    def prepare_data(self):

        # overall
        x = torch.Tensor(self.num_points*4//5, self.dim).uniform_(
            0.0, 1.0
        )

        for i in range(self.dim):
            min_val, max_val = self.ranges[i]
            x[:, i] = x[:, i] * (max_val - min_val) + min_val


        # init 
        y = torch.Tensor(int(self.num_points//5), self.dim).uniform_(
            0.0, 1.0
        )

        for i in range(2):
            y[:, i] = y[:, i] * (self.env.st_pos - (-self.env.st_pos)) + (-self.env.st_pos)

        y[:, 2] = y[:, 2] * (self.env.theta_init - (-self.env.theta_init)) + (-self.env.theta_init)

        for j in range(2):
            y[:, j+3] = y[:, j+3] * (self.env.vel_init - (-self.env.vel_init)) + (-self.env.vel_init)

        y[:, 5] = y[:, 5] * (self.env.omega_init - (-self.env.omega_init)) + (-self.env.omega_init)


        # # close to goal
        # z = torch.Tensor(int(self.num_points*0.05), self.dim).uniform_(
        #     0.0, 1.0
        # )
        # # Small position ±0.15m
        # z[:, 0:2] = z[:, 0:2] * 0.3 - 0.15
        # # Small theta ±0.15 rad
        # z[:, 2] = z[:, 2] * 0.3 - 0.15
        # # Small velocity ±0.3 m/s
        # z[:, 3:5] = z[:, 3:5] * 0.6 - 0.3
        # # Small omega ±0.3 rad/s
        # z[:, 5] = z[:, 5] * 0.6 - 0.3

        x = torch.cat((x,y))
        
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

        torch.save(x_train, self.train_file_name)

        x = torch.Tensor(self.num_points*4//50, self.dim).uniform_(
            0.0, 1.0
        )

        for i in range(self.dim):
            min_val, max_val = self.ranges[i]
            x[:, i] = x[:, i] * (max_val - min_val) + min_val

        y = torch.Tensor(int(self.num_points//50), self.dim).uniform_(
            0.0, 1.0
        )

        for i in range(2):
            y[:, i] = y[:, i] * (self.env.st_pos - (-self.env.st_pos)) + (-self.env.st_pos)

        y[:, 2] = y[:, 2] * (self.env.theta_init - (-self.env.theta_init)) + (-self.env.theta_init)

        for j in range(2):
            y[:, j+3] = y[:, j+3] * (self.env.vel_init - (-self.env.vel_init)) + (-self.env.vel_init)

        y[:, 5] = y[:, 5] * (self.env.omega_init - (-self.env.omega_init)) + (-self.env.omega_init)

        #  # close to goal
        # z = torch.Tensor(int(self.num_points*0.005), self.dim).uniform_(
        #     0.0, 1.0
        # )
        # # Small position ±0.15m
        # z[:, 0:2] = z[:, 0:2] * 0.3 - 0.15
        # # Small theta ±0.15 rad
        # z[:, 2] = z[:, 2] * 0.3 - 0.15
        # # Small velocity ±0.3 m/s
        # z[:, 3:5] = z[:, 3:5] * 0.6 - 0.3
        # # Small omega ±0.3 rad/s
        # z[:, 5] = z[:, 5] * 0.6 - 0.3

        x = torch.cat((x,y))

        random_indices = torch.randperm(len(x))

        x_val = x[random_indices]

        torch.save(x_val, self.val_file_name)

    def setup(self, stage=None):
        self.x_train = torch.load(
            self.train_file_name, weights_only=False)
        self.x_val = torch.load(
            self.val_file_name, weights_only=False)

        self.training_data = TensorDataset(
            self.x_train,
            self.env.init_mask(self.x_train),
            self.env.goal_mask(self.x_train),
            self.env.nongoal_mask(self.x_train),
            self.env.safe_mask(self.x_train),
            self.env.unsafe_mask(self.x_train)
        )
        self.validation_data = TensorDataset(
            self.x_val,
            self.env.init_mask(self.x_val),
            self.env.goal_mask(self.x_val),
            self.env.nongoal_mask(self.x_val),
            self.env.safe_mask(self.x_val),
            self.env.unsafe_mask(self.x_val)
        )

    
    def add_data(self):
        """Adding data -- nothing to do here"""
        pass

    def train_dataloader(self):
        """Make the DataLoader for training data"""
        return DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            num_workers=10,
        )
    
    def val_dataloader(self):
        """Make the DataLoader for validation data"""
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            num_workers=10,
        )
    

class Trainer(pl.LightningModule):
    def __init__(self, model, V, controller, datamodule, out_model, out_controller, out_timing, threshold, primal_learning_rate = 1e-3, goalfactor = 1, decreasefactor = 10, nongoalfactor = 1, goaleps=1e-4, nongoaleps=1e-4, descenteps=1e-3, safe_level = 1, safe_factor = 1, unsafe_factor = 1e2,  eps = 1e-5):
        super().__init__()
        self.automatic_optimization = False
        self.epoch = 0

        self.model = model
        self.V = V
        self.controller = controller
        self.datamodule = datamodule
        self.primal_learning_rate = primal_learning_rate
        self.goalfactor = goalfactor
        self.decreasefactor = decreasefactor
        self.nongoalfactor = nongoalfactor
        self.descenteps = descenteps
        self.goaleps = goaleps
        self.nongoaleps = nongoaleps
        self.safe_level = safe_level
        self.safe_factor = safe_factor
        self.unsafe_factor = unsafe_factor
        self.eps = eps
        self.threshold = threshold
        self.init_val = 0

        self.out_model = out_model
        self.out_controller = out_controller
        self.out_timing = out_timing

        self.lcs = []
        self.losses_train = []
        self.losses_val = []

        self.init_losses_train = []
        self.descent_losses_train = []
        self.goal_losses_train = []

        self.init_losses_val = []
        self.noninit_losses_val = []
        self.descent_losses_val = []
        self.goal_losses_val = []
        self.lc = 0
        self.prev_lc = None
        self.fluctuation_count = 0

    def prepare_data(self):
        return self.datamodule.prepare_data()

    def setup(self, stage = None):
        return self.datamodule.setup(stage)

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()


    def init_loss(self, x, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask):
        # for init state, V(x) should be less than safe_level
        V = self.V(x)

        V_init_nongoal = V[init_mask & nongoal_mask]
        # print(V_init_nongoal[:30])
        # print(x[init_mask & nongoal_mask][:30])
        if len(V_init_nongoal) == 0:
            safe_term = 0
        else:
            safe_violation = F.relu(
                self.eps + V_init_nongoal - self.safe_level)
            #valid_violation = F.relu(self.eps - V_init_nongoal)
            safe_term = self.safe_factor * \
                (safe_violation.mean())

        return safe_term
    
    # def goal_loss(self, x, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask):
    #     V = self.V(x)
        
    #     # Only apply to states in goal region
    #     V_goal = V[goal_mask]
        
    #     if len(V_goal) == 0:
    #         goal_term = 0.0
    #     else:
    #         # Push V ≤ -0.05 in goal (strong negative for clear sublevel set)
    #         goal_violation = F.relu(V_goal + self.eps)
    #         goal_term = self.goalfactor * goal_violation.mean()
        
    #     return goal_term
    
    # def maximize_in_neighborhood(self, x0, epsilon, alpha=0.001, steps=30, norm="linf"):
    #     x = x0.detach()
    #     x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    #     for i in range(steps):
    #         x.requires_grad_()
    #         with torch.enable_grad():
    #             V_value = self.V(x)
    #             loss = V_value.sum()
    #         grad = torch.autograd.grad(loss, [x])[0]
    #         x = x.detach() + alpha * torch.sign(grad.detach())
    #         x = torch.min(torch.max(x, x0 - epsilon), x0 + epsilon)
    #         x = torch.clamp(x, -0.3, 0.3)
    #     return x


    def descent_loss(self, x, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask):

        x = x[nongoal_mask & safe_mask]
        unsafe_mask = self.model.unsafe_mask(x)
        out_space_mask = ~self.model.space_mask(x)
        if len(x) == 0:
            descent_term = 0
        else:
            V_nongoal = self.V(x)
            V_nongoal[unsafe_mask] = 1.2
            V_nongoal[out_space_mask] = 1.2
            condition_original = (
                V_nongoal <= self.safe_level)  # eq5: V <= \beta
            condition_original = condition_original.float()

            x_next = self.controller.next_step(x)

            unsafe_mask = self.model.unsafe_mask(x_next)
            goal_mask = self.model.goal_mask(x_next)
            out_space_mask = ~self.model.space_mask(x_next)
            V_next = self.V(x_next)
            V_next[unsafe_mask] = 1.2
            V_next[goal_mask] = -10
            V_next[out_space_mask] = 1.2
            # modify ----------------------
            # d = 0.005
            # x_perturbed = self.maximize_in_neighborhood(x_next, d)

            # perturbations_region = torch.tensor([
            #     [d, d], [d, -d], [-d, d], [-d, -d]
            # ], device='cuda:1')  # Shape: (4, 2)

            # x_perturbed_region = x_next.unsqueeze(
            #     1) + perturbations_region.unsqueeze(0)  # Shape: (n, 4, 2)
            # x_perturbed_region = x_perturbed_region.view(-1, 2)
            # # print(x_perturbed_region[:30])
            # unsafe_safe_region = self.model.unsafe_mask(
            #     x_perturbed_region) | ~self.model.space_mask(x_perturbed_region)  # Shape: (n*4,)
            # unsafe_safe_region = unsafe_safe_region.view(-1, 4)
            # # print(torch.cat((x,x_next, x_perturbed, unsafe_safe_region), dim=1)[:30])
            # unsafe_safe_region = torch.any(unsafe_safe_region, dim=1)

            # unsafe_mask = self.model.unsafe_mask(x_perturbed)
            # goal_mask = self.model.goal_mask_smaller(x_perturbed)
            # out_space_mask = ~self.model.space_mask(x_perturbed)
            # V_next = self.V(x_perturbed)
            # V_next[unsafe_mask] = 1.2
            # V_next[goal_mask] = -10
            # V_next[out_space_mask] = 1.2
            # V_next[unsafe_safe_region] = 1.2

            # d = 0.01
            # num_samples = 10
            # # l-inf
            # x_perturbed = torch.empty_like(x_next.unsqueeze(
            #     1).expand(-1, num_samples, -1)).uniform_(-d, d)
            # x_perturbed += x_next.unsqueeze(1)
            # x_perturbed = x_perturbed.reshape(-1, x_next.shape[-1])

            # l-1
            # x_perturbed = torch.randn_like(x_next.unsqueeze(
            #     1).expand(-1, num_samples, -1))
            # x_perturbed = x_perturbed / \
            #     torch.norm(x_perturbed, p=1, dim=2, keepdim=True)
            # x_perturbed = (x_next.unsqueeze(1) + x_perturbed * d).reshape(
            #     -1, x_next.shape[-1])

            # V_next_perturbed = self.V(x_perturbed)
            # unsafe_mask = self.model.unsafe_mask(x_perturbed)
            # goal_mask = self.model.goal_mask(x_perturbed)
            # V_next_perturbed[unsafe_mask] = 1.2
            # V_next_perturbed[goal_mask] = -10
            # V_next = V_next_perturbed

            #######

            # V_next_perturbed = V_next_perturbed.view(-1, num_samples, 1)
            # V_max, _ = V_next_perturbed.max(dim=1, keepdim=True)
            # V_next_perturbed_max = V_max.expand(-1,
            #                                     num_samples, -1).reshape(-1, 1)
            #######

            # V_nongoal_expanded = V_nongoal.expand(-1,
            #                                       num_samples).reshape(-1, 1)
            # condition_original = (V_nongoal_expanded <=
            #                       self.safe_level).float()

            # V_next_perturbed = V_next_perturbed.view(x.shape[0], num_samples)
            # V_next, _ = V_next_perturbed.max(dim=1, keepdim=True)

            # l-inf lip

            # ---------------------- modify
            # d = 1e-3
            # _, lc = self.compute_lipschitz_constant("global")
            descent_violation = F.relu(
                self.descenteps + (V_next - V_nongoal) )
            #######
            # V_next[descent_violation >
            #        0] = V_next_perturbed_max[descent_violation > 0]
            # descent_violation = F.relu(
            #     self.descenteps + (V_next - V_nongoal_expanded)/(self.controller.t))
            #######
            descent_term = self.decreasefactor * \
                (descent_violation * condition_original).mean()
            # print(V_next[~goal_mask].min())
            large_descent_term = (
                (V_next-V_nongoal) * condition_original)[~goal_mask].mean()

        return descent_term
    
    # def l2_reg_loss(self, l2_reg_upper=20):
    #     reg_loss = None
    #     for param in self.V.parameters():
    #         if reg_loss is None:
    #             reg_loss = torch.sum(param**2)
    #         else:
    #             reg_loss = reg_loss + param.norm(2)**2
    #     return 0.0005*max(reg_loss-l2_reg_upper, 0)
    
    # def global_lipschitz_loss(self, global_lip_upper=3):
    #     L2_product = 1
    #     for param in self.V.parameters():  # Iterate directly over parameters
    #         if param.ndim > 1:  # Only process weight matrices (ignore biases)
    #             # Use differentiable SVD
    #             U, S, V = torch.linalg.svd(param, full_matrices=False)
    #             spectral_norm = S[0]  # Largest singular value (spectral norm)
    #             L2_product *= spectral_norm
    #     return max(L2_product-global_lip_upper, 0)

    # def global_lipschitz_calculate(self):
    #     L2_product = 1.0
    #     # Disable gradient tracking for this computation
    #     with torch.no_grad():
    #         for param in self.V.parameters():
    #             # Process weight matrices only (ignore biases)
    #             if param.ndim > 1:
    #                 U, S, V = torch.linalg.svd(param, full_matrices=False)
    #                 # Extract largest singular value as a float
    #                 spectral_norm = S[0].item()
    #                 L2_product *= spectral_norm
    #     return L2_product
    
    # def l2_reg_calculate(self):
    #     l2_sum = 0.0
    #     with torch.no_grad():  # Disable gradients if you don't need them
    #         for param in self.V.parameters():
    #             # Sum of squared weights
    #             l2_sum += torch.sum(param ** 2).item()
    #     return l2_sum
    
    # def step_length_loss(self, x):
    #     safe_mask = self.model.safe_mask(x)
    #     nongoal_mask = self.model.nongoal_mask(x)
    #     x = x[safe_mask & nongoal_mask]
    #     x_next = self.controller.next_step(x)
    #     step_length = torch.max(
    #         torch.abs(torch.abs(x_next) - torch.abs(x)), dim=1).values
    #     sl_term = F.relu(0.011-step_length.min())
    #     return sl_term


    def training_step(self, batch, batch_idx):
        x, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask = batch
        #_, lc = self.compute_lipschitz_constant("global")
        torch.set_grad_enabled(True)
        # self.goal_factor = 1
        # self.decreasefactor = 1
        # self.safe_factor = 1
        # if self.epoch <= 5:  # vallina: 2; lip: 4
        #     self.safe_factor = 1e2
        #     self.decreasefactor = 1
        # else:
        #     self.safe_factor = 1
        #     self.decreasefactor = 1e2
        init_term = self.init_loss(
            x, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask)
        # goal_term = self.goal_loss(
        #     x, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask)
        descent_term = self.descent_loss(
            x, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask)


        goal_term = 0.0
        l2_reg_term = 0.0
        lip_term = 0.0
        sl_term = 0.0
        #lip_term = self.global_lipschitz_loss()
        # lip_term = self.local_lipschitz_loss(
        #     self.controller.next_step(x[nongoal_mask & safe_mask]))
        # l2_reg_term = self.l2_reg_loss()
        # sl_term = self.step_length_loss(x)
        total_loss = descent_term + init_term + goal_term + lip_term + l2_reg_term + sl_term
        # if batch_idx % 100 == 0:
        #     lip_term = self.global_lipschitz_calculate()
        #     l2_reg_term = self.l2_reg_calculate()
        #     print(lip_term, l2_reg_term, init_term,
        #           descent_term)
        #     print(total_loss)

        opt_v, opt_c, opt_a = self.optimizers()

        if total_loss > 0:
            if self.epoch <= 5:  # vallina: 1; lip-neighbor: 10
                opt_v.zero_grad()
                self.manual_backward(total_loss)
                opt_v.step()
            elif 5 <= self.epoch <= 10:  # vallina: 2,3; lip-neighbor: 11,15
                opt_c.zero_grad()
                self.manual_backward(descent_term)
                opt_c.step()
            else:
                opt_a.zero_grad()
                self.manual_backward(total_loss)
                opt_a.step()

        # opt_a.zero_grad()
        # self.manual_backward(total_loss)
        # opt_a.step()

        batch_dict = {"loss": total_loss}
        self.losses_train.append(total_loss)
        #self.lcs.append(lc)
        self.init_losses_train.append(init_term)
        #self.goal_losses_train.append(goal_term)
        self.descent_losses_train.append(descent_term)
        return batch_dict

    def validation_step(self, batch, batch_idx):
        x, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask = batch
        #_, lc = self.compute_lipschitz_constant("global")
        init_term = self.init_loss(
            x, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask)
        # goal_term = self.goal_loss(
        #     x, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask)
        descent_term = self.descent_loss(
            x, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask)
        
        goal_term = 0

        total_loss = descent_term + init_term + goal_term

        batch_dict = {"loss": total_loss}

        self.losses_val.append(total_loss)
        self.init_losses_val.append(init_term)
        self.descent_losses_val.append(descent_term)
        #self.goal_losses_val.append(goal_term)
        return batch_dict
    
    def on_train_epoch_end(self):
        self.epoch += 1

        total = 0
        init = 0
        descent = 0
        #goal = 0
        # gross = 0
        for i in range(len(self.losses_train)):
            total += self.losses_train[i]
            init += self.init_losses_train[i]
            descent += self.descent_losses_train[i]
            #goal += self.goal_losses_train[i]

        print("Current loss: ", total/len(self.losses_train))
        print("Init loss: ", init/len(self.losses_train))
        print("Descent loss: ", descent/len(self.losses_train))
        #print("Goal loss: ", goal/len(self.losses_train))
        print("Total loss:", total)
        # self.log("step_loss", descent/len(self.losses_train))
        # torch.save(self.V, self.out_model)
        # torch.save(self.controller.nn, self.out_controller)
        #lc = self.lcs[-1]
        with open(self.out_timing, 'a') as f:
            f.write("total loss: " + str(total) + " "  + "\n")
        # print("Gross: ", gross/len(self.losses_train))

        # if self.epoch!=0 and self.epoch%30==0:
        #     torch.save(self.V, self.out_model)
        #     torch.save(self.controller.nn, self.out_controller)
        #     self.log("saved_loss", torch.tensor(0, dtype=torch.float32))

       # lc = self.lcs[-1]
        if (total.item() <= self.threshold):
            #     print("total loss is 0")
            #     with open('log_d001.txt', 'a') as f:
            #         f.write("lc: " + str(lc) + " " + str(self.prev_lc) + "\n")
            #     if self.prev_lc is not None:
            #         lc_fluctuation = abs(lc - self.prev_lc)
            #         if lc_fluctuation <= 0.01:
            #             self.fluctuation_count += 1
            #         else:
            #             print(self.prev_lc, lc)
            #             self.fluctuation_count = 0  # Reset if condition fails

            #     self.prev_lc = lc

            # if self.fluctuation_count >= 3:
            torch.save(self.V, self.out_model)
            torch.save(self.controller.nn, self.out_controller)
            self.log("saved_loss", torch.tensor(0, dtype=torch.float32))

            # x = torch.randn(10,4,requires_grad=True)

            # torch.onnx.export(self.V,x,"cur_model_20n.onnx",export_params=True,opset_version=10,do_constant_folding=True,input_names = ['input'],output_names = ['output'])
            # torch.onnx.export(self.controller.nn,x,"cur_controller_20n.onnx",export_params=True,opset_version=10,do_constant_folding=True,input_names = ['input'],output_names = ['output'])
            print("saved!")
        else:
            self.init_val += 1
            self.log("saved_loss", torch.tensor(
                self.init_val, dtype=torch.float32))

        self.losses_train = []
        self.init_losses_train = []
        self.descent_losses_train = []
        #self.goal_losses_train = []
       # self.lcs = []

    # def compute_lipschitz_constant(self, type="global", x=None, r=None):
    #     """Compute the Lipschitz constant based on your criteria."""
    #     if type == "global":
    #         L2_product = torch.tensor(1.0)
    #         for param in self.V.parameters():  # Iterate directly over parameters
    #             # Only process weight matrices (ignore biases)
    #             if param.ndim > 1:
    #                 # Differentiable SVD
    #                 U, S, V = torch.linalg.svd(param, full_matrices=False)
    #                 # Largest singular value (spectral norm)
    #                 spectral_norm = S[0]
    #                 L2_product = L2_product * spectral_norm
    #         return "global", L2_product
    #     elif type == "local":
    #         num_samples = 100
    #         n, d = x.shape

    #         noise = torch.empty(n, num_samples, d).uniform_(-1, 1)
    #         noise /= torch.amax(torch.abs(noise), dim=-1,
    #                             keepdim=True)
    #         x_prime = x[:, None, :] + noise * r
    #         f_x = self.V(x[:, None, :])
    #         f_x_prime = self.V(x_prime)
    #         diff = torch.amax(torch.abs(f_x - f_x_prime),
    #                           dim=-1)  # (n, num_samples)
    #         denom = torch.amax(torch.abs(noise * r), dim=-1)

    #         lipschitz_ratios = diff / denom

    #         return "local", lipschitz_ratios
    
    def on_validation_epoch_end(self):
        # generate new data and also plot before (TODO)
        total = 0
        init = 0
        descent = 0
        #goal = 0
        for i in range(len(self.losses_val)):
            total += self.losses_val[i]
            init += self.init_losses_val[i]
            descent += self.descent_losses_val[i]
            #goal += self.goal_losses_val[i]

        print("Current loss: ", total/len(self.losses_val))
        print("Init loss: ", init/len(self.losses_val))
        print("Descent loss: ", descent/len(self.losses_val))
       # print("Goal loss: ", goal/len(self.losses_val))

        self.losses_val = []
        self.init_losses_val = []
        self.descent_losses_val = []
        self.goal_losses_val = []


    def configure_optimizers(self):
        optimizer_V = torch.optim.Adam(
            list(self.V.parameters()), lr=self.primal_learning_rate)

        optimizer_controller = torch.optim.Adam(
            list(self.controller.nn.parameters()), lr=self.primal_learning_rate)

        optimizer_together = torch.optim.Adam(
            list(self.V.parameters()) + list(self.controller.nn.parameters()), lr=self.primal_learning_rate)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_together, mode='min', factor=0.1, patience=10
        )
        return [optimizer_V, optimizer_controller, optimizer_together], [{"scheduler": scheduler, "monitor": "loss"}]

class SampleDataRetrain(pl.LightningDataModule):
    def __init__(self, epoch, two_dim_docking, V, controller, datapoints, counterexample_ranges, in_train_file, in_traj_file, train_file, val_file, traj_file, counterexample_folder, num_dpoints, num_points=10000000, dim=2, val_split=0.1, batch_size=10000, max_tries=5000, safe_level=1, is_traj=False):
        super().__init__()
        self.epoch = epoch
        self.num_points = num_points
        self.env = two_dim_docking

        self.V = V
        self.controller = controller
        self.safe_level = safe_level
        self.dim = dim
        self.val_split = val_split
        self.batch_size = batch_size
        self.max_tries = max_tries
        self.ranges = []
        self.counterexamples = datapoints
        self.counterexample_ranges = counterexample_ranges
        self.num_dpoints = num_dpoints

        self.in_train_file = in_train_file
        self.in_traj_file = in_traj_file
        self.train_file = train_file
        self.val_file = val_file
        self.traj_file = traj_file
        self.counterexample_file = os.path.join(
            counterexample_folder, "counterexamples.pt")
        self.is_traj = is_traj

        for _ in range(2):
            self.ranges.append([-self.env.unsafe_pos-0.02, self.env.unsafe_pos+0.02])

        self.ranges.append([-self.env.theta_limit-0.02, self.env.theta_limit+0.02])

        for _ in range(2):
            #ranges.append([-self.vel_limit,self.vel_limit])
            self.ranges.append([-self.env.vel_limit-0.02,self. env.vel_limit+0.02])

        self.ranges.append([-self.env.omega_limit-0.02, self.env.omega_limit+0.02])
        

    def prepare_data(self):
        x_train = torch.load(self.in_train_file, weights_only=False)

        # if self.is_traj:
        #     num_samples = 100000

        #     x = torch.Tensor(
        #         num_samples, 2).uniform_(-0.3, 0.3)
        #     safe_mask = abs(x[:, 0]) >= 0.21
        #     safe_mask.logical_or_(abs(x[:, 1]) >= 0.21)
        #     x = x[safe_mask]
        #     num_samples = len(x)
        #     steps = torch.zeros(num_samples, dtype=torch.int32)
        #     active = torch.ones(num_samples, dtype=torch.bool)
        #     all_x = []
        #     while active.any() and max(steps) < 200:
        #         if max(steps) % 10 == 0:
        #             print(max(steps))
        #         x_next = self.controller.next_step(x)
        #         new_reached_goal = (x_next[:, 0].abs() <= 0.2 + 0.011) & (
        #             x_next[:, 1].abs() <= 0.2 + 0.011)
        #         steps[active & new_reached_goal] += 1
        #         active &= ~new_reached_goal
        #         all_x.append(x_next[active].clone())
        #         x = x_next
        #         steps[active] += 1
        #     x_traj = torch.cat(all_x, dim=0)
        #     random_indices = torch.randperm(len(x_traj))
        #     x_traj = x_traj[random_indices]
        #     torch.save(x_traj.detach(), self.traj_file)

        if len(self.counterexamples):
            num_added = self.num_dpoints

            x_counterexamples = torch.Tensor(num_added, self.dim).uniform_(
                0.0, 1.0
            )
            for i in range(self.dim):
                min_val, max_val = self.ranges[i]
                x_counterexamples[:, i] = x_counterexamples[:,
                                                            i] * (max_val - min_val) + min_val
            if self.epoch > 0:
                x_original = torch.load(
                    self.counterexample_file, weights_only=False)
                x_counterexamples = torch.cat((x_counterexamples, x_original))

            for i in range(len(self.counterexamples)):

                x_counterexamples = torch.cat(
                    (x_counterexamples, torch.unsqueeze(torch.Tensor(self.counterexamples[i]), 0)))

                new_points = torch.Tensor(
                    num_added//len(self.counterexamples), self.dim).uniform_(0.0, 1.0)
                
                for j in range(2):
                    new_points[:, j] = new_points[:, j] * (0.04) + self.counterexamples[i][j] - 0.02

                new_points[:, 2] = new_points[:, 2] * (0.04) + self.counterexamples[i][j] - 0.02

                for j in range(2):
                    new_points[:, j+3] = new_points[:, j+3] * (0.04) + self.counterexamples[i][j+2] - 0.02
                
                new_points[:, 5] = new_points[:, 5] * (0.04) + self.counterexamples[i][j] - 0.02 


                # x_train = torch.cat((x_train,new_points))
                # new_points在counterexamples周围
                x_counterexamples = torch.cat((x_counterexamples, new_points))

            for i in range(len(self.counterexample_ranges)):
                new_points = torch.Tensor(
                    num_added//len(self.counterexample_ranges), self.dim).uniform_(0.0, 1.0)
                for j in range(6):
                    new_points[:, j] = self.counterexample_ranges[i][j][0] + (
                        self.counterexample_ranges[i][j][1] - self.counterexample_ranges[i][j][0]) * new_points[:, j]  # new_points在counterexamples range里

                # x_train = torch.cat((x_train,new_points))
                x_counterexamples = torch.cat((x_counterexamples, new_points))

            # random_indices = torch.randperm(num_train)
            # x_train = x_train[random_indices]
            # torch.save(x_train, self.train_file)

            random_indices = torch.randperm(len(x_counterexamples))
            x_counterexamples = x_counterexamples[random_indices]
            torch.save(x_counterexamples, self.counterexample_file)

    def add_data(self):
        pass

    def setup(self, stage=None):
        self.x_train = torch.load(self.train_file, weights_only=False)

        self.x_val = torch.load(self.val_file, weights_only=False)

        len_train = len(self.x_train)
        len_val = len(self.x_val)
        if self.is_traj:
            self.x_traj = torch.load(self.traj_file, weights_only=False)
            len_traj = len(self.x_traj)

            traj_flags_train = torch.cat((
                torch.zeros(len_train, dtype=torch.float32),
                torch.ones(
                    len(self.x_traj[len_traj//50:]), dtype=torch.float32)
            ))

            traj_flags_val = torch.cat((
                # original reused in val
                torch.zeros(len_val, dtype=torch.float32),
                torch.ones(len_traj//50, dtype=torch.float32)
            ))
            # merge train and traj into train
            self.x_train = torch.cat(
                (self.x_train, self.x_traj[len_traj//50:]))
            self.x_val = torch.cat((self.x_val, self.x_traj[:len_traj//50]))
            len_train = len(self.x_train)
            len_val = len(self.x_val)
        else:
            traj_flags_train = torch.zeros(len_train, dtype=torch.float32)

            traj_flags_val = torch.zeros(len_val, dtype=torch.float32)

        if os.path.exists(self.counterexample_file):
            self.x_counterexamples = torch.load(
                self.counterexample_file, weights_only=False)
            len_counterexamples = len(self.x_counterexamples)

        # --------- modify

            self.x_train = torch.cat(
                (self.x_train, self.x_counterexamples[len_counterexamples//50:]))
            self.x_val = torch.cat(
                (self.x_val, self.x_counterexamples[:len_counterexamples//50]))

            counterexamples_flags_train = torch.cat((
                torch.zeros(len_train, dtype=torch.float32),
                torch.ones(
                    len(self.x_counterexamples[len_counterexamples//50:]), dtype=torch.float32)
            ))

            counterexamples_flags_val = torch.cat((
                # original reused in val
                torch.zeros(len_val, dtype=torch.float32),
                torch.ones(len_counterexamples//50, dtype=torch.float32)
            ))

            traj_flags_train = torch.cat((traj_flags_train, torch.zeros(
                len(self.x_counterexamples[len_counterexamples//50:]))))
            traj_flags_val = torch.cat(
                (traj_flags_val, torch.zeros(len_counterexamples//50)))
        else:
            # consistency
            counterexamples_flags_train = torch.cat((
                torch.zeros(len_train, dtype=torch.float32),
                torch.ones(
                    0, dtype=torch.float32)
            ))

            counterexamples_flags_val = torch.cat((
                # original reused in val
                torch.zeros(len_val, dtype=torch.float32),
                torch.ones(0, dtype=torch.float32)
            ))
        self.training_data = TensorDataset(
            self.x_train,
            self.env.init_mask(self.x_train),
            self.env.goal_mask(self.x_train),
            self.env.nongoal_mask(self.x_train),
            self.env.safe_mask(self.x_train),
            self.env.unsafe_mask(self.x_train),
            traj_flags_train,
            counterexamples_flags_train
        )
        self.validation_data = TensorDataset(
            self.x_val,
            self.env.init_mask(self.x_val),
            self.env.goal_mask(self.x_val),
            self.env.nongoal_mask(self.x_val),
            self.env.safe_mask(self.x_val),
            self.env.unsafe_mask(self.x_val),
            traj_flags_val,
            counterexamples_flags_val
        )
        # modify ----------

        # random_indices = torch.randperm(len_train)
        # x_train = self.x_train[random_indices]
        # self.x_val = x_train[:len_counterexamples]

        # random_indices = torch.randperm(len_train)
        # x_train = self.x_train[random_indices]
        # self.x_train = x_train[:len_counterexamples]

        # self.training_data = TensorDataset(
        #     self.x_train,
        #     self.x_counterexamples,
        #     self.env.init_mask(self.x_train),
        #     self.env.goal_mask(self.x_train),
        #     self.env.nongoal_mask(self.x_train),
        #     self.env.safe_mask(self.x_train),
        #     self.env.unsafe_mask(self.x_train),
        #     self.env.init_mask(self.x_counterexamples),
        #     self.env.goal_mask(self.x_counterexamples),
        #     self.env.nongoal_mask(self.x_counterexamples),
        #     self.env.safe_mask(self.x_counterexamples),
        #     self.env.unsafe_mask(self.x_counterexamples)
        # )
        # self.validation_data = TensorDataset(
        #     self.x_val,
        #     self.x_counterexamples,
        #     self.env.init_mask(self.x_val),
        #     self.env.goal_mask(self.x_val),
        #     self.env.nongoal_mask(self.x_val),
        #     self.env.safe_mask(self.x_val),
        #     self.env.unsafe_mask(self.x_val),
        #     self.env.init_mask(self.x_counterexamples),
        #     self.env.goal_mask(self.x_counterexamples),
        #     self.env.nongoal_mask(self.x_counterexamples),
        #     self.env.safe_mask(self.x_counterexamples),
        #     self.env.unsafe_mask(self.x_counterexamples)
        # )

    def train_dataloader(self):
        """Make the DataLoader for training data"""
        return DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True
        )

    def val_dataloader(self):
        """Make the DataLoader for validation data"""
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True
        )


class TrainerRetrain(pl.LightningModule):
    def __init__(self, model, V, controller, datamodule, out_model, out_controller, out_timing, threshold, primal_learning_rate=1e-4, goalfactor=1, decreasefactor=1e1, nongoalfactor=1, goaleps=1e-4, nongoaleps=1e-4, descenteps=1e-3, safe_level=1, safe_factor=1, unsafe_factor=1e2, eps=1e-5):
        super().__init__()
        self.epoch = 0
        self.automatic_optimization = False

        self.model = model
        self.V = V
        self.controller = controller
        self.datamodule = datamodule
        self.primal_learning_rate = primal_learning_rate
        self.goalfactor = goalfactor
        self.decreasefactor = decreasefactor
        self.nongoalfactor = nongoalfactor
        self.descenteps = descenteps
        self.goaleps = goaleps
        self.nongoaleps = nongoaleps
        self.safe_level = safe_level
        self.safe_factor = safe_factor
        self.unsafe_factor = unsafe_factor
        self.eps = eps
        self.threshold = threshold
        self.init_val = 0

        self.out_model = out_model
        self.out_controller = out_controller
        self.out_timing = out_timing
        self.losses_train = []
        self.losses_val = []

        self.init_losses_train = []
        self.goal_acc_train = []
        # self.safe_losses_train = []
        # self.safe_acc_train = []
        # self.unsafe_losses_train = []
        # self.unsafe_acc_train = []
        self.noninit_losses_train = []
        self.nongoal_acc_train = []
        self.goal_losses_train = []
        # self.gross_viol_train = []

        self.init_losses_val = []
        self.goal_acc_val = []
        # self.safe_losses_val = []
        # self.safe_acc_val = []
        # self.unsafe_losses_val = []
        # self.unsafe_acc_val = []
        self.noninit_losses_val = []
        self.nongoal_acc_val = []
        self.descent_losses_val = []
        self.descent_acc_val = []
        self.goal_losses_val = []
        self.goal_acc_val = []
        # self.gross_viol_val = []
        self.Lip = 0
        self.is_traj = False

    def prepare_data(self):
        return self.datamodule.prepare_data()

    def setup(self, stage=None):
        return self.datamodule.setup(stage)

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    def init_loss(self, x, x_c, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask, init_mask_c, goal_mask_c, nongoal_mask_c, safe_mask_c, unsafe_mask_c):

        V = self.V(x)
        # V[goal_mask] = -0.1
        '''
        V_init_nongoal = V[init_mask & nongoal_mask]
        # print(V_init_nongoal[:30])
        # print(x[init_mask & nongoal_mask][:30])
        if len(V_init_nongoal) == 0:
            safe_term = 0
        else:
            safe_violation = F.relu(
                self.eps + V_init_nongoal - self.safe_level)
            valid_violation = F.relu(self.eps - V_init_nongoal)
            safe_term = self.safe_factor * \
                (safe_violation.mean() + valid_violation.mean())

        return safe_term
        '''
        V_init_nongoal = V[init_mask & nongoal_mask]
        if len(V_init_nongoal) == 0:
            safe_violation, safe_term = 0, 0
        else:
            safe_violation = F.relu(
                self.eps + (V_init_nongoal - self.safe_level))
            safe_term = self.safe_factor * safe_violation.mean()
        if len(x_c) != 0:
            V_c = self.V(x_c)
            x_c = x_c[init_mask_c & nongoal_mask_c]

            V_init_nongoal_c = V_c[init_mask_c & nongoal_mask_c]
            if len(V_init_nongoal_c) == 0:
                safe_violation_c, safe_term_c = 0, 0
            else:
                safe_violation_c = F.relu(
                    self.eps + V_init_nongoal_c - self.safe_level)
                safe_term_c = (self.safe_factor * 100 *
                               safe_violation_c).mean()

            safe_term += safe_term_c

        return safe_term
    
    # def goal_loss(self, x, x_c, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask, init_mask_c, goal_mask_c, nongoal_mask_c, safe_mask_c, unsafe_mask_c):
    #     V = self.V(x)
        
    #     # Only apply to states in goal region
    #     V_goal = V[goal_mask]
        
    #     if len(V_goal) == 0:
    #         goal_term = 0.0
    #     else:
    #         # Push V ≤ -0.05 in goal (strong negative for clear sublevel set)
    #         goal_violation = F.relu(V_goal + self.eps)
    #         goal_term = self.goalfactor * goal_violation.mean()
        
    #     if len(x_c) != 0:
    #         V_c = self.V(x_c)
    #         V_goal_c = V_c[goal_mask_c]
    #         if len(V_goal_c) > 0:
    #             goal_violation_c = F.relu(V_goal_c + self.eps)
    #             goal_term_c = self.goalfactor * 100 * goal_violation_c.mean()
    #             goal_term += goal_term_c

    #     return goal_term

    # def maximize_in_neighborhood(self, x0, epsilon, alpha=0.001, steps=30, norm="linf"):
    #     x = x0.detach()
    #     x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    #     for i in range(steps):
    #         x.requires_grad_()
    #         with torch.enable_grad():
    #             V_value = self.V(x)
    #             loss = V_value.sum()
    #         grad = torch.autograd.grad(loss, [x])[0]
    #         x = x.detach() + alpha * torch.sign(grad.detach())
    #         x = torch.min(torch.max(x, x0 - epsilon), x0 + epsilon)
    #         x = torch.clamp(x, -0.5, 0.5)
    #     return x

    def descent_loss(self, x, x_c, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask, init_mask_c, goal_mask_c, nongoal_mask_c, safe_mask_c, unsafe_mask_c, traj_mask):
        x = x[nongoal_mask & safe_mask]
        traj_mask = traj_mask[nongoal_mask & safe_mask]
        x_t = x[traj_mask]
        if len(x) == 0:
            descent_term = 0
        else:
            unsafe_mask = self.model.unsafe_mask(x)
            out_space_mask = ~self.model.space_mask(x)
            V_nongoal = self.V(x)
            V_nongoal[unsafe_mask] = 1.2
            V_nongoal[out_space_mask] = 1.2
            # condition_active = torch.sigmoid(10 * (self.safe_level + self.goaleps - V))
            condition_original = (V_nongoal <= self.safe_level)
            # condition_original = (V_nongoal <= 1.3)
            condition_original = condition_original.float()

            x_next = self.controller.next_step(x)

            unsafe_mask = self.model.unsafe_mask(x_next)
            goal_mask = self.model.goal_mask(x_next)
            out_space_mask = ~self.model.space_mask(x_next)
            V_next = self.V(x_next)
            V_next[unsafe_mask] = 1.2
            V_next[out_space_mask] = 1.2
            V_next[goal_mask] = -10

            # modify ----------------------
            # d = 0.005
            # x_perturbed = self.maximize_in_neighborhood(x_next, d)
            # perturbations_region = torch.tensor([
            #     [d, d], [d, -d], [-d, d], [-d, -d]
            # ], device='cuda:1')  # Shape: (4, 2)

            # x_perturbed_region = x_next.unsqueeze(
            #     1) + perturbations_region.unsqueeze(0)  # Shape: (n, 4, 2)
            # x_perturbed_region = x_perturbed_region.view(-1, 2)
            # # print(x_perturbed_region[:30])
            # unsafe_safe_region = self.model.unsafe_mask(
            #     x_perturbed_region) | ~self.model.space_mask(x_perturbed_region)  # Shape: (n*4,)
            # unsafe_safe_region = unsafe_safe_region.view(-1, 4)
            # # print(torch.cat((x,x_next, x_perturbed, unsafe_safe_region), dim=1)[:30])
            # unsafe_safe_region = torch.any(unsafe_safe_region, dim=1)

            # unsafe_mask = self.model.unsafe_mask(x_perturbed)
            # goal_mask = self.model.goal_mask_smaller(x_perturbed)
            # out_space_mask = ~self.model.space_mask(x_perturbed)
            # V_next = self.V(x_perturbed)
            # V_next[unsafe_mask] = 1.2
            # V_next[goal_mask] = -10
            # V_next[out_space_mask] = 1.2
            # V_next[unsafe_safe_region] = 1.2

            # d = 0.01
            # num_samples = 10
            # # l-inf
            # x_perturbed = torch.empty_like(x_next.unsqueeze(
            #     1).expand(-1, num_samples, -1)).uniform_(-d, d)
            # x_perturbed += x_next.unsqueeze(1)
            # x_perturbed = x_perturbed.reshape(-1, x_next.shape[-1])\

            # l-1
            # x_perturbed = torch.randn_like(x_next.unsqueeze(
            #     1).expand(-1, num_samples, -1))
            # x_perturbed = x_perturbed / \
            #     torch.norm(x_perturbed, p=1, dim=2, keepdim=True)
            # x_perturbed = (x_next.unsqueeze(1) + x_perturbed * d).reshape(
            #     -1, x_next.shape[-1])

            # V_next_perturbed = self.V(x_perturbed)
            # unsafe_mask = self.model.unsafe_mask(x_perturbed)
            # goal_mask = self.model.goal_mask(x_perturbed)
            # V_next_perturbed[unsafe_mask] = 1.2
            # V_next_perturbed[goal_mask] = -10
            # V_next = V_next_perturbed

            #######

            # V_next_perturbed = V_next_perturbed.view(-1, num_samples, 1)
            # V_max, _ = V_next_perturbed.max(dim=1, keepdim=True)
            # V_next_perturbed_max = V_max.expand(-1,
            #                                     num_samples, -1).reshape(-1, 1)
            #######

            # V_nongoal_expanded = V_nongoal.expand(-1,
            #                                       num_samples).reshape(-1, 1)
            # condition_original = (V_nongoal <=
            #                       self.safe_level).float()
            # ---------------------- modify
            # _, lc = self.compute_lipschitz_constant("global")
            # d = 0.001
            descent_violation = F.relu(
                self.descenteps + (V_next - V_nongoal))

            #######
            # V_next[descent_violation >
            #        0] = V_next_perturbed_max[descent_violation > 0]
            # descent_violation = F.relu(
            #     self.descenteps + (V_next - V_nongoal_expanded)/(self.controller.t))
            #######

            descent_term = self.decreasefactor * \
                (descent_violation * condition_original).mean()
            if self.is_traj:
                d = 0.02
                _, lc = self.compute_lipschitz_constant("local", x_t, 0.02)
                V_nongoal_t = V_nongoal[traj_mask]
                V_next_t = V_next[traj_mask]
                condition_original_t = (V_nongoal_t <= self.safe_level)
                condition_original_t = condition_original_t.float()
                descent_violation_tjlip = F.relu(
                    self.descenteps + (V_next_t + lc * d - V_nongoal_t))
                descent_term_tj = self.decreasefactor * \
                    (descent_violation_tjlip * condition_original_t).mean()
                descent_term += descent_term_tj
        if len(x_c) != 0:
            x_c = x_c[nongoal_mask_c & safe_mask_c]
            if len(x_c) == 0:
                descent_term_c = 0
            else:
                unsafe_mask_c = self.model.unsafe_mask(x_c)
                out_space_mask_c = ~self.model.space_mask(x_c)
                V_nongoal_c = self.V(x_c)
                V_nongoal_c[unsafe_mask_c] = 1.2
                V_nongoal_c[out_space_mask_c] = 1.2
                condition_original_c = (V_nongoal_c <= self.safe_level)
                condition_original_c = condition_original_c.float()

                x_next_c = self.controller.next_step(x_c)
                unsafe_mask_c = self.model.unsafe_mask(x_next_c)
                goal_mask_c = self.model.goal_mask(x_next_c)
                out_space_mask_c = ~self.model.space_mask(x_next_c)
                V_next_c = self.V(x_next_c)
                V_next_c[unsafe_mask_c] = 1.2
                V_next_c[goal_mask_c] = -10
                V_next_c[out_space_mask_c] = 1.2
                # modify ----------------------
                # d = 0.01
                # x_perturbed_c = self.maximize_in_neighborhood(x_next_c, d)
                # perturbations_region = torch.tensor([
                #     [d, d], [d, -d], [-d, d], [-d, -d]
                # ])  # Shape: (4, 2)

                # x_perturbed_region_c = x_next_c.unsqueeze(
                #     1) + perturbations_region.unsqueeze(0)  # Shape: (n, 4, 2)
                # x_perturbed_region_c = x_perturbed_region_c.view(-1, 2)
                # unsafe_safe_region_c = self.model.unsafe_mask(
                #     x_perturbed_region_c) | ~self.model.space_mask(x_perturbed_region_c)  # Shape: (n*4,)
                # unsafe_safe_region_c = unsafe_safe_region_c.view(-1, 4)
                # # print(torch.cat((x,x_next, x_perturbed, unsafe_safe_region), dim=1)[:30])
                # unsafe_safe_region_c = torch.any(unsafe_safe_region_c, dim=1)

                # unsafe_mask_c = self.model.unsafe_mask(x_perturbed_c)
                # goal_mask_c = self.model.goal_mask_smaller(x_perturbed_c)
                # out_space_mask_c = ~self.model.space_mask(x_perturbed_c)
                # V_next_c = self.V(x_perturbed_c)
                # V_next_c[unsafe_mask_c] = 1.2
                # V_next_c[goal_mask_c] = -10
                # V_next_c[out_space_mask_c] = 1.2
                # V_next_c[unsafe_safe_region_c] = 1.2

                # d = 0.01
                # num_samples = 10

                # # l-inf
                # x_perturbed_c = torch.empty_like(x_next_c.unsqueeze(
                #     1).expand(-1, num_samples, -1)).uniform_(-d, d)
                # x_perturbed_c += x_next_c.unsqueeze(1)
                # x_perturbed_c = x_perturbed_c.reshape(-1, x_next_c.shape[-1])

                # l-1
                # x_perturbed_c = torch.randn_like(x_next_c.unsqueeze(
                #     1).expand(-1, num_samples, -1))
                # x_perturbed_c = x_perturbed_c / \
                #     torch.norm(x_perturbed_c, p=1, dim=2, keepdim=True)
                # x_perturbed_c = (x_next_c.unsqueeze(1) + x_perturbed_c * d).reshape(
                #     -1, x_next_c.shape[-1])

                # V_next_perturbed_c = self.V(x_perturbed_c)
                # unsafe_mask = self.model.unsafe_mask(x_perturbed_c)
                # goal_mask = self.model.goal_mask(x_perturbed_c)
                # V_next_perturbed_c[unsafe_mask] = 1.2
                # V_next_perturbed_c[goal_mask] = -10
                # V_next_c = V_next_perturbed_c

                #######

                # V_next_perturbed_c = V_next_perturbed_c.view(-1, num_samples, 1)
                # V_max_c, _ = V_next_perturbed_c.max(dim=1, keepdim=True)
                # V_next_perturbed_max_c = V_max_c.expand(-1,
                #                                         num_samples, -1).reshape(-1, 1)
                #######

                # V_nongoal_c_expanded = V_nongoal_c.expand(-1,
                #                                           num_samples).reshape(-1, 1)
                # condition_original_c = (V_nongoal_c_expanded <=
                #                         self.safe_level).float()

                # ---------------------- modify
                # d = 0.001
                # _, lc = self.compute_lipschitz_constant("global")
                descent_violation_c = F.relu(
                    self.descenteps + (V_next_c - V_nongoal_c))

                #######
                # V_next_c[descent_violation_c >
                #          0] = V_next_perturbed_max_c[descent_violation_c > 0]
                # descent_violation_c = F.relu(
                #     self.descenteps + (V_next_c - V_nongoal_c_expanded)/(self.controller.t))
                #######

                descent_term_c = (100 * descent_violation_c *
                                  condition_original_c).mean()
            descent_term += descent_term_c

        return descent_term

    # def l2_reg_loss(self, l2_reg_upper=20):
    #     reg_loss = None
    #     for param in self.V.parameters():
    #         if reg_loss is None:
    #             reg_loss = torch.sum(param**2)
    #         else:
    #             reg_loss = reg_loss + param.norm(2)**2
    #     return 0.0005*max(reg_loss-l2_reg_upper, 0)

    # def local_lipschitz_loss(self, x, x_c, local_lip_upper=0.5):
    #     num_samples = 1000
    #     r = 0.01
    #     n, d = x.shape
    #     # Generate noise and normalize using L-infinity norm
    #     noise = torch.empty(n, num_samples, d).uniform_(-1, 1)
    #     noise /= torch.amax(torch.abs(noise), dim=-1,
    #                         keepdim=True)
    #     x_prime = x[:, None, :] + noise * r
    #     f_x = self.V(x[:, None, :])
    #     f_x_prime = self.V(x_prime)
    #     diff = torch.amax(torch.abs(f_x - f_x_prime),
    #                       dim=-1)  # (n, num_samples)
    #     denom = torch.amax(torch.abs(noise * r), dim=-1)

    #     lipschitz_ratios = diff / denom

    #     # Compute max_violation
    #     max_violation = torch.maximum(
    #         lipschitz_ratios - local_lip_upper, torch.tensor(0.0)).mean()
    #     if len(x_c) != 0:
    #         num_samples_c = 10000
    #         r = 0.01

    #         n, d = x_c.shape

    #         # Generate noise and normalize using L-infinity norm
    #         noise = torch.empty(n, num_samples_c, d).uniform_(-1, 1)
    #         noise /= torch.amax(torch.abs(noise), dim=-1,
    #                             keepdim=True)
    #         x_c_prime = x_c[:, None, :] + noise * r
    #         f_x_c = self.V(x_c[:, None, :])
    #         f_x_c_prime = self.V(x_c_prime)
    #         diff = torch.amax(torch.abs(f_x_c - f_x_c_prime),
    #                           dim=-1)  # (n, num_samples)
    #         denom = torch.amax(torch.abs(noise * r), dim=-1)
    #         assert denom.any() != 0

    #         lipschitz_ratios = diff / denom

    #         # Compute max_violation
    #         max_violation_c = torch.maximum(
    #             lipschitz_ratios - local_lip_upper, torch.tensor(0.0)).mean()

    #         # Sum over all samples
    #         max_violation += max_violation_c
    #     return max_violation

    # def global_lipschitz_loss(self, global_lip_upper=3):
    #     L2_product = 1
    #     for param in self.V.parameters():  # Iterate directly over parameters
    #         if param.ndim > 1:  # Only process weight matrices (ignore biases)
    #             # Use differentiable SVD
    #             U, S, V = torch.linalg.svd(param, full_matrices=False)
    #             spectral_norm = S[0]  # Largest singular value (spectral norm)
    #             L2_product *= spectral_norm
    #     return max(L2_product-global_lip_upper, 0)

    # def global_lipschitz_calculate(self):
    #     L2_product = 1.0
    #     # Disable gradient tracking for this computation
    #     with torch.no_grad():
    #         for param in self.V.parameters():
    #             # Process weight matrices only (ignore biases)
    #             if param.ndim > 1:
    #                 U, S, V = torch.linalg.svd(param, full_matrices=False)
    #                 # Extract largest singular value as a float
    #                 spectral_norm = S[0].item()
    #                 L2_product *= spectral_norm
    #     return L2_product

    # def l2_reg_calculate(self):
    #     l2_sum = 0.0
    #     with torch.no_grad():  # Disable gradients if you don't need them
    #         for param in self.V.parameters():
    #             # Sum of squared weights
    #             l2_sum += torch.sum(param ** 2).item()
    #     return l2_sum
    

    def training_step(self, batch, batch_idx):
        # x, x_c, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask, init_mask_c, goal_mask_c, nongoal_mask_c, safe_mask_c, unsafe_mask_c = batch
        x, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask, tj_flag, ce_flag = batch
        x, x_c = x[ce_flag == 0], x[ce_flag == 1]
        init_mask, init_mask_c = init_mask[ce_flag ==
                                           0], init_mask[ce_flag == 1]
        goal_mask, goal_mask_c = goal_mask[ce_flag ==
                                           0], goal_mask[ce_flag == 1]
        nongoal_mask, nongoal_mask_c = nongoal_mask[ce_flag ==
                                                    0], nongoal_mask[ce_flag == 1]
        safe_mask, safe_mask_c = safe_mask[ce_flag ==
                                           0], safe_mask[ce_flag == 1]
        unsafe_mask, unsafe_mask_c = unsafe_mask[ce_flag ==
                                                 0], unsafe_mask[ce_flag == 1]
        torch.set_grad_enabled(True)
        #_, lc = self.compute_lipschitz_constant("global")
        # safe_term, safe_acc, unsafe_term, unsafe_acc = self.goal_loss(x, goal_mask, nongoal_mask,safe_mask,unsafe_mask)
        init_term = self.init_loss(
            x, x_c, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask, init_mask_c, goal_mask_c, nongoal_mask_c, safe_mask_c, unsafe_mask_c)
        
        # goal_term = self.goal_loss(
        #     x, x_c, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask, init_mask_c, goal_mask_c, nongoal_mask_c, safe_mask_c, unsafe_mask_c)
        
        descent_term = self.descent_loss(
            x, x_c, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask, init_mask_c, goal_mask_c, nongoal_mask_c, safe_mask_c, unsafe_mask_c, tj_flag[ce_flag == 0].bool())
        
        goal_term = 0
        l2_reg_term = 0
        lip_term = 0
        #lip_term = self.global_lipschitz_loss()
        # lip_term = self.local_lipschitz_loss(x, x_c)
        # l2_reg_term = self.l2_reg_loss()
        total_loss = descent_term + init_term + goal_term + lip_term + l2_reg_term
        # if batch_idx % 100 == 0:
        #     lip_term = self.global_lipschitz_calculate()
        #     l2_reg_term = self.l2_reg_calculate()
        #     print(lip_term, l2_reg_term, init_term, descent_term)
        #     print(total_loss)

        opt_v, opt_c, opt_a = self.optimizers()

        if total_loss > 0:
            if self.epoch <= 5:  # vallina: 1; lip: 4
                opt_v.zero_grad()
                self.manual_backward(total_loss)
                opt_v.step()
            else:
                opt_a.zero_grad()
                self.manual_backward(total_loss)
                opt_a.step()

        batch_dict = {"loss": total_loss}
        self.losses_train.append(total_loss)
        self.init_losses_train.append(init_term)
        self.descent_losses_train.append(descent_term)
        # self.goal_losses_train.append(goal_term)
        return batch_dict

    def validation_step(self, batch, batch_idx):
        # x, x_c, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask, init_mask_c, goal_mask_c, nongoal_mask_c, safe_mask_c, unsafe_mask_c = batch
        x, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask, tj_flag, ce_flag = batch
        x, x_c = x[ce_flag == 0], x[ce_flag == 1]
        init_mask, init_mask_c = init_mask[ce_flag ==
                                           0], init_mask[ce_flag == 1]
        goal_mask, goal_mask_c = goal_mask[ce_flag ==
                                           0], goal_mask[ce_flag == 1]
        nongoal_mask, nongoal_mask_c = nongoal_mask[ce_flag ==
                                                    0], nongoal_mask[ce_flag == 1]
        safe_mask, safe_mask_c = safe_mask[ce_flag ==
                                           0], safe_mask[ce_flag == 1]
        unsafe_mask, unsafe_mask_c = unsafe_mask[ce_flag ==
                                                 0], unsafe_mask[ce_flag == 1]
        # _, lc = self.compute_lipschitz_constant("global")
        init_term = self.init_loss(
            x, x_c, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask, init_mask_c, goal_mask_c, nongoal_mask_c, safe_mask_c, unsafe_mask_c)
        descent_term = self.descent_loss(
            x, x_c, init_mask, goal_mask, nongoal_mask, safe_mask, unsafe_mask, init_mask_c, goal_mask_c, nongoal_mask_c, safe_mask_c, unsafe_mask_c, tj_flag[ce_flag == 0].bool())
        total_loss = descent_term + init_term
        batch_dict = {"loss": total_loss}

        self.losses_val.append(total_loss)
        self.init_losses_val.append(init_term)
        self.descent_losses_val.append(descent_term)
        return batch_dict

    def on_train_epoch_end(self):
        self.epoch += 1
        total = 0
        init = 0
        descent = 0
        for i in range(len(self.losses_train)):
            total += self.losses_train[i]
            init += self.init_losses_train[i]
            descent += self.descent_losses_train[i]

        print("Current loss: ", total/len(self.losses_train))
        print("Init loss: ", init/len(self.losses_train))
        print("Descent loss: ", descent/len(self.losses_train))
        print("Total loss:", total)
        # print("Gross: ", gross/len(self.losses_train))

        if (total.item() <= self.threshold):
            torch.save(self.V, self.out_model)
            torch.save(self.controller.nn, self.out_controller)
            self.log("saved_loss", torch.tensor(0, dtype=torch.float32))

            # x = torch.randn(10,4,requires_grad=True)

            # torch.onnx.export(self.V,x,"cur_model_20n.onnx",export_params=True,opset_version=10,do_constant_folding=True,input_names = ['input'],output_names = ['output'])
            # torch.onnx.export(self.controller.nn,x,"cur_controller_20n.onnx",export_params=True,opset_version=10,do_constant_folding=True,input_names = ['input'],output_names = ['output'])
            print("saved!")
            if self.epoch == 1:
                torch.save(self.V, "models/final_cert_reached.pt")
                exit()
        else:
            self.init_val += 1
            self.log("saved_loss", torch.tensor(
                self.init_val, dtype=torch.float32))
        with open(self.out_timing, 'a') as f:
            f.write("total loss: " + str(total) + "\n")
        self.losses_train = []
        self.init_losses_train = []
        self.descent_losses_train = []
        # if self.epoch >= 10:
        #     # update traj data
        #     self.reload_data()

    def reload_data(self):
        self.datamodule.controller = self.controller
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="fit")

        # # Rebuild the training dataloader manually
        # self.training = self.datamodule.train_dataloader()
        self.train_dataloader()

    # def compute_lipschitz_constant(self, type="global", x=None, r=None):
    #     """Compute the Lipschitz constant based on your criteria."""
    #     if type == "global":
    #         L2_product = torch.tensor(1.0)
    #         for param in self.V.parameters():  # Iterate directly over parameters
    #             # Only process weight matrices (ignore biases)
    #             if param.ndim > 1:
    #                 # Differentiable SVD
    #                 U, S, V = torch.linalg.svd(param, full_matrices=False)
    #                 # Largest singular value (spectral norm)
    #                 spectral_norm = S[0]
    #                 L2_product = L2_product * spectral_norm
    #         return "global", L2_product
    #     elif type == "local":
    #         num_samples = 1000
    #         n, d = x.shape

    #         noise = torch.empty(n, num_samples, d).uniform_(-1, 1)
    #         noise /= torch.amax(torch.abs(noise), dim=-1,
    #                             keepdim=True)
    #         x_prime = x[:, None, :] + noise * r
    #         f_x = self.V(x[:, None, :])
    #         f_x_prime = self.V(x_prime)
    #         diff = torch.amax(torch.abs(f_x - f_x_prime),
    #                           dim=-1)  # (n, num_samples)
    #         denom = torch.amax(torch.abs(noise * r), dim=-1)

    #         lipschitz_ratios = diff / denom

    #         return "local", lipschitz_ratios

    def on_validation_epoch_end(self):
        # generate new data and also plot before (TODO)
        total = 0
        goal = 0
        descent = 0
        for i in range(len(self.losses_val)):
            total += self.losses_val[i]
            goal += self.init_losses_val[i]
            descent += self.descent_losses_val[i]

        print("Current loss: ", total/len(self.losses_val))
        print("Init loss: ", goal/len(self.losses_val))
        print("Descent loss: ", descent/len(self.losses_val))

        self.losses_val = []
        self.init_losses_val = []
        self.descent_losses_val = []

    def configure_optimizers(self):
        # if sum(self.losses_train) <= 1e-4:
        #     optimizer_V = torch.optim.SGD(
        #         list(self.V.parameters()), lr=self.primal_learning_rate, momentum=0.9)
        #     optimizer_controller = torch.optim.SGD(list(self.V.parameters(
        #     )) + list(self.controller.nn.parameters()), lr=self.primal_learning_rate, momentum=0.9)
        # else:
        optimizer_V = torch.optim.Adam(
            list(self.V.parameters()), lr=self.primal_learning_rate)
        optimizer_controller = torch.optim.Adam(
            list(self.controller.nn.parameters()), lr=self.primal_learning_rate)
        optimizer_all = torch.optim.Adam(list(self.V.parameters(
        )) + list(self.controller.nn.parameters()), lr=self.primal_learning_rate)
        return optimizer_V, optimizer_controller, optimizer_all


def train_model(out_train_file, out_val_file, out_model_file, out_controller_file, threshold, initial_controller_file, out_timing_file, dynamic_file, device="cpu", lr=1e-3):
    model = Quadrator2D()
    V = LyapunovNetworkV().to(device)
    # V = torch.load("models_epsl_5e-3_lip3_2/cert_0.pt", weights_only=False)
    datamodule = SampleData(out_train_file, out_val_file, model, num_points=10000000 ,device=device)

    dynamic = Dynamic(dynamic_file)
    for param in dynamic.parameters():
        param.requires_grad = False

    controller = Controller(dynamic = dynamic, model = model,
        file_name=initial_controller_file, isInitial=True, device=device)
    # controller = Controller(
    #     file_name="controllers_epsl_5e-3_lip3_2/controller_0.pt", isInitial=False, device=device)

    trainer = Trainer(model, V, controller, datamodule, out_model_file,
                      out_controller_file, out_timing_file, threshold, primal_learning_rate=lr)
    # pltrainer = pl.Trainer(max_epochs=300, accelerator='cpu', callbacks=[
    #     EarlyStopping(monitor="step_loss", patience=3, min_delta=0.001, mode='min', verbose=True), ModelCheckpoint(
    #         monitor="step_loss",
    #         save_top_k=1,
    #         mode="min",
    #         dirpath="controllers_cross/",  # Optional: Specify directory to save checkpoints
    #         # Optional: Customize checkpoint file name
    #         filename="controller_1.pt"
    #     )])
    pltrainer = pl.Trainer(max_epochs=10000, accelerator='cpu', callbacks = [EarlyStopping(monitor="saved_loss", patience = 0, mode = 'max', verbose = True)])
    # 10000
    pltrainer.fit(trainer)
    # V = torch.load(out_model_file, weights_only=False)
    # trainer_lip = Trainer_Lip(model, V, controller, datamodule, out_model_file,
    #                           out_controller_file, threshold, primal_learning_rate=lr)
    # pltrainer = pl.Trainer(max_epochs=10000, accelerator='cpu', callbacks=[
    #                        EarlyStopping(monitor="saved_loss", patience=0, mode='max', verbose=True)])
    # pltrainer.fit(trainer_lip)
    # get a small loss # train_lip's first loop
    # after that do a sdp
    # while abs(Lip_prev - Lip_now) >= 1:
    # train
    # sdp


def retrain_model(epoch, counterexamples, counterexample_ranges, in_train_file, in_traj_file, out_train_file, out_val_file, out_traj_file, in_model_file, in_controller_file, out_model_file, out_controller_file, dynamic_file, counterexample_folder, out_timing_file, threshold, lr=1e-4, device="cpu"):
    # counterexamples = read_in_points(in_data_file, in_folder)
    model = Quadrator2D()
    V = torch.load(in_model_file, weights_only=False, map_location=device)

    dynamic = Dynamic(dynamic_file)
    for param in dynamic.parameters():
        param.requires_grad = False

    controller = Controller(dynamic = dynamic, model = model, file_name=in_controller_file,
                            isInitial=False, device=device)
    datamodule = SampleDataRetrain(
        epoch, model, V, controller, counterexamples, counterexample_ranges, in_train_file,  in_traj_file, out_train_file, out_val_file, out_traj_file, counterexample_folder, 50000)

    trainer = TrainerRetrain(model, V, controller, datamodule, out_model_file,
                             out_controller_file, out_timing_file, threshold, primal_learning_rate=lr)
    pltrainer = pl.Trainer(max_epochs=10000, accelerator='cpu', callbacks = [EarlyStopping(monitor="saved_loss", patience = 0, mode = 'max', verbose = True)])
    pltrainer.fit(trainer)

# train_model
# then generate counterexample files
# feed in counterexample files to retrain model
# frame this as a loop in a shell file somehow


if __name__ == "__main__":
    initial_controller_file = "/Users/javiergutierrez/Documents/Thesis/Thesis_Repo/2D_Quadrotor/controller_initial.pt"
    cur_data_file = 'data.pt'
    cur_val_file = 'val.pt'
    cur_model_file = 'model.pt'
    cur_controller_file = 'controller.pt'
    dynamic_file = 'dynamic.pt'
    threshold = 0

    train_model(cur_data_file, cur_val_file, cur_model_file,
                cur_controller_file, threshold, initial_controller_file, dynamic_file)
