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

#from experimental_rollouts import Rollout

#trying easier (larger docking regions) at a time
#try further away (50-100)
#try tanh
class LyapunovNetworkV(nn.Module):
    def __init__(self, two_dim_docking):
        super().__init__()
        self.two_dim_docking = two_dim_docking
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4,30),
            nn.ReLU(),
            nn.Linear(30,30),
            nn.ReLU(),
            nn.Linear(30,1),
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        #logits_goal_mask = self.two_dim_docking.goal_mask(x)
        #logits[logits_goal_mask] = -0.1
        #logits_unsafe_mask = self.two_dim_docking.unsafe_mask(x)
        #logits[logits_unsafe_mask] = 1.1
        return logits
    
class TwoDimDocking():
    def __init__(self, st_pos, unsafe_pos, vel_limit):
        n = 0.001027
        self.v0 = 0.2
        self.v1 = 2*n
        #max_dist = math.sqrt(math.pow(st_pos,2) + math.pow(st_pos,2))
        #self.st_vel_limit = round(self.v0 + self.v1*max_dist,4)
        self.st_vel_limit = 0
        self.vel_limit = vel_limit
        self.st_pos = st_pos
        self.unsafe_pos = unsafe_pos

    def gen_vel_limit(self, x, y):
        dist = torch.sqrt(torch.pow(x,2) + torch.pow(y,2))
        return torch.round(self.v0 + self.v1*dist,decimals=4)
    def goal_mask(self, x):
        goal_mask = abs(x[:, 1]) < 0.35
        goal_mask.logical_and_(abs(x[:,0]) < 0.35)
        goal_mask.logical_and_(torch.sqrt(torch.pow(x[:,2],2) + torch.pow(x[:,3],2)) < self.gen_vel_limit(x[:,0], x[:,1]) - 0.0001)
        #goal_mask.logical_and(self.safe_mask(x))
        return goal_mask
    def nongoal_mask(self, x):
        #nongoal_mask.logical_and(self.safe_mask(x))
        #nongoal_mask.logical_and_(~self.unsafe_mask(x))
        nongoal_mask = abs(x[:, 1]) >= 0.35
        nongoal_mask.logical_or_(abs(x[:,0]) >= 0.35)
        nongoal_mask.logical_or_(torch.sqrt(torch.pow(x[:,2],2) + torch.pow(x[:,3],2)) >= self.gen_vel_limit(x[:,0], x[:,1]) - 0.0001)
        #goal_mask.logical_and(self.safe_mask(x))
        return nongoal_mask
        #return (~self.goal_mask(x))
    def unsafe_mask(self,x):
        '''
        unsafe_mask = x[:,0] > 5
        unsafe_mask.logical_or(x[:,1] > 5)
        unsafe_mask.logical_or(x[:,0] < 0)
        unsafe_mask.logical_or(x[:,1] < 0)

        unsafe_mask.logical_or(x[:,2] > 0.0001)
        unsafe_mask.logical_or(x[:,3] > 0.0001)
        unsafe_mask.logical_or(x[:,2] < -self.vel_limit)
        unsafe_mask.logical_or(x[:,3] < -self.vel_limit)
        '''
        #a relaxation: next best thing:
        #unsafe_mask = (abs(x[:,2]) + abs(x[:,3])) > (self.v0 + torch.max(abs(x[:,0]),abs(x[:,1])) * self.v1)
        unsafe_mask = abs(x[:,0]) >= self.unsafe_pos - 0.01
        unsafe_mask.logical_or_(abs(x[:,1]) >= self.unsafe_pos - 0.01)
        unsafe_mask.logical_or_(torch.sqrt(torch.pow(x[:,2],2) + torch.pow(x[:,3],2)) >= self.gen_vel_limit(x[:,0], x[:,1]) - 0.0001)
        return unsafe_mask
    def safe_mask(self,x):
        safe_mask = abs(x[:,0]) <= self.st_pos
        safe_mask.logical_and_(abs(x[:,1]) <= self.st_pos)
        safe_mask.logical_and_(abs(x[:,2]) <= self.st_vel_limit)
        safe_mask.logical_and_(abs(x[:,3]) <= self.st_vel_limit)
        safe_mask.logical_and_(self.nongoal_mask(x))
        #safe_mask.logical_or_(self.goal_mask(x))


        #safe_mask.logical_and_(x[:,2] >= -0.3125*x[:,0] - 0.225)
        #safe_mask.logical_and_(x[:,2] <= -0.3125*x[:,0] + 0.225)
        #safe_mask.logical_and_(x[:,3] >= -0.3125*x[:,1] - 0.225)
        #safe_mask.logical_and_(x[:,3] <= -0.3125*x[:,1] + 0.225)


        #safe_mask.logical_and(~self.goal_mask(x))
        '''
        safe_mask = x[:,0] <= 2
        safe_mask.logical_and(x[:,0] >= 1)
        safe_mask.logical_and(x[:,1] <= 2)
        safe_mask.logical_and(x[:,1] >= 1)
        '''
        #safe_mask.logical_and_(abs(x[:,2]) == 0.0)
        #safe_mask.logical_and_(abs(x[:,3]) == 0.0)
        return safe_mask
    def unsafe_mask_init(self,x):
        unsafe_mask = abs(x[:,0]) > self.unsafe_pos
        unsafe_mask.logical_or_(abs(x[:,1]) > self.unsafe_pos)
        unsafe_mask.logical_or_(torch.sqrt(torch.pow(x[:,2],2) + torch.pow(x[:,3],2)) >= self.gen_vel_limit(x[:,0], x[:,1]) + 0.0001)
        return unsafe_mask

class Controller():
    def __init__(self, file_name = "fixed_controller_20n.pt", isInitial=False, t=1):
        self.file_name = file_name

        m = 12
        n = 0.001027

        self.t = t
        
        if (isInitial):
            self.nn = LearnedController()
            self.nn.load_state_dict(torch.load(file_name))
        else:
            self.nn = torch.load(file_name)
        #self.nn = self.nn.to(device="cuda")

        # checking a more elaborate inductive property holds (closer or velocity decreases in appropriate direction)

        # velocity expansion is arbitrary

        # Matrix encoding of system dynamics
        self.coeffs_x_t = [
            4 - 3 * np.cos(n * t),
            0,
            1 / n * np.sin(n * t),
            2 / n - 2 / n * np.cos(n * t),
            (1 - np.cos(n * t)) / (m * n ** 2),
            2 * t / (m * n) - 2 * np.sin(n * t) / (m * n ** 2),
        ]
        self.coeffs_y_t = [
            -6 * n * t + 6 * np.sin(n * t),
            1,
            -2 / n + 2 / n * np.cos(n * t),
            -3 * t + 4 / n * np.sin(n * t),
            (-2 * t) / (m * n) + (2 * np.sin(n * t)) / (m * n ** 2),
            4 / (m * n ** 2) - (3 * t ** 2) / (2 * m) - (4 * np.cos(n * t)) / (m * n ** 2),
        ]
        self.coeffs_v_x_t = [
            3 * n * np.sin(n * t),
            0,
            np.cos(n * t),
            2 * np.sin(n * t),
            np.sin(n * t) / (m * n),
            2 / (m * n) - (2 * np.cos(n * t)) / (m * n),
        ]
        self.coeffs_v_y_t = [
            -6 * n + 6 * n * np.cos(n * t),
            0,
            -2 * np.sin(n * t),
            -3 + 4 * np.cos(n * t),
            (2 * np.cos(n * t) - 2) / (m * n),
            (-3 * t) / (m) + (4 * np.sin(n * t)) / (m * n),
        ]
        self.coeff_arr = torch.Tensor([self.coeffs_x_t,self.coeffs_y_t,self.coeffs_v_x_t,self.coeffs_v_y_t])
        #self.coeff_arr = self.coeff_arr.to(device="cuda")

    def next_step(self, x):
        forces_not_clipped = self.nn(x)
        forces = torch.clip(forces_not_clipped,-1,1)
        
        total_input = torch.cat((x,forces),1)

        next_step = torch.mm(total_input,torch.transpose(self.coeff_arr,0,1))

        return next_step
#num_points = 10000000
class SampleData(pl.LightningDataModule):
    def __init__(self, train_file_name, val_file_name, two_dim_docking, num_points = 10000000, dim = 4, val_split = 0.1, batch_size = 10000, max_tries = 5000):
        super().__init__()
        self.num_points = num_points
        self.two_dim_docking = two_dim_docking
        self.dim = dim
        self.val_split = val_split
        self.batch_size = batch_size
        self.max_tries = max_tries
        self.ranges = []
        self.train_file_name = train_file_name
        self.val_file_name = val_file_name
        self.rand_vel_upper = 1.0
        self.save_file = "counterexamples/safes.pt"

        for _ in range(2):
            self.ranges.append([-self.two_dim_docking.unsafe_pos-0.2, self.two_dim_docking.unsafe_pos+0.2])

        for _ in range(2):
            #ranges.append([-self.vel_limit,self.vel_limit])
            self.ranges.append([-self.two_dim_docking.vel_limit-0.05,self. two_dim_docking.vel_limit+0.05])
        
    '''
    def safe_mask(self, x):
        safe_mask = x[:, 3:].norm(dim=-1, p=2) <= self.v0+self.v1*(x[:, :2].norm(dim=-1, p=2))
        return safe_mask
    '''
    def gen_vel_limit(self, x, y):
        n = 0.001027
        v0 = 0.2
        v1 = 2*n
        dist = torch.sqrt(torch.pow(x,2) + torch.pow(y,2))
        return torch.round(v0 + v1*dist,decimals=4)

    def prepare_data(self):
        # First sample safe velocities
        x = torch.Tensor(self.num_points*5, 2).uniform_(0.0, 1.0)

        for i in range(2):
            min_val, max_val = self.ranges[i]
            x[:, i] = x[:, i] * (max_val - min_val) + min_val

        th = torch.Tensor(self.num_points*5).uniform_(0.0, 2*np.pi)
        v_l2 = torch.Tensor(self.num_points*5).uniform_(0.0, 1.0)
        safe_v = self.gen_vel_limit(x[:,0], x[:,1])
        v_l2 = v_l2 * (safe_v - 0.0) + 0.0
        vx = v_l2 * torch.cos(th)
        x = torch.cat((x, torch.unsqueeze(vx, 1)), 1)
        vy = v_l2 * torch.sin(th)
        x = torch.cat((x, torch.unsqueeze(vy, 1)), 1)

        print("preparing data")
        nongoal_mask_x = self.two_dim_docking.nongoal_mask(x)
        x = x[nongoal_mask_x]
        not_unsafe_mask_x = ~self.two_dim_docking.unsafe_mask_init(x)
        x = x[not_unsafe_mask_x]

        while (len(x) < 4 * self.num_points//5):
            print("num_typical " + str(len(x)))
            x = torch.Tensor(self.num_points*5, 2).uniform_(0.0, 1.0)

            for i in range(2):
                min_val, max_val = self.ranges[i]
                x[:, i] = x[:, i] * (max_val - min_val) + min_val

            th = torch.Tensor(self.num_points*5).uniform_(0.0, 2*np.pi)
            v_l2 = torch.Tensor(self.num_points*5).uniform_(0.0, 1.0)
            safe_v = self.gen_vel_limit(x[:,0], x[:,1])
            v_l2 = v_l2 * (safe_v - 0.0) + 0.0
            vx = v_l2 * torch.cos(th)
            x = torch.cat((x, torch.unsqueeze(vx, 1)), 1)
            vy = v_l2 * torch.sin(th)
            x = torch.cat((x, torch.unsqueeze(vy, 1)), 1)

            nongoal_mask_x = self.two_dim_docking.nongoal_mask(x)
            x = x[nongoal_mask_x]
            not_unsafe_mask_x = ~self.two_dim_docking.unsafe_mask_init(x)
            x = x[not_unsafe_mask_x]

        random_indices = torch.randperm(len(x))

        x = x[random_indices]

        x_train = x[:4 * self.num_points//5]
        #finishing part 1 sampling


        # Now sample for safe regions
        x = torch.Tensor(self.num_points*5, 2).uniform_(0.0, 1.0)

        for i in range(2):
            min_val, max_val = self.ranges[i]
            x[:, i] = x[:, i] * (max_val - min_val) + min_val

        zeros_x = torch.zeros(self.num_points*5)
        x = torch.cat((x, torch.unsqueeze(zeros_x, 1)), 1)
        x = torch.cat((x, torch.unsqueeze(zeros_x, 1)), 1)

        safe_mask_x = self.two_dim_docking.safe_mask(x)
        x = x[safe_mask_x]
        
        while (len(x) < self.num_points//5):
            print("num_safe " + str(len(x)))
            x = torch.Tensor(self.num_points*5, 2).uniform_(0.0, 1.0)

            for i in range(2):
                min_val, max_val = self.ranges[i]
                x[:, i] = x[:, i] * (max_val - min_val) + min_val

            zeros_x = torch.zeros(self.num_points*5)
            x = torch.cat((x, torch.unsqueeze(zeros_x, 1)), 1)
            x = torch.cat((x, torch.unsqueeze(zeros_x, 1)), 1)

            safe_mask_x = self.two_dim_docking.safe_mask(x)
            x = x[safe_mask_x]

        random_indices = torch.randperm(len(x))

        x = x[random_indices]

        x_train_2 = x[:self.num_points//5]
        torch.save(x_train_2, self.save_file)
        #finishing part 2 sampling

        x_train = torch.cat((x_train, x_train_2))

        random_indices = torch.randperm(len(x_train))

        x_train = x_train[random_indices]

        torch.save(x_train, self.train_file_name)

        '''
        # Next sample unsafe velocities
        z = torch.Tensor(self.num_points*2//5, 2).uniform_(0.0, 1.0)

        for i in range(2):
            min_val, max_val = self.ranges[i]
            z[:, i] = z[:, i] * (max_val - min_val) + min_val

        th = torch.Tensor(self.num_points*2//5).uniform_(0.0, 2*np.pi)
        v_l2 = torch.Tensor(self.num_points*2//5).uniform_(0.0, 1.0)
        safe_v = self.gen_vel_limit(z[:,0], z[:,1])
        v_l2 = v_l2 * (self.rand_vel_upper - safe_v) + safe_v
        vx = v_l2 * torch.cos(th)
        z = torch.cat((z, torch.unsqueeze(vx, 1)), 1)
        vy = v_l2 * torch.sin(th)
        z = torch.cat((z, torch.unsqueeze(vy, 1)), 1)

        
        x = torch.cat((x,z))
    
        y = torch.Tensor(self.num_points//5, self.dim).uniform_(
            0.0, 1.0
        )

        for i in range(2):
            y[:, i] = y[:, i] * (self.two_dim_docking.st_pos - (-self.two_dim_docking.st_pos)) + (-self.two_dim_docking.st_pos)

        for j in range(2):
            y[:, j+2] = y[:, j+2] * (self.two_dim_docking.st_vel_limit - (-self.two_dim_docking.st_vel_limit)) + (-self.two_dim_docking.st_vel_limit)

        x = torch.cat((x,y))
        '''

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

        # Sample safe velocities
        x = torch.Tensor(self.num_points*2//50, 2).uniform_(0.0, 1.0)

        for i in range(2):
            min_val, max_val = self.ranges[i]
            x[:, i] = x[:, i] * (max_val - min_val) + min_val

        th = torch.Tensor(self.num_points*2//50).uniform_(0.0, 2*np.pi)
        v_l2 = torch.Tensor(self.num_points*2//50).uniform_(0.0, 1.0)
        safe_v = self.gen_vel_limit(x[:,0], x[:,1])
        v_l2 = v_l2 * (safe_v - 0.0) + 0.0
        vx = v_l2 * torch.cos(th)
        x = torch.cat((x, torch.unsqueeze(vx, 1)), 1)
        vy = v_l2 * torch.sin(th)
        x = torch.cat((x, torch.unsqueeze(vy, 1)), 1)

        # Sample unsafe velocities
        z = torch.Tensor(self.num_points*2//50, 2).uniform_(0.0, 1.0)

        for i in range(2):
            min_val, max_val = self.ranges[i]
            z[:, i] = z[:, i] * (max_val - min_val) + min_val

        th = torch.Tensor(self.num_points*2//50).uniform_(0.0, 2*np.pi)
        v_l2 = torch.Tensor(self.num_points*2//50).uniform_(0.0, 1.0)
        safe_v = self.gen_vel_limit(z[:,0], z[:,1])
        v_l2 = v_l2 * (self.rand_vel_upper - safe_v) + safe_v
        vx = v_l2 * torch.cos(th)
        z = torch.cat((z, torch.unsqueeze(vx, 1)), 1)
        vy = v_l2 * torch.sin(th)
        z = torch.cat((z, torch.unsqueeze(vy, 1)), 1)

        x = torch.cat((x,z))

        y = torch.Tensor(self.num_points//50, self.dim).uniform_(
            0.0, 1.0
        )

        for i in range(2):
            y[:, i] = y[:, i] * (self.two_dim_docking.st_pos - (-self.two_dim_docking.st_pos)) + (-self.two_dim_docking.st_pos)

        for j in range(2):
            y[:, j+2] = y[:, j+2] * (self.two_dim_docking.st_vel_limit - (-self.two_dim_docking.st_vel_limit)) + (-self.two_dim_docking.st_vel_limit)

        x = torch.cat((x,y))

        random_indices = torch.randperm(len(x))
        x_val = x[random_indices]

        torch.save(x_val, self.val_file_name)

    def setup(self, stage = None):
        self.x_train = torch.load(self.train_file_name)
        self.x_val = torch.load(self.val_file_name)

        self.training_data = TensorDataset(
            self.x_train,
            self.two_dim_docking.goal_mask(self.x_train),
            self.two_dim_docking.nongoal_mask(self.x_train),
            self.two_dim_docking.safe_mask(self.x_train),
            self.two_dim_docking.unsafe_mask(self.x_train)
        )
        self.validation_data = TensorDataset(
            self.x_val,
            self.two_dim_docking.goal_mask(self.x_val),
            self.two_dim_docking.nongoal_mask(self.x_val),
            self.two_dim_docking.safe_mask(self.x_val),
            self.two_dim_docking.unsafe_mask(self.x_val)
        )

    
    def add_data(self):
        """Adding data -- nothing to do here"""
        pass

    def train_dataloader(self):
        """Make the DataLoader for training data"""
        return DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            num_workers=0,
        )
    
    def val_dataloader(self):
        """Make the DataLoader for validation data"""
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            num_workers=0,
        )
    

class Trainer(pl.LightningModule):
    def __init__(self, model, V, controller, datamodule, out_model, out_controller, threshold, primal_learning_rate = 1e-3, goalfactor = 1, decreasefactor = 1e1, nongoalfactor = 1, goaleps = 1e-4, nongoaleps=1e-4, descenteps = 1e-4, safe_level = 1, safe_factor = 1, unsafe_factor = 1e2, eps = 1e-4):
        super().__init__()
        #crucial for optimizer setting
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

        self.losses_train = []
        self.losses_val = []

        self.goal_losses_train = []
        self.goal_acc_train = []
        #self.safe_losses_train = []
        #self.safe_acc_train = []
        #self.unsafe_losses_train = []
        #self.unsafe_acc_train = []
        self.nongoal_losses_train = []
        self.nongoal_acc_train = []
        self.descent_losses_train = []
        self.descent_acc_train = []
        #self.gross_viol_train = []

        self.goal_losses_val = []
        self.goal_acc_val = []
        #self.safe_losses_val = []
        #self.safe_acc_val = []
        #self.unsafe_losses_val = []
        #self.unsafe_acc_val = []
        self.nongoal_losses_val = []
        self.nongoal_acc_val = []
        self.descent_losses_val = []
        self.descent_acc_val = []
        #self.gross_viol_val = []

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
    
    def _to_float(self, x):
        if torch.is_tensor(x):
            return float(x.detach().cpu().item())
        return float(x)


    def goal_loss(self, x, goal_mask, nongoal_mask, safe_mask, unsafe_mask):
        V = self.V(x)
        #V[goal_mask] = -0.1
        '''
        V_goal = V[goal_mask]
        goal_violation = F.relu(self.goaleps + V_goal - self.safe_level)
        #goal_violation = F.relu(self.goaleps + V_goal)
        goal_term = self.goalfactor * goal_violation.mean()
        goal_acc = self.goalfactor * (goal_violation >= self.goaleps).sum()/goal_violation.nelement()
        '''
        V_safe = V[safe_mask]
        if len(V_safe) == 0:
            safe_violation,safe_term,safe_acc = 0,0,0
        else:
            safe_violation = F.relu(self.eps + V_safe - self.safe_level)
            safe_term = self.safe_factor * safe_violation.mean()
            safe_acc = 0

        '''
        goal_term += safe_term
        goal_acc += safe_acc
        '''
        '''
        V_safe = V[safe_mask]
        safe_violation = F.relu(self.eps + V_safe - self.safe_level)
        safe_term = self.safe_factor * safe_violation.mean()
        safe_acc = (safe_violation >= self.eps).sum()/safe_violation.nelement()

        V_safe = V[goal_mask]
        safe_violation = F.relu(self.eps + V_safe - self.safe_level)
        safe_term += self.safe_factor * safe_violation.mean()
        safe_acc += (safe_violation >= self.eps).sum()/safe_violation.nelement()
        '''

        '''
        V_unsafe = V[unsafe_mask]
        if len(V_unsafe) == 0:
            unsafe_violation, unsafe_term, unsafe_acc = 0,0,0
        else:
            unsafe_violation = F.relu(self.eps - V_unsafe + self.safe_level)
            unsafe_term = self.unsafe_factor * unsafe_violation.mean()
            unsafe_acc = 0

        safe_term += unsafe_term
        safe_acc += unsafe_acc
        '''

        return safe_term, safe_acc
        '''
        unsafe_violation = F.relu(self.eps - V_unsafe + self.safe_level)
        unsafe_term = self.unsafe_factor * unsafe_violation.mean()
        unsafe_acc = (unsafe_violation >= self.eps).sum()/unsafe_violation.nelement()

        goal_term += unsafe_term
        goal_acc += unsafe_acc
        return safe_term, safe_acc
        '''

    
    def descent_loss(self, x, goal_mask, nongoal_mask, safe_mask, unsafe_mask):
        x = x[nongoal_mask]
        if len(x) == 0:
            nongoal_term, nongoal_acc = 0, 0
            descent_acc, descent_term = 0, 0
        else:
            V_nongoal = self.V(x)
            
            unsafe_mask = self.model.unsafe_mask_init(x)
            V_nongoal[unsafe_mask] = 1.2

            #condition_active = torch.sigmoid(10 * (self.safe_level + self.goaleps - V))

            condition_original = (V_nongoal <= self.safe_level)
            condition_original = torch.flatten(condition_original)
            
            #condition_original = condition_original.float()
            x = x[condition_original]
            V_nongoal = V_nongoal[condition_original]
            if len(V_nongoal) == 0:
                descent_term, descent_acc, nongoal_term, nongoal_acc = 0,0,0,0
                return descent_term, descent_acc, nongoal_term, nongoal_acc
            #print(V_nongoal)

            #default_violation = F.relu(self.nongoaleps-V_nongoal)
            nongoal_term = 0
            nongoal_acc = 0

            
            #nongoal_term = self.nongoalfactor * (default_violation).mean()

            x_next = self.controller.next_step(x)
            unsafe_mask = self.model.unsafe_mask(x_next)
            goal_mask = self.model.goal_mask(x_next)
            V_next = self.V(x_next)
            V_next[unsafe_mask] = 1.2
            V_next[goal_mask] = -10

            '''
            condition_new = x_next[:,0] > 0.35
            condition_new.logical_or_(x_next[:,1] > 0.35)
            condition_new.logical_or_(x_next[:,2] > 0.6)
            condition_new.logical_or_(x_next[:,3] > 0.6)
            condition_new = torch.reshape(condition_new, (len(x_next),1))

            condition_new_active = condition_new.float()
            '''

            #int_inactive_next = (V_next >= (self.safe_level + self.eps)).float()

            descent_violation = F.relu(self.descenteps + (V_next - V_nongoal)/(self.controller.t))
            descent_term = self.decreasefactor * (descent_violation).mean()
            descent_acc = 0
            #descent_acc = (descent_violation >= self.descenteps).sum()/(descent_violation.nelement())
            #descent_term = descent_violation.mean()
            #gross_descent_fail = 1e3 * (V_next * int_inactive_next * condition_active).mean()

            #tuning_loss = 2.2 * (x[:,2].norm(dim=-1,p=-2)-6*V).mean()

            return descent_term, descent_acc, nongoal_term, nongoal_acc

    def training_step(self, batch, batch_idx):
        x, goal_mask,nongoal_mask,safe_mask,unsafe_mask = batch
        #safe_term, safe_acc, unsafe_term, unsafe_acc = self.goal_loss(x, goal_mask, nongoal_mask,safe_mask,unsafe_mask)
        torch.set_grad_enabled(True) 
        goal_term, goal_acc = self.goal_loss(x, goal_mask, nongoal_mask,safe_mask,unsafe_mask)
        descent_term, descent_acc, nongoal_term, nongoal_acc = self.descent_loss(x, goal_mask, nongoal_mask,safe_mask,unsafe_mask)
        total_loss = descent_acc + descent_term + nongoal_term + nongoal_acc + goal_term + goal_acc

        opt_v, opt_f = self.optimizers()

        if total_loss > 0:
            if self.epoch >= 3:
                opt_f.zero_grad()
                self.manual_backward(total_loss)
                opt_f.step()
            else:
                opt_v.zero_grad()
                self.manual_backward(total_loss)
                opt_v.step()
        else:
            total_loss = torch.zeros(1)[0]

        '''
        if batch_idx == 0:
            print(total_loss.item())
            print(safe_acc.item())
            print(unsafe_acc.item())
            print(descent_acc.item())
        '''

        '''
        if total_loss.item() == 0:
            print("optimum of 0 for batch reached")
            print(total_loss)
        '''
        
        
        #batch_dict = {"loss": total_loss}
        self.losses_train.append(self._to_float(total_loss))
        self.goal_losses_train.append(self._to_float(goal_term))
        self.goal_acc_train.append(self._to_float(goal_acc))
        #self.safe_losses_train.append(safe_term)
        #self.safe_acc_train.append(safe_acc)
        #self.unsafe_losses_train.append(unsafe_term)
        #self.unsafe_acc_train.append(unsafe_acc)
        self.nongoal_losses_train.append(self._to_float(nongoal_term))
        self.nongoal_acc_train.append(self._to_float(nongoal_acc))
        self.descent_losses_train.append(self._to_float(descent_term))
        self.descent_acc_train.append(self._to_float(descent_acc))
        #self.gross_viol_train.append(gross_viol)
        #return batch_dict

    def validation_step(self, batch, batch_idx):
        x, goal_mask,nongoal_mask,safe_mask,unsafe_mask = batch
        #safe_term, safe_acc, unsafe_term, unsafe_acc = self.goal_loss(x, goal_mask, nongoal_mask,safe_mask,unsafe_mask) 
        goal_term, goal_acc = self.goal_loss(x, goal_mask, nongoal_mask,safe_mask,unsafe_mask)
        descent_term, descent_acc, nongoal_term, nongoal_acc = self.descent_loss(x, goal_mask, nongoal_mask,safe_mask,unsafe_mask)
        total_loss = descent_acc + descent_term + nongoal_term + nongoal_acc + goal_term + goal_acc
        
        #batch_dict = {"loss": total_loss}

        '''
        if total_loss.item() == 0:
            print("optimum of 0 for batch reached")
            return None
        '''
        
        self.losses_val.append(self._to_float(total_loss))
        self.goal_losses_val.append(self._to_float(goal_term))
        self.goal_acc_val.append(self._to_float(goal_acc))
        #self.safe_losses_val.append(safe_term)
        #self.safe_acc_val.append(safe_acc)
        #self.unsafe_losses_val.append(unsafe_term)
        #self.unsafe_acc_val.append(unsafe_acc)
        self.nongoal_losses_val.append(self._to_float(nongoal_term))
        self.nongoal_acc_val.append(self._to_float(nongoal_acc))
        self.descent_losses_val.append(self._to_float(descent_term))
        self.descent_acc_val.append(self._to_float(descent_acc))
        #self.gross_viol_val.append(gross_viol)
        #return batch_dict
    
    def on_train_epoch_end(self):
        self.epoch += 1

        total = 0
        goal = 0
        goal_acc = 0
        safe = 0
        safe_acc = 0
        unsafe = 0
        unsafe_acc = 0
        nongoal = 0
        nongoal_acc = 0
        descent = 0
        descent_acc = 0
        #gross = 0
        for i in range(len(self.losses_train)):
            total += self.losses_train[i]
            goal += self.goal_losses_train[i]
            goal_acc += self.goal_acc_train[i]
            #safe += self.safe_losses_train[i]
            #safe_acc += self.safe_acc_train[i]
            #unsafe += self.unsafe_losses_train[i]
            #unsafe_acc += self.unsafe_acc_train[i]
            nongoal += self.nongoal_losses_train[i]
            nongoal_acc += self.nongoal_acc_train[i]
            descent += self.descent_losses_train[i]
            descent_acc += self.descent_acc_train[i]
            #gross += self.gross_viol_train[i]

        print("Current loss: ", total/len(self.losses_train))
        print("Goal loss: ", goal/len(self.losses_train))
        print("Goal acc: ", goal_acc/len(self.losses_train))
        #print("Safe loss: ", safe/len(self.losses_train))
        #print("Safe acc: ", safe_acc/len(self.losses_train))
        #print("Unsafe loss: ", unsafe/len(self.losses_train))
        #print("Unsafe acc: ", unsafe_acc/len(self.losses_train))
        print("Nongoal loss: ", nongoal/len(self.losses_train))
        print("Nongoal acc: ", nongoal_acc/len(self.losses_train))
        print("Descent loss: ", descent/len(self.losses_train))
        print("Descent acc: ", descent_acc/len(self.losses_train))
        print("Total loss:", total)
        #print("Gross: ", gross/len(self.losses_train))

        if (total <= self.threshold):
            torch.save(self.V, self.out_model)
            torch.save(self.controller.nn, self.out_controller)
            self.log("saved_loss", torch.tensor(0,dtype=torch.float32))

            #x = torch.randn(10,4,requires_grad=True)

            #torch.onnx.export(self.V,x,"cur_model_20n.onnx",export_params=True,opset_version=10,do_constant_folding=True,input_names = ['input'],output_names = ['output'])
            #torch.onnx.export(self.controller.nn,x,"cur_controller_20n.onnx",export_params=True,opset_version=10,do_constant_folding=True,input_names = ['input'],output_names = ['output'])
            print("saved!")
        else:
            self.init_val += 1
            self.log("saved_loss", torch.tensor(self.init_val,dtype=torch.float32))

        self.losses_train = []
        self.goal_losses_train = []
        self.goal_acc_train = []
        #self.safe_losses_train = []
        #self.safe_acc_train = []
        #self.unsafe_losses_train = []
        #self.unsafe_acc_train = []
        self.nongoal_losses_train = []
        self.nongoal_acc_train = []
        self.descent_losses_train = []
        self.descent_acc_train = []
        #self.gross_viol_train = []
    
    def on_validation_epoch_end(self):
        #generate new data and also plot before (TODO)
        total = 0
        goal = 0
        goal_acc = 0
        safe = 0
        safe_acc = 0
        unsafe = 0
        unsafe_acc = 0
        nongoal = 0
        nongoal_acc = 0
        descent = 0
        descent_acc = 0
        #gross = 0
        for i in range(len(self.losses_val)):
            total += self.losses_val[i]
            goal += self.goal_losses_val[i]
            goal_acc += self.goal_acc_val[i]
            #safe += self.safe_losses_val[i]
            #safe_acc += self.safe_acc_val[i]
            #unsafe += self.unsafe_losses_val[i]
            #unsafe_acc += self.unsafe_acc_val[i]
            nongoal += self.nongoal_losses_val[i]
            nongoal_acc += self.nongoal_acc_val[i]
            descent += self.descent_losses_val[i]
            descent_acc += self.descent_acc_val[i]
            #gross += self.gross_viol_val[i]

        print("Current loss: ", total/len(self.losses_val))
        print("Goal loss: ", goal/len(self.losses_val))
        print("Goal acc: ", goal_acc/len(self.losses_val))
        #print("Safe loss: ", safe/len(self.losses_val))
        #print("Safe acc: ", safe_acc/len(self.losses_val))
        #print("Unsafe loss: ", unsafe/len(self.losses_val))
        #print("Unsafe acc: ", unsafe_acc/len(self.losses_val))
        print("Nongoal loss: ", nongoal/len(self.losses_val))
        print("Nongoal acc: ", nongoal_acc/len(self.losses_val))
        print("Descent loss: ", descent/len(self.losses_val))
        print("Descent acc: ", descent_acc/len(self.losses_val))
        #print("Gross: ", gross/len(self.losses_val))
        
        #self.datamodule.add_data()

        self.losses_val = []
        self.goal_losses_val = []
        self.goal_acc_val = []
        #self.safe_losses_val = []
        #self.safe_acc_val = []
        #self.unsafe_losses_val = []
        #self.unsafe_acc_val = []
        self.nongoal_losses_val = []
        self.nongoal_acc_val = []
        self.descent_losses_val = []
        self.descent_acc_val = []
        #self.gross_viol_val = []

        #rollout = Rollout(10,self.V,controller,model.vel_limit)
        #rollout.plots()


    def configure_optimizers(self):
        optimizer_V = torch.optim.Adam(list(self.V.parameters()), lr=self.primal_learning_rate)
        optimizer_controller = torch.optim.Adam(list(self.V.parameters()) + list(self.controller.nn.parameters()), lr=self.primal_learning_rate)
        return optimizer_V, optimizer_controller

class SampleDataRetrain(pl.LightningDataModule):
    def __init__(self, epoch, two_dim_docking, datapoints, counterexample_ranges, in_train_file, train_file, val_file, num_dpoints, num_points = 10000000, dim = 4, val_split = 0.1, batch_size = 10000, max_tries = 5000):
        super().__init__()
        self.epoch = epoch
        self.num_points = num_points
        self.two_dim_docking = two_dim_docking
        self.dim = dim
        self.val_split = val_split
        self.batch_size = batch_size
        self.max_tries = max_tries
        self.ranges = []
        self.counterexamples = datapoints
        self.counterexample_ranges = counterexample_ranges
        self.num_dpoints = num_dpoints

        self.in_train_file = in_train_file
        self.train_file = train_file
        self.val_file = val_file
        self.counterexample_file = "counterexamples/counterexamples.pt"
        self.save_file = "counterexamples/safes.pt"

        for _ in range(2):
            self.ranges.append([-self.two_dim_docking.unsafe_pos-0.2, self.two_dim_docking.unsafe_pos+0.2])

        for _ in range(2):
            #ranges.append([-self.vel_limit,self.vel_limit])
            self.ranges.append([-self.two_dim_docking.vel_limit-0.05,self. two_dim_docking.vel_limit+0.05])
    '''
    def safe_mask(self, x):
        safe_mask = x[:, 3:].norm(dim=-1, p=2) <= self.v0+self.v1*(x[:, :2].norm(dim=-1, p=2))
        return safe_mask
    '''
    def gen_vel_limit(self, x, y):
        n = 0.001027
        v0 = 0.2
        v1 = 2*n
        dist = torch.sqrt(torch.pow(x,2) + torch.pow(y,2))
        return torch.round(v0 + v1*dist,decimals=4)

    def prepare_data(self):
        x_train = torch.load(self.in_train_file)
        torch.save(x_train, self.train_file)

        #num_train = len(x_train)
        num_added = self.num_dpoints
        
        if self.epoch > 0:
            x_original = torch.load(self.counterexample_file)
            x_counterexamples = x_original[:num_added*50]
        else:
            x = torch.Tensor(num_added*100, 2).uniform_(0.0, 1.01)

            for i in range(2):
                min_val, max_val = self.ranges[i]
                x[:, i] = x[:, i] * (max_val - min_val) + min_val

            th = torch.Tensor(num_added*100).uniform_(0.0, 2*np.pi)
            v_l2 = torch.Tensor(num_added*100).uniform_(0.0, 1.01)
            safe_v = self.gen_vel_limit(x[:,0], x[:,1])
            v_l2 = v_l2 * (safe_v - 0.0) + 0.0
            vx = v_l2 * torch.cos(th)
            x = torch.cat((x, torch.unsqueeze(vx, 1)), 1)
            vy = v_l2 * torch.sin(th)
            x = torch.cat((x, torch.unsqueeze(vy, 1)), 1)

            print("preparing data")
            nongoal_mask_x = self.two_dim_docking.nongoal_mask(x)
            x = x[nongoal_mask_x]
            not_unsafe_mask_x = ~self.two_dim_docking.unsafe_mask_init(x)
            x = x[not_unsafe_mask_x]
            
            random_indices = torch.randperm(len(x))

            x = x[random_indices]

            x_counterexamples = x[:num_added]

        x_2 = torch.load(self.save_file)
        x_counterexamples = torch.cat((x_counterexamples,x_2))

        for i in range(len(self.counterexamples)):
            '''
            new_points = torch.zeros(num_added * 3, self.dim)
            for j in range(4):
                new_points[:, j] = new_points[:, j] + self.counterexamples[i][j]
            
            x_train = torch.cat((x_train,new_points))
            '''
            x_counterexamples = torch.cat((x_counterexamples,torch.unsqueeze(torch.Tensor(self.counterexamples[i]),0)))
            x_2 = torch.cat((x_2,torch.unsqueeze(torch.Tensor(self.counterexamples[i]),0)))
            '''
            new_points = torch.Tensor(num_added, self.dim).uniform_(0.0, 1.0)
            for j in range(2):
                new_points[:, j] = new_points[:, j] * (0.01) + self.counterexamples[i][j]
            for j in range(2):
                new_points[:, j+2] = new_points[:, j+2] * (0.001) + self.counterexamples[i][j+2]

            #x_train = torch.cat((x_train,new_points))
            x_counterexamples = torch.cat((x_counterexamples,new_points))
            '''

            new_points = torch.Tensor(num_added * 100, self.dim).uniform_(0.0, 1.01)
            for j in range(2):
                new_points[:, j] = new_points[:, j] * (0.2) + self.counterexamples[i][j] - 0.1
            for j in range(2):
                new_points[:, j+2] = new_points[:, j+2] * (0.1) + self.counterexamples[i][j+2] - 0.05

            nongoal_mask_x = self.two_dim_docking.nongoal_mask(new_points)
            new_points = new_points[nongoal_mask_x]
            not_unsafe_mask_x = ~self.two_dim_docking.unsafe_mask_init(new_points)
            new_points = new_points[not_unsafe_mask_x]

            while (len(new_points) < num_added-1):
                print("num counterexamples 1 " + str(len(num_added)))
                new_points = torch.Tensor(num_added * 100, self.dim).uniform_(0.0, 1.01)
                for j in range(2):
                    new_points[:, j] = new_points[:, j] * (0.2) + self.counterexamples[i][j] - 0.1
                for j in range(2):
                    new_points[:, j+2] = new_points[:, j+2] * (0.1) + self.counterexamples[i][j+2] - 0.05

                nongoal_mask_x = self.two_dim_docking.nongoal_mask(new_points)
                new_points = new_points[nongoal_mask_x]
                not_unsafe_mask_x = ~self.two_dim_docking.unsafe_mask_init(new_points)
                new_points = new_points[not_unsafe_mask_x]

            new_points_safe = new_points.detach().clone()
            for j in range(2):
                new_points_safe[:,j+2] = 0
            safe_mask_x = self.two_dim_docking.safe_mask(new_points_safe)
            new_points_safe = new_points_safe[safe_mask_x]
            new_points_safe = new_points_safe[:num_added//10]
            x_2 = torch.cat((x_2, new_points_safe))

            new_points = new_points[:num_added-1]

            #x_train = torch.cat((x_train,new_points))
            x_counterexamples = torch.cat((x_counterexamples,new_points))

        for i in range(len(self.counterexample_ranges)):
            new_points = torch.Tensor(100*num_added, self.dim).uniform_(0.0, 1.01)
            for j in range(4):
                new_points[:, j] = self.counterexample_ranges[i][j][0] + (self.counterexample_ranges[i][j][1] - self.counterexample_ranges[i][j][0]) * new_points[:, j]
            
            nongoal_mask_x = self.two_dim_docking.nongoal_mask(new_points)
            new_points = new_points[nongoal_mask_x]
            not_unsafe_mask_x = ~self.two_dim_docking.unsafe_mask_init(new_points)
            new_points = new_points[not_unsafe_mask_x]

            new_points_safe = new_points.detach().clone()
            for j in range(2):
                new_points_safe[:,j+2] = 0
            safe_mask_x = self.two_dim_docking.safe_mask(new_points_safe)
            new_points_safe = new_points_safe[safe_mask_x]
            new_points_safe = new_points_safe[:num_added//10]
            x_2 = torch.cat((x_2, new_points_safe))

            print("num counterexamples 2 " + str(len(new_points)))

            new_points = new_points[:num_added]
            #x_train = torch.cat((x_train,new_points))
            x_counterexamples = torch.cat((x_counterexamples,new_points))

        #random_indices = torch.randperm(num_train)
        #x_train = x_train[random_indices]
        #torch.save(x_train, self.train_file)

        random_indices = torch.randperm(len(x_counterexamples))
        x_counterexamples = x_counterexamples[random_indices]
        torch.save(x_counterexamples, self.counterexample_file)
        torch.save(x_2, self.save_file)
        '''
        x = torch.Tensor(self.num_points*4//50, self.dim).uniform_(
            0.0, 1.0
        )

        for i in range(self.dim):
            min_val, max_val = self.ranges[i]
            x[:, i] = x[:, i] * (max_val - min_val) + min_val

        y = torch.Tensor(self.num_points//50, self.dim).uniform_(
            0.0, 1.0
        )

        for i in range(2):
            y[:, i] = y[:, i] * (self.two_dim_docking.st_pos - (-self.two_dim_docking.st_pos)) + (-self.two_dim_docking.st_pos)

        for j in range(2):
            y[:, j+2] = y[:, j+2] * (self.two_dim_docking.st_vel_limit - (-self.two_dim_docking.st_vel_limit)) + (-self.two_dim_docking.st_vel_limit)

        x = torch.cat((x,y))

        random_indices = torch.randperm(len(x))
        x_val = x[random_indices]

        torch.save(x_val, self.val_file)
        '''

    def add_data(self):
        pass

    def setup(self, stage = None):
        self.x_train = torch.load(self.train_file)
        self.x_counterexamples = torch.load(self.counterexample_file)
        #self.x_val = torch.load(self.val_file)

        len_train = len(self.x_train)
        len_counterexamples = len(self.x_counterexamples)

        random_indices = torch.randperm(len_train)
        x_train = self.x_train[random_indices]
        self.x_val = x_train[:len_counterexamples]

        random_indices = torch.randperm(len_train)
        x_train = self.x_train[random_indices]
        self.x_train = x_train[:len_counterexamples]


        self.training_data = TensorDataset(
            self.x_train,
            self.x_counterexamples,
            self.two_dim_docking.goal_mask(self.x_train),
            self.two_dim_docking.nongoal_mask(self.x_train),
            self.two_dim_docking.safe_mask(self.x_train),
            self.two_dim_docking.unsafe_mask(self.x_train),
            self.two_dim_docking.goal_mask(self.x_counterexamples),
            self.two_dim_docking.nongoal_mask(self.x_counterexamples),
            self.two_dim_docking.safe_mask(self.x_counterexamples),
            self.two_dim_docking.unsafe_mask(self.x_counterexamples)
        )
        self.validation_data = TensorDataset(
            self.x_val,
            self.x_counterexamples,
            self.two_dim_docking.goal_mask(self.x_val),
            self.two_dim_docking.nongoal_mask(self.x_val),
            self.two_dim_docking.safe_mask(self.x_val),
            self.two_dim_docking.unsafe_mask(self.x_val),
            self.two_dim_docking.goal_mask(self.x_counterexamples),
            self.two_dim_docking.nongoal_mask(self.x_counterexamples),
            self.two_dim_docking.safe_mask(self.x_counterexamples),
            self.two_dim_docking.unsafe_mask(self.x_counterexamples)
        )

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
    def __init__(self, model, V, controller, datamodule, out_model, out_controller, threshold, primal_learning_rate = 1e-5, goalfactor = 1, decreasefactor = 1e1, nongoalfactor = 1, goaleps = 1e-4, nongoaleps=1e-4, descenteps = 1e-4, safe_level = 1, safe_factor = 1, unsafe_factor = 1e2, eps = 1e-4):
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

        self.losses_train = []
        self.losses_val = []

        self.goal_losses_train = []
        self.goal_acc_train = []
        #self.safe_losses_train = []
        #self.safe_acc_train = []
        #self.unsafe_losses_train = []
        #self.unsafe_acc_train = []
        self.nongoal_losses_train = []
        self.nongoal_acc_train = []
        self.descent_losses_train = []
        self.descent_acc_train = []
        #self.gross_viol_train = []

        self.goal_losses_val = []
        self.goal_acc_val = []
        #self.safe_losses_val = []
        #self.safe_acc_val = []
        #self.unsafe_losses_val = []
        #self.unsafe_acc_val = []
        self.nongoal_losses_val = []
        self.nongoal_acc_val = []
        self.descent_losses_val = []
        self.descent_acc_val = []
        #self.gross_viol_val = []

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
    
    def _to_float(self, x):
        if torch.is_tensor(x):
            return float(x.detach().cpu().item())
        return float(x)


    def goal_loss(self, x, x_c, goal_mask, nongoal_mask, safe_mask, unsafe_mask, goal_mask_c, nongoal_mask_c, safe_mask_c, unsafe_mask_c):
        V = self.V(x)
        #V[goal_mask] = -0.1
        '''
        V_goal = V[goal_mask]
        goal_violation = F.relu(self.goaleps + V_goal - self.safe_level)
        #goal_violation = F.relu(self.goaleps + V_goal)
        goal_term = self.goalfactor * goal_violation.mean()
        goal_acc = self.goalfactor * (goal_violation >= self.goaleps).sum()/goal_violation.nelement()
        '''
        V_safe = V[safe_mask]
        if len(V_safe) == 0:
            safe_violation,safe_term,safe_acc = 0,0,0
        else:
            safe_violation = F.relu(self.eps + V_safe - self.safe_level)
            safe_term = self.safe_factor * safe_violation.mean()
            safe_acc = 0

        '''
        V_unsafe = V[unsafe_mask]
        if len(V_unsafe) == 0:
            unsafe_violation, unsafe_term, unsafe_acc = 0,0,0
        else:
            unsafe_violation = F.relu(self.eps - V_unsafe + self.safe_level)
            unsafe_term = self.unsafe_factor * unsafe_violation.mean()
            unsafe_acc = 0
            
        safe_term += unsafe_term
        safe_acc += unsafe_acc
        '''

        V_c = self.V(x_c)
        V_safe_c = V_c[safe_mask_c]
        if len(V_safe_c) == 0:
            safe_violation_c,safe_term_c,safe_acc_c = 0,0,0
        else:
            safe_violation_c = F.relu(self.eps + V_safe_c - self.safe_level)
            safe_term_c = (self.safe_factor * 100 * safe_violation_c).mean()
            safe_acc_c = 0

        safe_term += safe_term_c
        safe_acc += safe_acc_c
        '''
        V_unsafe_c = V_c[unsafe_mask_c]
        if len(V_unsafe_c) == 0:
            unsafe_violation_c, unsafe_term_c, unsafe_acc_c = 0,0,0
        else:
            unsafe_violation_c = F.relu(self.eps - V_unsafe_c + self.safe_level)
            unsafe_term_c = self.unsafe_factor * unsafe_violation.mean()
            unsafe_acc_c = 0
            
        safe_term += unsafe_term_c
        safe_acc += unsafe_acc_c
        '''

        '''
        goal_term += safe_term
        goal_acc += safe_acc
        '''
        '''
        V_safe = V[safe_mask]
        safe_violation = F.relu(self.eps + V_safe - self.safe_level)
        safe_term = self.safe_factor * safe_violation.mean()
        safe_acc = (safe_violation >= self.eps).sum()/safe_violation.nelement()

        V_safe = V[goal_mask]
        safe_violation = F.relu(self.eps + V_safe - self.safe_level)
        safe_term += self.safe_factor * safe_violation.mean()
        safe_acc += (safe_violation >= self.eps).sum()/safe_violation.nelement()
        V_unsafe = V[unsafe_mask]
        unsafe_violation = F.relu(self.eps - V_unsafe + self.safe_level)
        unsafe_term = self.unsafe_factor * unsafe_violation.mean()
        unsafe_acc = (unsafe_violation >= self.eps).sum()/unsafe_violation.nelement()

        goal_term += unsafe_term
        goal_acc += unsafe_acc
        '''
        return safe_term, safe_acc

    
    def descent_loss(self, x, x_c, goal_mask, nongoal_mask, safe_mask, unsafe_mask, goal_mask_c, nongoal_mask_c, safe_mask_c, unsafe_mask_c):
        x = x[nongoal_mask]
        if len(x) == 0:
            nongoal_term, nongoal_acc = 0, 0
            descent_acc, descent_term = 0, 0
        else:
            V_nongoal = self.V(x)
            unsafe_mask = self.model.unsafe_mask_init(x)
            V_nongoal[unsafe_mask] = 1.2

            #condition_active = torch.sigmoid(10 * (self.safe_level + self.goaleps - V))

            condition_original = (V_nongoal <= self.safe_level)
            condition_original = torch.flatten(condition_original)
            
            #condition_original = condition_original.float()
            x = x[condition_original]
            V_nongoal = V_nongoal[condition_original]
            if len(V_nongoal) == 0:
                descent_term, descent_acc, nongoal_term, nongoal_acc = 0,0,0,0
            else:
                #print(V_nongoal)

                #default_violation = F.relu(self.nongoaleps-V_nongoal)
                nongoal_term = 0
                nongoal_acc = 0

                
                #nongoal_term = self.nongoalfactor * (default_violation).mean()

                x_next = self.controller.next_step(x)
                unsafe_mask = self.model.unsafe_mask(x_next)
                goal_mask = self.model.goal_mask(x_next)
                V_next = self.V(x_next)
                V_next[unsafe_mask] = 1.2
                V_next[goal_mask] = -10

                '''
                condition_new = x_next[:,0] > 0.35
                condition_new.logical_or_(x_next[:,1] > 0.35)
                condition_new.logical_or_(x_next[:,2] > 0.6)
                condition_new.logical_or_(x_next[:,3] > 0.6)
                condition_new = torch.reshape(condition_new, (len(x_next),1))

                condition_new_active = condition_new.float()
                '''

                #int_inactive_next = (V_next >= (self.safe_level + self.eps)).float()

                descent_violation = F.relu(self.descenteps + (V_next - V_nongoal)/(self.controller.t))
                descent_term = self.decreasefactor * (descent_violation).mean()
                descent_acc = 0

        x_c = x_c[nongoal_mask_c]
        if len(x_c) == 0:
            nongoal_term_c, nongoal_acc_c = 0, 0
            descent_acc_c, descent_term_c = 0, 0
        else:
            V_nongoal_c = self.V(x_c)
            unsafe_mask_c = self.model.unsafe_mask_init(x_c)
            V_nongoal_c[unsafe_mask_c] = 1.2
            #condition_active = torch.sigmoid(10 * (self.safe_level + self.goaleps - V))
            condition_original_c = (V_nongoal_c <= self.safe_level)
            condition_original_c = torch.flatten(condition_original_c)
            #condition_original_c = condition_original_c.float()

            x_c = x_c[condition_original_c]
            V_nongoal_c = V_nongoal_c[condition_original_c]
            if len(V_nongoal_c) == 0:
                descent_term_c, descent_acc_c, nongoal_term_c, nongoal_acc_c = 0,0,0,0
            else:
                #print(V_nongoal)

                #default_violation = F.relu(self.nongoaleps-V_nongoal)
                nongoal_term_c = 0
                nongoal_acc_c = 0

                
                #nongoal_term = self.nongoalfactor * (default_violation).mean()

                x_next_c = self.controller.next_step(x_c)
                unsafe_mask_c = self.model.unsafe_mask(x_next_c)
                goal_mask_c = self.model.goal_mask(x_next_c)
                V_next_c = self.V(x_next_c)
                V_next_c[unsafe_mask_c] = 1.2
                V_next_c[goal_mask_c] = -10

                '''
                condition_new = x_next[:,0] > 0.35
                condition_new.logical_or_(x_next[:,1] > 0.35)
                condition_new.logical_or_(x_next[:,2] > 0.6)
                condition_new.logical_or_(x_next[:,3] > 0.6)
                condition_new = torch.reshape(condition_new, (len(x_next),1))

                condition_new_active = condition_new.float()
                '''

                #int_inactive_next = (V_next >= (self.safe_level + self.eps)).float()

                descent_violation_c = F.relu(self.descenteps + (V_next_c - V_nongoal_c)/(self.controller.t))
                descent_term_c = (100 * descent_violation_c * condition_original_c).mean()
                descent_acc_c = 0

        descent_term += descent_term_c
        descent_acc += descent_acc_c
        nongoal_term += nongoal_term_c
        nongoal_acc += nongoal_acc_c

        #descent_acc = (descent_violation >= self.descenteps).sum()/(descent_violation.nelement())
        #descent_term = descent_violation.mean()
        #gross_descent_fail = 1e3 * (V_next * int_inactive_next * condition_active).mean()

        #tuning_loss = 2.2 * (x[:,2].norm(dim=-1,p=-2)-6*V).mean()

        return descent_term, descent_acc, nongoal_term, nongoal_acc

    def training_step(self, batch, batch_idx):
        x, x_c, goal_mask,nongoal_mask,safe_mask,unsafe_mask, goal_mask_c,nongoal_mask_c,safe_mask_c,unsafe_mask_c = batch
        torch.set_grad_enabled(True)
        #safe_term, safe_acc, unsafe_term, unsafe_acc = self.goal_loss(x, goal_mask, nongoal_mask,safe_mask,unsafe_mask) 
        goal_term, goal_acc = self.goal_loss(x, x_c, goal_mask, nongoal_mask,safe_mask,unsafe_mask,goal_mask_c,nongoal_mask_c,safe_mask_c,unsafe_mask_c)
        descent_term, descent_acc, nongoal_term, nongoal_acc = self.descent_loss(x, x_c, goal_mask, nongoal_mask,safe_mask,unsafe_mask,goal_mask_c,nongoal_mask_c,safe_mask_c,unsafe_mask_c)
        total_loss = descent_acc + descent_term + nongoal_term + nongoal_acc + goal_term + goal_acc
        opt_v, opt_f = self.optimizers()

        if total_loss > 0:
            if self.epoch >= 3:
                opt_f.zero_grad()
                self.manual_backward(total_loss)
                opt_f.step()
            else:
                opt_v.zero_grad()
                self.manual_backward(total_loss)
                opt_v.step()
        else:
            total_loss = torch.zeros(1)[0]

        '''
        if batch_idx == 0:
            print(total_loss.item())
            print(safe_acc.item())
            print(unsafe_acc.item())
            print(descent_acc.item())
        '''

        '''
        if total_loss.item() == 0:
            print("optimum of 0 for batch reached")
            print(total_loss)
        '''
        
        
        #batch_dict = {"loss": total_loss}
        self.losses_train.append(self._to_float(total_loss))
        self.goal_losses_train.append(self._to_float(goal_term))
        self.goal_acc_train.append(self._to_float(goal_acc))
        #self.safe_losses_train.append(safe_term)
        #self.safe_acc_train.append(safe_acc)
        #self.unsafe_losses_train.append(unsafe_term)
        #self.unsafe_acc_train.append(unsafe_acc)
        self.nongoal_losses_train.append(self._to_float(nongoal_term))
        self.nongoal_acc_train.append(self._to_float(nongoal_acc))
        self.descent_losses_train.append(self._to_float(descent_term))
        self.descent_acc_train.append(self._to_float(descent_acc))
        #self.gross_viol_train.append(gross_viol)
        #return batch_dict

    def validation_step(self, batch, batch_idx):
        x, x_c, goal_mask,nongoal_mask,safe_mask,unsafe_mask, goal_mask_c,nongoal_mask_c,safe_mask_c,unsafe_mask_c = batch
        #safe_term, safe_acc, unsafe_term, unsafe_acc = self.goal_loss(x, goal_mask, nongoal_mask,safe_mask,unsafe_mask) 
        goal_term, goal_acc = self.goal_loss(x, x_c, goal_mask, nongoal_mask,safe_mask,unsafe_mask,goal_mask_c,nongoal_mask_c,safe_mask_c,unsafe_mask_c)
        descent_term, descent_acc, nongoal_term, nongoal_acc = self.descent_loss(x, x_c, goal_mask, nongoal_mask,safe_mask,unsafe_mask,goal_mask_c,nongoal_mask_c,safe_mask_c,unsafe_mask_c)
        total_loss = descent_acc + descent_term + nongoal_term + nongoal_acc + goal_term + goal_acc
        
        #batch_dict = {"loss": total_loss}

        '''
        if total_loss.item() == 0:
            print("optimum of 0 for batch reached")
            return None
        '''
        
        self.losses_val.append(self._to_float(total_loss))
        self.goal_losses_val.append(self._to_float(goal_term))
        self.goal_acc_val.append(self._to_float(goal_acc))
        #self.safe_losses_val.append(safe_term)
        #self.safe_acc_val.append(safe_acc)
        #self.unsafe_losses_val.append(unsafe_term)
        #self.unsafe_acc_val.append(unsafe_acc)
        self.nongoal_losses_val.append(self._to_float(nongoal_term))
        self.nongoal_acc_val.append(self._to_float(nongoal_acc))
        self.descent_losses_val.append(self._to_float(descent_term))
        self.descent_acc_val.append(self._to_float(descent_acc))
        #self.gross_viol_val.append(gross_viol)
        #return batch_dict
    
    def on_train_epoch_end(self):
        self.epoch += 1

        total = 0
        goal = 0
        goal_acc = 0
        safe = 0
        safe_acc = 0
        unsafe = 0
        unsafe_acc = 0
        nongoal = 0
        nongoal_acc = 0
        descent = 0
        descent_acc = 0
        #gross = 0
        for i in range(len(self.losses_train)):
            total += self.losses_train[i]
            goal += self.goal_losses_train[i]
            goal_acc += self.goal_acc_train[i]
            #safe += self.safe_losses_train[i]
            #safe_acc += self.safe_acc_train[i]
            #unsafe += self.unsafe_losses_train[i]
            #unsafe_acc += self.unsafe_acc_train[i]
            nongoal += self.nongoal_losses_train[i]
            nongoal_acc += self.nongoal_acc_train[i]
            descent += self.descent_losses_train[i]
            descent_acc += self.descent_acc_train[i]
            #gross += self.gross_viol_train[i]

        print("Current loss: ", total/len(self.losses_train))
        print("Goal loss: ", goal/len(self.losses_train))
        print("Goal acc: ", goal_acc/len(self.losses_train))
        #print("Safe loss: ", safe/len(self.losses_train))
        #print("Safe acc: ", safe_acc/len(self.losses_train))
        #print("Unsafe loss: ", unsafe/len(self.losses_train))
        #print("Unsafe acc: ", unsafe_acc/len(self.losses_train))
        print("Nongoal loss: ", nongoal/len(self.losses_train))
        print("Nongoal acc: ", nongoal_acc/len(self.losses_train))
        print("Descent loss: ", descent/len(self.losses_train))
        print("Descent acc: ", descent_acc/len(self.losses_train))
        print("Total loss:", total)
        #print("Gross: ", gross/len(self.losses_train))

        if (total <= self.threshold):
            torch.save(self.V, self.out_model)
            torch.save(self.controller.nn, self.out_controller)
            self.log("saved_loss", torch.tensor(0,dtype=torch.float32))

            #x = torch.randn(10,4,requires_grad=True)

            #torch.onnx.export(self.V,x,"cur_model_20n.onnx",export_params=True,opset_version=10,do_constant_folding=True,input_names = ['input'],output_names = ['output'])
            #torch.onnx.export(self.controller.nn,x,"cur_controller_20n.onnx",export_params=True,opset_version=10,do_constant_folding=True,input_names = ['input'],output_names = ['output'])
            print("saved!")
            if self.epoch == 1:
                torch.save(self.V, "models/final_cert_reached.pt")
                quit()
        else:
            self.init_val += 1
            self.log("saved_loss", torch.tensor(self.init_val,dtype=torch.float32))

        self.losses_train = []
        self.goal_losses_train = []
        self.goal_acc_train = []
        #self.safe_losses_train = []
        #self.safe_acc_train = []
        #self.unsafe_losses_train = []
        #self.unsafe_acc_train = []
        self.nongoal_losses_train = []
        self.nongoal_acc_train = []
        self.descent_losses_train = []
        self.descent_acc_train = []
        #self.gross_viol_train = []
    
    def on_validation_epoch_end(self):
        #generate new data and also plot before (TODO)
        total = 0
        goal = 0
        goal_acc = 0
        safe = 0
        safe_acc = 0
        unsafe = 0
        unsafe_acc = 0
        nongoal = 0
        nongoal_acc = 0
        descent = 0
        descent_acc = 0
        #gross = 0
        for i in range(len(self.losses_val)):
            total += self.losses_val[i]
            goal += self.goal_losses_val[i]
            goal_acc += self.goal_acc_val[i]
            #safe += self.safe_losses_val[i]
            #safe_acc += self.safe_acc_val[i]
            #unsafe += self.unsafe_losses_val[i]
            #unsafe_acc += self.unsafe_acc_val[i]
            nongoal += self.nongoal_losses_val[i]
            nongoal_acc += self.nongoal_acc_val[i]
            descent += self.descent_losses_val[i]
            descent_acc += self.descent_acc_val[i]
            #gross += self.gross_viol_val[i]

        print("Current loss: ", total/len(self.losses_val))
        print("Goal loss: ", goal/len(self.losses_val))
        print("Goal acc: ", goal_acc/len(self.losses_val))
        #print("Safe loss: ", safe/len(self.losses_val))
        #print("Safe acc: ", safe_acc/len(self.losses_val))
        #print("Unsafe loss: ", unsafe/len(self.losses_val))
        #print("Unsafe acc: ", unsafe_acc/len(self.losses_val))
        print("Nongoal loss: ", nongoal/len(self.losses_val))
        print("Nongoal acc: ", nongoal_acc/len(self.losses_val))
        print("Descent loss: ", descent/len(self.losses_val))
        print("Descent acc: ", descent_acc/len(self.losses_val))
        #print("Gross: ", gross/len(self.losses_val))
        
        #self.datamodule.add_data()

        self.losses_val = []
        self.goal_losses_val = []
        self.goal_acc_val = []
        #self.safe_losses_val = []
        #self.safe_acc_val = []
        #self.unsafe_losses_val = []
        #self.unsafe_acc_val = []
        self.nongoal_losses_val = []
        self.nongoal_acc_val = []
        self.descent_losses_val = []
        self.descent_acc_val = []
        #self.gross_viol_val = []

        #rollout = Rollout(10,self.V,controller,model.vel_limit)
        #rollout.plots()

    def configure_optimizers(self):
        optimizer_V = torch.optim.Adam(list(self.V.parameters()), lr=self.primal_learning_rate)
        optimizer_controller = torch.optim.Adam(list(self.V.parameters()) + list(self.controller.nn.parameters()), lr=self.primal_learning_rate)
        return optimizer_V, optimizer_controller

def train_model(st_pos, unsafe_pos, vel_limit, out_train_file, out_val_file, out_model_file, out_controller_file, threshold, initial_controller_file,  lr=5e-3):
    model = TwoDimDocking(st_pos, unsafe_pos, vel_limit)
    V = LyapunovNetworkV(model)
    datamodule = SampleData(out_train_file, out_val_file, model)
    controller = Controller(file_name = initial_controller_file, isInitial = True)
    trainer = Trainer(model, V, controller, datamodule, out_model_file, out_controller_file, threshold, primal_learning_rate=lr)
    pltrainer = pl.Trainer(max_epochs=10000, accelerator='cpu', callbacks = [EarlyStopping(monitor="saved_loss", patience = 0, mode = 'max', verbose = True)])
    pltrainer.fit(trainer)
    
def retrain_model(epoch, counterexamples, counterexample_ranges, st_pos, unsafe_pos, vel_limit, in_train_file, out_train_file, out_val_file, in_model_file, in_controller_file, out_model_file, out_controller_file, threshold, lr=1e-4):
    #counterexamples = read_in_points(in_data_file, in_folder)
    model = TwoDimDocking(st_pos, unsafe_pos, vel_limit)
    V = torch.load(in_model_file)
    datamodule = SampleDataRetrain(epoch, model, counterexamples, counterexample_ranges, in_train_file, out_train_file, out_val_file, 5000)
    controller = Controller(file_name = in_controller_file, isInitial = False)
    trainer = TrainerRetrain(model, V, controller, datamodule, out_model_file, out_controller_file, threshold, primal_learning_rate=lr)
    pltrainer = pl.Trainer(max_epochs=10000, accelerator='cpu', callbacks = [EarlyStopping(monitor="saved_loss", patience = 0, mode = 'max', verbose = True)])
    pltrainer.fit(trainer)

#train_model
#then generate counterexample files
#feed in counterexample files to retrain model
#frame this as a loop in a shell file somehow

if __name__ == "__main__":
    initial_controller_file = "../../fixed_controller_20n.pt"
    cur_data_file = 'data.pt'
    cur_val_file = 'val.pt'
    cur_model_file = 'model.pt'
    cur_controller_file = 'controller.pt'
    threshold = 0

    train_model(4, 5, 0.22, cur_data_file, cur_val_file, cur_model_file, cur_controller_file, threshold, initial_controller_file)

        


