import torch
import onnx
from onnx2torch import convert
from torch import nn
from torch import Tensor
import numpy as np


class LearnedController(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 128
        self.net = nn.Sequential(
            nn.Linear(2, dim),   # Input layer (matches Dense_0)
            nn.ReLU(),           # Activation
            nn.Linear(dim, dim),  # Hidden layer (matches Dense_1)
            nn.ReLU(),           # Activation
            nn.Linear(dim, 1)   # Output layer (matches Dense_2)
        )

    def forward(self, x):
        return self.net(x)
