import torch
import onnx
from onnx2torch import convert
from torch import nn
from torch import Tensor
import numpy as np

class LearnedController(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 196
        self.net = nn.Sequential(
            nn.Linear(6, dim),   # Input layer (matches Dense_0)
            nn.ReLU(),           # Activation       
            nn.Linear(dim, dim),
            nn.ReLU(),      
            nn.Linear(dim, dim),
            nn.ReLU(),       # Activation     
            nn.Linear(dim, dim),
            nn.ReLU(),       # Activation      
            nn.Linear(dim, 2)    # Output layer (matches Dense_2)
        ).double()
        # nn.Linear(6, 6, bias=True),
        # nn.LeakyReLU(0.01),
        # nn.Linear(6, 4, bias=True),
        # nn.LeakyReLU(0.01),
        # nn.Linear(4, 2, bias=True), 
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.net(x)
        return logits
