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
            nn.Linear(6, 6, bias=True),
            
            nn.Linear(6, 4, bias=True),
            nn.ReLU(),
            nn.Linear(4, 2, bias=True),    # Output layer (matches Dense_2)
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.net(x)
        return logits
