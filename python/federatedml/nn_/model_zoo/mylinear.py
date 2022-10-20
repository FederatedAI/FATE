import torch as t
from torch import nn

class MyLinear(nn.Module):
    
    def __init__(self, input_dim=30):
        super(MyLinear, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.seq(x)
