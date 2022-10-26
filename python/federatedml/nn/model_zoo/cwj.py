import torch as t
from torch import nn

class CWJ(nn.Module):
    
    def __init__(self, input_dim=100):
        super(CWJ, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.seq(x)

