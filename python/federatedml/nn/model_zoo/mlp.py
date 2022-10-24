import torch as t
from torch import nn

class MLP(nn.Module):
    
    def __init__(self, input_dim=30, num_class=2):
        
        super(MLP, self).__init__()
        self.num_class = num_class
        
        self.seq = t.nn.Sequential(
            t.nn.Linear(input_dim, 16),
            t.nn.ReLU(),
            t.nn.Linear(16, num_class)
        )
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        seq_out = self.seq(x)
        if self.num_class == 2:
            return self.sigmoid(seq_out)
        else:
            if self.training:
                return seq_out
            else:
                return self.softmax(seq_out)
