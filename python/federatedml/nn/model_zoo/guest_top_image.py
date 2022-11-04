
from torch import nn
import torch as t
from torch.nn import functional as F

class ImgTopNet(nn.Module):
    def __init__(self):
        super(ImgTopNet, self).__init__()
        
        self.fc = t.nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x.flatten()
