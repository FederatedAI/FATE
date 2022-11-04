from torch import nn
import torch as t
from torch.nn import functional as F

class ImgTopNet(nn.Module):
    def __init__(self, num_class=10):
        super(ImgTopNet, self).__init__()
        
        self.fc = t.nn.Sequential(
            nn.Linear(8, num_class)
        )

    def forward(self, x):
        x = self.fc(x)
        return x if self.training else nn.Softmax(dim=1)(x)
