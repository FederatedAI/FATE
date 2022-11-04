from torch import nn
import torch as t
from torch.nn import functional as F

class ImgNet(nn.Module):
    def __init__(self, class_num=10):
        super(ImgNet, self).__init__()
        self.seq = t.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3),
            nn.AvgPool2d(kernel_size=3)
        )
        
        self.fc = t.nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, class_num)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.seq(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        if self.training:
            return x
        else:
            return self.softmax(x)
