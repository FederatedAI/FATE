import torchvision as tv
from torch import nn


class AlexNet(nn.Module):
    
    def __init__(self, num_class=2):
        super(AlexNet, self).__init__()
        self.model = tv.models.AlexNet(num_classes=num_class)
        
    def forward(self, x):
        return self.model(x) if self.training else nn.Softmax(dim=1)(self.model(x))
    