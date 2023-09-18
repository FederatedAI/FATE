from torch import nn


class Multi(nn.Module):
    def __init__(self, feat=18, class_num=4) -> None:
        super().__init__()
        self.class_num = class_num
        self.model = nn.Sequential(nn.Linear(feat, 10), nn.ReLU(), nn.Linear(10, class_num))

    def forward(self, x):
        if self.training:
            return self.model(x)
        else:
            return nn.Softmax(dim=-1)(self.model(x))
