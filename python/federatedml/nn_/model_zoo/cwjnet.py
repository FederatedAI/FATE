from torch import nn


class TestNet(nn.Module):

    def __init__(self, input_size):
        super(TestNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.activation = nn.Sigmoid()

    def forward(self, input_data):
        out = self.seq(input_data)
        return self.activation(out)