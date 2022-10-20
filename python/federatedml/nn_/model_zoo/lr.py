from torch import nn


class LogisticRegression(nn.Module):

    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, 1)
        )
        self.activation = nn.Sigmoid()

    def forward(self, input_data):
        out = self.seq(input_data)
        return self.activation(out)