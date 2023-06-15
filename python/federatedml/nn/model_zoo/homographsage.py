
import torch as t
from torch import nn
from torch.nn import Module
import torch_geometric.nn as pyg


class Sage(nn.Module):
    def __init__(self, in_channels, hidden_channels, class_num):
        super().__init__()
        self.model = nn.ModuleList([
            pyg.SAGEConv(in_channels=in_channels, out_channels=hidden_channels, project=True),
            pyg.SAGEConv(in_channels=hidden_channels, out_channels=class_num),
            nn.LogSoftmax()]
        )

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.model):
            if isinstance(conv, pyg.SAGEConv):
                x = conv(x, edge_index)
            else:
                x = conv(x)
        return x
