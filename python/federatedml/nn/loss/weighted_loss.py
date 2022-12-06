import torch as t
from torch.nn import BCELoss


class WeightedBCE(t.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = BCELoss(reduce=False)

    def forward(self, pred, label_and_weight):
        label, weights = label_and_weight
        losses = self.loss_fn(pred, label)
        losses = losses * weights
        loss_val = losses.sum() / weights.sum()
        return loss_val
