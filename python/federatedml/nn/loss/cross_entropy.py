import torch as t
from federatedml.util import consts
from torch.nn.functional import one_hot


def cross_entropy(p2, p1, reduction='mean'):
    p2 = p2 + consts.FLOAT_ZERO  # to avoid nan
    assert p2.shape == p1.shape
    if reduction == 'sum':
        return -t.sum(p1 * t.log(p2))
    elif reduction == 'mean':
        return -t.mean(t.sum(p1 * t.log(p2), dim=1))
    elif reduction == 'none':
        return -t.sum(p1 * t.log(p2), dim=1)
    else:
        raise ValueError('unknown reduction')


class CrossEntropyLoss(t.nn.Module):

    """
    A CrossEntropy Loss that will not compute Softmax
    """

    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, label):

        one_hot_label = one_hot(label.flatten())
        loss_ = cross_entropy(pred, one_hot_label, self.reduction)

        return loss_
