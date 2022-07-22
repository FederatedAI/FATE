from federatedml.util import LOGGER
from federatedml.util import consts
try:
    import torch
    import torch as t
    from torch import nn
    from torch.nn import Module
    from torch.nn import functional as F
except ImportError:
    Module = object


def entropy(tensor):
    return -t.sum(tensor * t.log2(tensor))


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


def cross_entropy_for_one_hot(pred, target, reduce="mean"):
    if reduce == "mean":
        return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))
    elif reduce == "sum":
        return torch.sum(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))
    else:
        raise Exception("Does not support reduce [{}]".format(reduce))


def coae_loss(label, fake_label, reconstruct_label, lambda_1=10, lambda_2=2, verbose=False):

    loss_a = cross_entropy(reconstruct_label, label) - lambda_1 * cross_entropy(fake_label, label)
    loss_b = entropy(fake_label)
    if verbose:
        LOGGER.debug(
            'loss a is {} {}'.format(cross_entropy(reconstruct_label, label), cross_entropy(fake_label, label)))
        LOGGER.debug('loss b is {}'.format(loss_b))
    return loss_a - lambda_2 * loss_b


class CrossEntropy(object):

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, p2, p1):
        return cross_entropy(p2, p1, self.reduction)


class CoAE(Module):

    def __init__(self, input_dim=2, encode_dim=None):
        super(CoAE, self).__init__()
        self.d = input_dim

        if encode_dim is None:
            encode_dim = (6 * input_dim) ** 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encode_dim),
            nn.ReLU(),
            nn.Linear(encode_dim, input_dim),
            nn.Softmax(dim=1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, encode_dim),
            nn.ReLU(),
            nn.Linear(encode_dim, input_dim),
            nn.Softmax(dim=1)
        )

    def encode(self, x):
        x = t.Tensor(x)
        return self.encoder(x)

    def decode(self, fake_labels):
        fake_labels = t.Tensor(fake_labels)
        return self.decoder(fake_labels)

    def forward(self, x):
        x = t.Tensor(x)
        z = self.encoder(x)
        return self.decoder(z), z


def train_an_autoencoder_confuser(label_num, epoch=50, lambda1=1, lambda2=2, lr=0.001, verbose=False):
    coae = CoAE(label_num, )
    labels = torch.eye(label_num)
    opt = torch.optim.Adam(coae.parameters(), lr=lr)

    for i in range(epoch):
        opt.zero_grad()
        fake_labels = coae.encode(labels)
        reconstruct_labels = coae.decode(fake_labels)
        loss = coae_loss(labels, fake_labels, reconstruct_labels, lambda1, lambda2, verbose=verbose)
        loss.backward()
        opt.step()

    if verbose:
        LOGGER.debug('origin labels {}, fake labels {}, reconstruct labels {}'.format(labels, coae.encode(
            labels).detach().numpy(),
            coae.decode(coae.encode(
                labels)).detach().numpy()))

    return coae


def coae_label_reformat(labels):

    if labels.shape[1] == 1:
        return nn.functional.one_hot(t.Tensor(labels).flatten().type(t.int64), 2).numpy()
    else:
        return labels


if __name__ == '__main__':
    coae = train_an_autoencoder_confuser(2, epoch=1000, verbose=True, lambda1=2.0, lambda2=1.0, lr=0.02)
