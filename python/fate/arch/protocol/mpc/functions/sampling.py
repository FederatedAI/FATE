#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fate.arch.protocol import mpc

__all__ = [
    "bernoulli",
    "randn",
    "weighted_index",
    "weighted_sample",
]


def randn(*sizes, device=None):
    """
    Returns a tensor with normally distributed elements. Samples are
    generated using the Box-Muller transform with optimizations for
    numerical precision and MPC efficiency.
    """
    u = mpc.rand(*sizes, device=device).flatten()
    odd_numel = u.numel() % 2 == 1
    if odd_numel:
        u = mpc.cat([u, mpc.rand((1,), device=device)])

    n = u.numel() // 2
    u1 = u[:n]
    u2 = u[n:]

    # Radius = sqrt(- 2 * log(u1))
    r2 = -2 * u1.log(input_in_01=True)
    r = r2.sqrt()

    # Theta = cos(2 * pi * u2) or sin(2 * pi * u2)
    cos, sin = u2.sub(0.5).mul(6.28318531).cossin()

    # Generating 2 independent normal random variables using
    x = r.mul(sin)
    y = r.mul(cos)
    z = mpc.cat([x, y])

    if odd_numel:
        z = z[1:]

    return z.view(*sizes)


def bernoulli(self):
    """Returns a tensor with elements in {0, 1}. The i-th element of the
    output will be 1 with probability according to the i-th value of the
    input tensor."""
    return self > mpc.rand(self.size(), device=self.device)


def weighted_index(self, dim=None):
    """
    Returns a tensor with entries that are one-hot along dimension `dim`.
    These one-hot entries are set at random with weights given by the input
    `self`.

    Examples::

        >>> encrypted_tensor = MPCTensor(torch.tensor([1., 6.]))
        >>> index = encrypted_tensor.weighted_index().get_plain_text()
        # With 1 / 7 probability
        torch.tensor([1., 0.])

        # With 6 / 7 probability
        torch.tensor([0., 1.])
    """
    if dim is None:
        return self.flatten().weighted_index(dim=0).view(self.size())

    x = self.cumsum(dim)
    max_weight = x.index_select(dim, torch.tensor(x.size(dim) - 1, device=self.device))
    r = mpc.rand(max_weight.size(), device=self.device) * max_weight

    gt = x.gt(r)
    shifted = gt.roll(1, dims=dim)
    shifted.data.index_fill_(dim, torch.tensor(0, device=self.device), 0)

    return gt - shifted


def weighted_sample(self, dim=None):
    """
    Samples a single value across dimension `dim` with weights corresponding
    to the values in `self`

    Returns the sample and the one-hot index of the sample.

    Examples::

        >>> encrypted_tensor = MPCTensor(torch.tensor([1., 6.]))
        >>> index = encrypted_tensor.weighted_sample().get_plain_text()
        # With 1 / 7 probability
        (torch.tensor([1., 0.]), torch.tensor([1., 0.]))

        # With 6 / 7 probability
        (torch.tensor([0., 6.]), torch.tensor([0., 1.]))
    """
    indices = self.weighted_index(dim)
    sample = self.mul(indices).sum(dim)
    return sample, indices
