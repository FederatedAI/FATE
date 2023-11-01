#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .module import Module
from ..cryptensor import CrypTensor
from ...mpc import cryptensor


class _Loss(Module):
    """
    Base criterion class that mimics Pytorch's Loss.
    """

    def __init__(self, reduction="mean", skip_forward=False):
        super(_Loss, self).__init__()
        if reduction != "mean":
            raise NotImplementedError("reduction %s not supported")
        self.reduction = reduction
        self.skip_forward = skip_forward

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward not implemented")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __getattribute__(self, name):
        if name != "forward":
            return object.__getattribute__(self, name)

        def forward_function(*args, **kwargs):
            """Silently encrypt Torch tensors if needed."""
            if self.encrypted or any(isinstance(arg, CrypTensor) for arg in args):
                args = list(args)
                for idx, arg in enumerate(args):
                    if torch.is_tensor(arg):
                        args[idx] = cryptensor(arg)
            return object.__getattribute__(self, name)(*tuple(args), **kwargs)

        return forward_function


class MSELoss(_Loss):
    r"""
    Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the prediction :math:`x` and target :math:`y`.

    The loss can be described as:

    .. math::
        \ell(x, y) = mean(L) = mean(\{l_1,\dots,l_N\}^\top), \quad
        l_n = (x_n - y_n)^2,

    where :math:`N` is the batch size, :math:`x` and :math:`y` are tensors of
    arbitrary shapes with a total of :math:`n` elements each.
    """  # noqa: W605

    def forward(self, x, y):
        assert x.size() == y.size(), "input and target must have the same size"
        return (x - y).square().mean()


class L1Loss(_Loss):
    r"""
    Creates a criterion that measures the mean absolute error between each element in
    the prediction :math:`x` and target :math:`y`.

    The loss can be described as:

    .. math::
        \ell(x, y) = mean(L) = mean(\{l_1,\dots,l_N\}^\top), \quad
        l_n = \left | x_n - y_n \right |,

    where :math:`N` is the batch size, :math:`x` and :math:`y` are tensors of
    arbitrary shapes with a total of :math:`n` elements each.
    """  # noqa: W605

    def forward(self, x, y):
        assert x.size() == y.size(), "input and target must have the same size"
        return (x - y).abs().mean()


class BCELoss(_Loss):
    r"""
    Creates a criterion that measures the Binary Cross Entropy
    between the prediction :math:`x` and the target :math:`y`.

    The loss can be described as:

    .. math::
        \ell(x, y) = mean(L) = mean(\{l_1,\dots,l_N\}^\top), \quad
        l_n = - \left [ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right ],

    where :math:`N` is the batch size, :math:`x` and :math:`y` are tensors of
    arbitrary shapes with a total of :math:`n` elements each.

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets :math:`y` should be numbers
    between 0 and 1.
    """  # noqa: W605

    def forward(self, x, y):
        assert x.size() == y.size(), "input and target must have the same size"
        return x.binary_cross_entropy(y, skip_forward=self.skip_forward)


class CrossEntropyLoss(_Loss):
    r"""
    Creates a criterion that measures cross-entropy loss between the
    prediction :math:`x` and the target :math:`y`. It is useful when
    training a classification problem with `C` classes.

    The prediction `x` is expected to contain raw, unnormalized scores for each class.

    The prediction `x` has to be a Tensor of size either :math:`(N, C)` or
    :math:`(N, C, d_1, d_2, ..., d_K)`, where :math:`N` is the size of the minibatch,
    and with :math:`K \geq 1` for the `K`-dimensional case (described later).

    This criterion expects a class index in the range :math:`[0, C-1]` as the
    target `y` for each value of a 1D tensor of size `N`.

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = -\log \left(
        \frac{\exp(x[class])}{\sum_j \exp(x[j])} \right )
        = -x[class] + \log \left (\sum_j \exp(x[j]) \right)

    The losses are averaged across observations for each batch

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape.
    """  # noqa: W605

    def forward(self, x, y):
        x = x.squeeze()
        y = y.squeeze()
        assert x.size() == y.size(), "input and target must have the same size"
        return x.cross_entropy(y, skip_forward=self.skip_forward)


class BCEWithLogitsLoss(_Loss):
    r"""
    This loss combines a Sigmoid layer and the BCELoss in one single class.

    The loss can be described as:

    .. math::
        p = \sigma(x)

    .. math::
        \ell(x, y) = mean(L) = mean(\{l_1,\dots,l_N\}^\top), \quad
        l_n = - \left [ y_n \cdot \log p_n + (1 - y_n) \cdot \log (1 - p_n) \right ],

    This is used for measuring the error of a reconstruction in for example an
    auto-encoder. Note that the targets t[i] should be numbers between 0 and 1.
    """  # noqa: W605

    def forward(self, x, y):
        assert x.size() == y.size(), "input and target must have the same size"
        return x.binary_cross_entropy_with_logits(y, skip_forward=self.skip_forward)


class RAPPORLoss(_Loss):
    r"""
    This loss computes the BCEWithLogitsLoss with corrections applied to account
    for randomized response, where the input `alpha` represents the probability
    of flipping a label.

    The loss can be described as:

    .. math::
        p = \sigma(x)

    .. math::
        r = \alpha * p + (1 - \alpha) * (1 - p)

    .. math::
        \ell(x, y) = mean(L) = mean(\{l_1,\dots,l_N\}^\top), \quad
        l_n = - \left [ y_n \cdot \log r_n + (1 - y_n) \cdot \log (1 - r_n) \right ],

    This is used for measuring the error of a reconstruction in for example an
    auto-encoder. Note that the targets t[i] should be numbers between 0 and 1.
    """

    def __init__(self, alpha, reduction="mean", skip_forward=False):
        super(RAPPORLoss, self).__init__(reduction=reduction, skip_forward=skip_forward)
        self.alpha = alpha

    def forward(self, x, y):
        assert x.size() == y.size(), "input and target must have the same size"
        return x.rappor_loss(y, self.alpha, skip_forward=self.skip_forward)
