#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from fate.arch.protocol import mpc

__all__ = ["dropout"]


def dropout(self, p=0.5, training=True, inplace=False):
    r"""
    Randomly zeroes some of the elements of the input tensor with
    probability :attr:`p`.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place.
            Default: ``False``
    """
    if p == 0.0:
        return self
    elif p == 1.0:
        return self - self

    assert p > 0.0 and p < 1.0, "dropout probability has to be between 0 and 1"
    if training and inplace:
        logging.warning("CrypTen dropout does not support inplace computation during training.")

    if not training:
        if inplace:
            return self
        else:
            return self.clone()

    rand_tensor = mpc.rand(self.size(), device=self.device)
    dropout_tensor = rand_tensor > p
    if inplace:
        result_tensor = self.div_(1 - p)
        result_tensor = result_tensor.mul_(dropout_tensor)
    else:
        result_tensor = self.div(1 - p)
        result_tensor = result_tensor.mul(dropout_tensor)
    return result_tensor
