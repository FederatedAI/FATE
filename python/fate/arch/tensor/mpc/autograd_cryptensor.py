#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
import torch

from .gradients import AutogradContext as _AutogradContext


class AutogradContext(_AutogradContext):
    """
    DEPRECATED: Object used by AutogradFunctions for saving context information.
    """

    def __init__(self):
        raise DeprecationWarning(
            "crypten.autograd_cryptensor.AutogradContext is deprecated. Please "
            "use crypten.gradients.AutogradContext instead."
        )
        super().__init__(self)


def AutogradCrypTensor(tensor, requires_grad=True):
    """
    DEPRECATED: CrypTensor with support for autograd, akin to the `Variable`
    originally in PyTorch.
    """
    raise DeprecationWarning(
        "AutogradCrypTensor is deprecated. Please set the "
        "requires_grad attribute on the CrypTensor instead."
    )
    if torch.is_tensor(tensor):
        tensor = crypten.cryptensor(tensor)
    tensor.requires_grad = requires_grad
    return tensor
