#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

from .primitives import ArithmeticSharedTensor, BinarySharedTensor


class ptype(Enum):
    """Enumeration defining the private type attributes of encrypted tensors"""

    arithmetic = 0
    binary = 1

    def to_tensor(self):
        if self.value == 0:
            return ArithmeticSharedTensor
        elif self.value == 1:
            return BinarySharedTensor
        else:
            raise ValueError("Cannot convert %s to encrypted tensor" % (self.name))
