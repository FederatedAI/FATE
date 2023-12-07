#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fate.arch.protocol import mpc
from ..common.tensor_types import is_tensor

__all__ = [
    "__eq__",
    "__ge__",
    "__gt__",
    "__le__",
    "__lt__",
    "__ne__",
    "abs",
    "eq",
    "ge",
    "gt",
    "hardtanh",
    "le",
    "lt",
    "ne",
    "relu",
    "sign",
    "where",
]


def ge(self, y):
    """Returns self >= y"""
    return 1 - self.lt(y)


def gt(self, y):
    """Returns self > y"""
    return (-self + y)._ltz()


def le(self, y):
    """Returns self <= y"""
    return 1 - self.gt(y)


def lt(self, y):
    """Returns self < y"""
    return (self - y)._ltz()


def eq(self, y):
    """Returns self == y"""
    return 1 - self.ne(y)


def ne(self, y):
    """Returns self != y"""
    difference = self - y
    difference = type(difference).stack([difference, -difference])
    return difference._ltz().sum(0)


__eq__ = eq
__ge__ = ge
__gt__ = gt
__le__ = le
__lt__ = lt
__ne__ = ne


def sign(self):
    """Computes the sign value of a tensor (0 is considered positive)"""
    return 1 - 2 * self._ltz()


def abs(self):
    """Computes the absolute value of a tensor"""
    return self * self.sign()


def relu(self):
    """Compute a Rectified Linear function on the input tensor."""
    return self * self.ge(0)


def hardtanh(self, min_value=-1, max_value=1):
    r"""Applies the HardTanh function element-wise

    HardTanh is defined as:

    .. math::
        \text{HardTanh}(x) = \begin{cases}
            1 & \text{ if } x > 1 \\
            -1 & \text{ if } x < -1 \\
            x & \text{ otherwise } \\
        \end{cases}

    The range of the linear region :math:`[-1, 1]` can be adjusted using
    :attr:`min_val` and :attr:`max_val`.

    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1
    """
    intermediate = mpc.stack([self - min_value, self - max_value]).relu()
    intermediate = intermediate[0].sub(intermediate[1])
    return intermediate.add_(min_value)


def where(self, condition, y):
    """Selects elements from self or y based on condition

    Args:
        condition (torch.bool or MPCTensor): when True yield self,
            otherwise yield y
        y (torch.tensor or MPCTensor): values selected at indices
            where condition is False.

    Returns: MPCTensor or torch.tensor
    """
    if is_tensor(condition):
        condition = condition.float()
        y_masked = y * (1 - condition)
    else:
        # encrypted tensor must be first operand
        y_masked = (1 - condition) * y

    return self * condition + y_masked
