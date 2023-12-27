#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fate.arch.protocol import mpc
from ..common.tensor_types import is_tensor

__all__ = ["norm", "polynomial", "pos_pow", "pow"]


def pow(self, p, **kwargs):
    """
    Computes an element-wise exponent `p` of a tensor, where `p` is an
    integer.
    """
    if isinstance(p, float) and int(p) == p:
        p = int(p)

    if not isinstance(p, int):
        raise TypeError(
            "pow must take an integer exponent. For non-integer powers, use" " pos_pow with positive-valued base."
        )
    if p < -1:
        return self.reciprocal().pow(-p)
    elif p == -1:
        return self.reciprocal()
    elif p == 0:
        # Note: This returns 0 ** 0 -> 1 when inputs have zeros.
        # This is consistent with PyTorch's pow function.
        return self.new(torch.ones_like(self.data))
    elif p == 1:
        return self.clone()
    elif p == 2:
        return self.square()
    elif p % 2 == 0:
        return self.square().pow(p // 2)
    else:
        x = self.square().mul_(self)
        return x.pow((p - 1) // 2)


def pos_pow(self, p):
    """
    Approximates self ** p by computing: :math:`x^p = exp(p * log(x))`

    Note that this requires that the base `self` contain only positive values
    since log can only be computed on positive numbers.

    Note that the value of `p` can be an integer, float, public tensor, or
    encrypted tensor.
    """
    if isinstance(p, int) or (isinstance(p, float) and int(p) == p):
        return self.pow(p)
    return self.log().mul_(p).exp()


def polynomial(self, coeffs, func="mul"):
    """Computes a polynomial function on a tensor with given coefficients,
    `coeffs`, that can be a list of values or a 1-D tensor.

    Coefficients should be ordered from the order 1 (linear) term first,
    ending with the highest order term. (Constant is not included).
    """
    # Coefficient input type-checking
    if isinstance(coeffs, list):
        coeffs = torch.tensor(coeffs, device=self.device)
    assert is_tensor(coeffs) or mpc.is_encrypted_tensor(coeffs), "Polynomial coefficients must be a list or tensor"
    assert coeffs.dim() == 1, "Polynomial coefficients must be a 1-D tensor"

    # Handle linear case
    if coeffs.size(0) == 1:
        return self.mul(coeffs)

    # Compute terms of polynomial using exponentially growing tree
    terms = mpc.stack([self, self.square()])
    while terms.size(0) < coeffs.size(0):
        highest_term = terms.index_select(0, torch.tensor(terms.size(0) - 1, device=self.device))
        new_terms = getattr(terms, func)(highest_term)
        terms = mpc.cat([terms, new_terms])

    # Resize the coefficients for broadcast
    terms = terms[: coeffs.size(0)]
    for _ in range(terms.dim() - 1):
        coeffs = coeffs.unsqueeze(1)

    # Multiply terms by coefficients and sum
    return terms.mul(coeffs).sum(0)


def norm(self, p="fro", dim=None, keepdim=False):
    """Computes the p-norm of the input tensor (or along a dimension)."""
    if p == "fro":
        p = 2

    if isinstance(p, (int, float)):
        assert p >= 1, "p-norm requires p >= 1"
        if p == 1:
            if dim is None:
                return self.abs().sum()
            return self.abs().sum(dim, keepdim=keepdim)
        elif p == 2:
            if dim is None:
                return self.square().sum().sqrt()
            return self.square().sum(dim, keepdim=keepdim).sqrt()
        elif p == float("inf"):
            if dim is None:
                return self.abs().max()
            return self.abs().max(dim=dim, keepdim=keepdim)[0]
        else:
            if dim is None:
                return self.abs().pos_pow(p).sum().pos_pow(1 / p)
            return self.abs().pos_pow(p).sum(dim, keepdim=keepdim).pos_pow(1 / p)
    elif p == "nuc":
        raise NotImplementedError("Nuclear norm is not implemented")
    else:
        raise ValueError(f"Improper value p ({p})for p-norm")
