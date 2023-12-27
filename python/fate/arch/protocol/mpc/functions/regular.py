#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from ..common.tensor_types import is_tensor
from ..common.util import torch_cat, torch_stack

__all__ = [  # noqa: F822
    "__getitem__",
    "__len__",
    "__setitem__",
    "cat",
    "cumsum",
    "dim",
    "dot",
    "expand",
    "flatten",
    "flip",
    "gather",
    "ger",
    "index_add",
    "index_select",
    "mean",
    "narrow",
    "nelement",
    "numel",
    "pad",
    "permute",
    "prod",
    "repeat",
    "reshape",
    "roll",
    "scatter",
    "scatter_add",
    "size",
    "split",
    "squeeze",
    "stack",
    "sum",
    "t",
    "take",
    "trace",
    "transpose",
    "unbind",
    "unfold",
    "unsqueeze",
    "var",
    "view",
]


PROPERTY_FUNCTIONS = ["__len__", "nelement", "dim", "size", "numel"]


def __setitem__(self, index, value):
    """Set tensor values by index"""
    if not isinstance(value, type(self)):
        kwargs = {"device": self.device}
        if hasattr(self, "ptype"):
            kwargs["ptype"] = self.ptype
        value = self.new(value, **kwargs)
    self._tensor.__setitem__(index, value._tensor)


def pad(self, pad, mode="constant", value=0):
    result = self.shallow_copy()
    if hasattr(value, "_tensor"):
        value = value._tensor

    if hasattr(result._tensor, "pad"):
        result._tensor = self._tensor.pad(pad, mode=mode, value=value)
    else:
        result._tensor = torch.nn.functional.pad(self._tensor, pad, mode=mode, value=value)
    return result


def index_add(self, dim, index, tensor):
    """Performs out-of-place index_add: Accumulate the elements of tensor into the
    self tensor by adding to the indices in the order given in index.
    """
    result = self.clone()
    assert index.dim() == 1, "index needs to be a vector"
    tensor = getattr(tensor, "_tensor", tensor)
    result._tensor.index_add_(dim, index, tensor)
    return result


def scatter_add(self, dim, index, other):
    """Adds all values from the tensor other into self at the indices
    specified in the index tensor in a similar fashion as scatter_(). For
    each value in other, it is added to an index in self which is specified
    by its index in other for dimension != dim and by the corresponding
    value in index for dimension = dim.
    """
    result = self.clone()
    other = getattr(other, "_tensor", other)
    result._tensor.scatter_add_(dim, index, other)
    return result


def scatter(self, dim, index, src):
    """Out-of-place version of :meth:`CrypTensor.scatter_`"""
    result = self.clone()
    if is_tensor(src):
        src = self.new(src)
    assert isinstance(src, type(self)), "Unrecognized scatter src type: %s" % type(src)
    result._tensor.scatter_(dim, index, src._tensor)
    return result


def unbind(self, dim=0):
    tensors = self._tensor.unbind(dim=dim)
    results = tuple(self.shallow_copy() for _ in range(len(tensors)))
    for i in range(len(tensors)):
        results[i]._tensor = tensors[i]
    return results


def split(self, split_size, dim=0):
    tensors = self._tensor.split(split_size, dim=dim)
    results = tuple(self.shallow_copy() for _ in range(len(tensors)))
    for i in range(len(tensors)):
        results[i]._tensor = tensors[i]
    return results


def take(self, index, dimension=None):
    """Take entries of tensor along a dimension according to the index.
    This function is identical to torch.take() when dimension=None,
    otherwise, it is identical to ONNX gather() function.
    """
    result = self.shallow_copy()
    index = index.long()
    if dimension is None or self.dim() == 0:
        result._tensor = self._tensor.take(index)
    else:
        all_indices = [slice(0, x) for x in self.size()]
        all_indices[dimension] = index
        result._tensor = self._tensor[all_indices]
    return result


def mean(self, *args, **kwargs):
    """Computes mean of given tensor"""
    result = self.sum(*args, **kwargs)

    # Handle special case where input has 0 dimensions
    if self.dim() == 0:
        return result

    # Compute divisor to use to compute mean
    divisor = self.nelement() // result.nelement()
    return result.div(divisor)


def var(self, *args, **kwargs):
    """Computes variance of tensor along specified dimensions."""
    # preprocess inputs:
    if len(args) == 0:
        dim = None
        unbiased = kwargs.get("unbiased", False)
        mean = self.mean()
    elif len(args) == 1:
        dim = args[0]
        unbiased = kwargs.get("unbiased", False)
        keepdim = kwargs.get("keepdim", False)
    elif len(args) == 2:
        dim, unbiased = args[0], args[1]
        keepdim = kwargs.get("keepdim", False)
    else:
        dim, unbiased, keepdim = args[0], args[1], args[2]

    if dim is not None:  # dimension is specified
        mean = self.mean(dim, keepdim=True)

    # Compute square error
    result = (self - mean).square()
    if dim is None:
        result = result.sum()
    else:
        result = result.sum(dim, keepdim=keepdim)

    # Determine divisor
    divisor = self.nelement() // result.nelement()
    if not unbiased:
        divisor -= 1

    # Compute mean square error
    if divisor in [0, 1]:
        return result
    return result.div(divisor)


def prod(self, dim=None, keepdim=False):
    """
    Returns the product of each row of the `input` tensor in the given
    dimension `dim`.

    If `keepdim` is `True`, the output tensor is of the same size as `input`
    except in the dimension `dim` where it is of size 1. Otherwise, `dim` is
    squeezed, resulting in the output tensor having 1 fewer dimension than
    `input`.
    """
    if dim is None:
        return self.flatten().prod(dim=0)

    result = self.clone()
    while result.size(dim) > 1:
        size = result.size(dim)
        x, y, remainder = result.split([size // 2, size // 2, size % 2], dim=dim)
        result = x.mul_(y)
        result = type(self).cat([result, remainder], dim=dim)

    # Squeeze result if necessary
    if not keepdim:
        result = result.squeeze(dim)
    return result


def dot(self, y, weights=None):
    """Compute a dot product between two tensors"""
    assert self.size() == y.size(), "Number of elements do not match"
    if weights is not None:
        assert weights.size() == self.size(), "Incorrect number of weights"
        result = self * weights
    else:
        result = self.clone()

    return result.mul(y).sum()


def ger(self, y):
    """Computer an outer product between two vectors"""
    assert self.dim() == 1 and y.dim() == 1, "Outer product must be on 1D tensors"
    return self.view((-1, 1)).matmul(y.view((1, -1)))


def __cat_stack_helper(op, tensors, *args, **kwargs):
    assert op in ["cat", "stack"], "Unsupported op for helper function"
    assert isinstance(tensors, list), "%s input must be a list" % op
    assert len(tensors) > 0, "expected a non-empty list of CrypTensors"

    # Determine op-type
    funcs = {"cat": torch_cat, "stack": torch_stack}
    func = funcs[op]
    if hasattr(tensors[0]._tensor, op):
        func = getattr(tensors[0]._tensor, op)

    # type coordination
    for i, tensor in enumerate(tensors[1:]):
        if torch.is_tensor(tensor) or isinstance(tensor, (int, float)):
            tensors[i] = tensors[0].new(tensor)
        assert isinstance(tensors[i], type(tensors[0])), f"{op} tensor type mismatch"

    # Operate on all input tensors
    result = tensors[0].clone()
    result._tensor = func([tensor._tensor for tensor in tensors], *args, **kwargs)
    return result


def cat(tensors, *args, **kwargs):
    """Perform tensor concatenation"""
    return __cat_stack_helper("cat", tensors, *args, **kwargs)


def stack(tensors, *args, **kwargs):
    """Perform tensor stacking"""
    return __cat_stack_helper("stack", tensors, *args, **kwargs)


# Make static methods static
cat = staticmethod(cat)
stack = staticmethod(stack)


# Add remaining regular functions
def _add_regular_function(function_name):
    """
    Adds regular function that is applied directly on the underlying
    `_tensor` attribute, and stores the result in the same attribute.
    """

    def regular_func(self, *args, **kwargs):
        result = self.shallow_copy()
        result._tensor = getattr(result._tensor, function_name)(*args, **kwargs)
        return result

    if function_name not in globals():
        globals()[function_name] = regular_func


def _add_property_function(function_name):
    """
    Adds regular function that is applied directly on the underlying
    `_tensor` attribute, and returns the result of that function.
    """

    def property_func(self, *args, **kwargs):
        return getattr(self._tensor, function_name)(*args, **kwargs)

    if function_name not in globals():
        globals()[function_name] = property_func


for function_name in __all__:
    if function_name in PROPERTY_FUNCTIONS:
        continue
    _add_regular_function(function_name)

for function_name in PROPERTY_FUNCTIONS:
    _add_property_function(function_name)
