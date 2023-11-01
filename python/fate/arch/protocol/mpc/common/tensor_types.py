#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fate.arch.protocol.mpc.cuda import CUDALongTensor


# helper functions that determine if input is float, int, or base tensor:
def _is_type_tensor(tensor, types):
    """Checks whether the elements of the input tensor are of a given type"""
    if is_tensor(tensor):
        if any(tensor.dtype == type_ for type_ in types):
            return True
    return False


def is_tensor(tensor):
    """Checks if the input tensor is a Torch tensor or a CUDALongTensor"""
    from fate.arch.tensor import DTensor

    return torch.is_tensor(tensor) or isinstance(tensor, CUDALongTensor) or isinstance(tensor, DTensor)


def is_float_tensor(tensor):
    """Checks if the input tensor is a Torch tensor of a float type."""
    return _is_type_tensor(tensor, [torch.float16, torch.float32, torch.float64])


def is_int_tensor(tensor):
    """Checks if the input tensor is a Torch tensor of an int type."""
    return _is_type_tensor(tensor, [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
