#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fate.arch.protocol.mpc.encoder import FixedPointEncoder
from . import beaver
from .arithmetic import ArithmeticSharedTensor
from .binary import BinarySharedTensor
from .. import communicator as comm
from ..ptype import ptype as Ptype


def _A2B(arithmetic_tensor):
    # first try memory-inefficient implementation that takes O(log P) rounds:
    try:
        binary_tensor = BinarySharedTensor.stack(
            [BinarySharedTensor(arithmetic_tensor.share, src=i) for i in range(comm.get().get_world_size())]
        )
        binary_tensor = binary_tensor.sum(dim=0)

    # if we OOM, try memory-efficient implementation that uses O(P) rounds:
    except RuntimeError:
        binary_tensor = None
        for i in range(comm.get().get_world_size()):
            binary_share = BinarySharedTensor(arithmetic_tensor.share, src=i)
            binary_tensor = binary_share if i == 0 else binary_tensor + binary_share

    # return the result:
    binary_tensor.encoder = arithmetic_tensor.encoder
    return binary_tensor


def _B2A(binary_tensor, precision=None, bits=None):
    if bits is None:
        bits = torch.iinfo(torch.long).bits

    if bits == 1:
        binary_bit = binary_tensor & 1
        arithmetic_tensor = beaver.B2A_single_bit(binary_bit)
    else:
        binary_bits = BinarySharedTensor.stack([binary_tensor >> i for i in range(bits)])
        binary_bits = binary_bits & 1
        arithmetic_bits = beaver.B2A_single_bit(binary_bits)

        multiplier = torch.cat(
            [torch.tensor([1], dtype=torch.long, device=binary_tensor.device) << i for i in range(bits)]
        )
        while multiplier.dim() < arithmetic_bits.dim():
            multiplier = multiplier.unsqueeze(1)

        arithmetic_tensor = arithmetic_bits.mul_(multiplier).sum(0)

    arithmetic_tensor.encoder = FixedPointEncoder(precision_bits=precision)
    scale = arithmetic_tensor.encoder._scale // binary_tensor.encoder._scale
    arithmetic_tensor *= scale
    return arithmetic_tensor


def convert(tensor, ptype, **kwargs):
    tensor_name = ptype.to_tensor()
    if isinstance(tensor, tensor_name):
        return tensor
    if isinstance(tensor, ArithmeticSharedTensor) and ptype == Ptype.binary:
        return _A2B(tensor)
    elif isinstance(tensor, BinarySharedTensor) and ptype == Ptype.arithmetic:
        return _B2A(tensor, **kwargs)
    else:
        raise TypeError("Cannot convert %s to %s" % (type(tensor), ptype.__name__))
