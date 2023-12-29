#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#
#  Copyright 2023 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import torch

from . import communicator as comm
from .common.tensor_types import is_tensor
from .common.util import torch_stack
from .config import cfg
from .cryptensor import CrypTensor
from .cuda import CUDALongTensor
from .encoder import FixedPointEncoder
from .primitives.binary import BinarySharedTensor
from .primitives.converters import convert
from .ptype import ptype as Ptype


@CrypTensor.register_cryptensor("mpc")
class MPCTensor(CrypTensor):
    def __init__(self, ctx, tensor, ptype=Ptype.arithmetic, device=None, *args, **kwargs):
        """
        Creates the shared tensor from the input `tensor` provided by party `src`.
        The `ptype` defines the type of sharing used (default: arithmetic).

        The other parties can specify a `tensor` or `size` to determine the size
        of the shared tensor object to create. In this case, all parties must
        specify the same (tensor) size to prevent the party's shares from varying
        in size, which leads to undefined behavior.

        Alternatively, the parties can set `broadcast_size` to `True` to have the
        `src` party broadcast the correct size. The parties who do not know the
        tensor size beforehand can provide an empty tensor as input. This is
        guaranteed to produce correct behavior but requires an additional
        communication round.

        The parties can also set the `precision` and `device` for their share of
        the tensor. If `device` is unspecified, it is set to `tensor.device`.
        """
        # if tensor is None:
        #     raise ValueError("Cannot initialize tensor with None.")

        # take required_grad from kwargs, input tensor, or set to False:
        default = tensor.requires_grad if torch.is_tensor(tensor) else False
        requires_grad = kwargs.pop("requires_grad", default)

        # call CrypTensor constructor:
        super().__init__(requires_grad=requires_grad)
        if device is None and hasattr(tensor, "device"):
            device = tensor.device

        # create the MPCTensor:
        if tensor is []:
            self._tensor = torch.tensor([], device=device)
        else:
            tensor_type = ptype.to_tensor()
            self._tensor = tensor_type(ctx=ctx, tensor=tensor, device=device, *args, **kwargs)
        self.ptype = ptype
        self.ctx = ctx

    @staticmethod
    def new(*args, **kwargs):
        """
        Creates a new MPCTensor, passing all args and kwargs into the constructor.
        """
        return MPCTensor(*args, **kwargs)

    @staticmethod
    def from_shares(share, precision=None, ptype=Ptype.arithmetic):
        result = MPCTensor([])
        from_shares = ptype.to_tensor().from_shares
        result._tensor = from_shares(share, precision=precision)
        result.ptype = ptype
        return result

    def clone(self):
        """Create a deep copy of the input tensor."""
        # TODO: Rename this to __deepcopy__()?
        result = MPCTensor(self.ctx, [])
        result._tensor = self._tensor.clone()
        result.ptype = self.ptype
        return result

    def shallow_copy(self):
        """Create a shallow copy of the input tensor."""
        # TODO: Rename this to __copy__()?
        result = MPCTensor(self.ctx, [])
        result._tensor = self._tensor
        result.ptype = self.ptype
        return result

    def copy_(self, other):
        """Copies value of other MPCTensor into this MPCTensor."""
        assert isinstance(other, MPCTensor), "other must be MPCTensor"
        self._tensor.copy_(other._tensor)
        self.ptype = other.ptype

    def to(self, *args, **kwargs):
        r"""
        Depending on the input arguments,
        converts underlying share to the given ptype or
        performs `torch.to` on the underlying torch tensor

        To convert underlying share to the given ptype, call `to` as:
            to(ptype, **kwargs)

        It will call MPCTensor.to_ptype with the arguments provided above.

        Otherwise, `to` performs `torch.to` on the underlying
        torch tensor. See
        https://pytorch.org/docs/stable/tensors.html?highlight=#torch.Tensor.to
        for a reference of the parameters that can be passed in.

        Args:
            ptype: Ptype.arithmetic or Ptype.binary.
        """
        if "ptype" in kwargs:
            return self._to_ptype(**kwargs)
        elif args and isinstance(args[0], Ptype):
            ptype = args[0]
            return self._to_ptype(ptype, **kwargs)
        else:
            share = self.share.to(*args, **kwargs)
            if share.is_cuda:
                share = CUDALongTensor(share)
            self.share = share
            return self

    def _to_ptype(self, ptype, **kwargs):
        r"""
        Convert MPCTensor's underlying share to the corresponding ptype
        (ArithmeticSharedTensor, BinarySharedTensor)

        Args:
            ptype (Ptype.arithmetic or Ptype.binary): The ptype to convert
                the shares to.
            precision (int, optional): Precision of the fixed point encoder when
                converting a binary share to an arithmetic share. It will be ignored
                if the ptype doesn't match.
            bits (int, optional): If specified, will only preserve the bottom `bits` bits
                of a binary tensor when converting from a binary share to an arithmetic share.
                It will be ignored if the ptype doesn't match.
        """
        retval = self.clone()
        if retval.ptype == ptype:
            return retval
        retval._tensor = convert(self._tensor, ptype, **kwargs)
        retval.ptype = ptype
        return retval

    @property
    def device(self):
        """Return the `torch.device` of the underlying share"""
        return self.share.device

    @property
    def is_cuda(self):
        """Return True if the underlying share is stored on GPU, False otherwise"""
        return self.share.is_cuda

    def cuda(self, *args, **kwargs):
        """Call `torch.Tensor.cuda` on the underlying share"""
        self.share = CUDALongTensor(self.share.cuda(*args, **kwargs))
        return self

    def cpu(self):
        """Call `torch.Tensor.cpu` on the underlying share"""
        self.share = self.share.cpu()
        return self

    def get_plain_text(self, dst=None, group=None):
        """Decrypts the tensor."""
        return self._tensor.get_plain_text(dst=dst, group=group)

    def reveal(self, dst=None, group=None):
        """Decrypts the tensor without any downscaling."""
        return self._tensor.reveal(dst=dst, group=group)

    def __repr__(self):
        """Returns a representation of the tensor useful for debugging."""
        debug_mode = cfg.safety.mpc.debug.debug_mode

        share = self.share
        plain_text = self._tensor.get_plain_text() if debug_mode else "HIDDEN"
        ptype = self.ptype
        return f"MPCTensor(\n\t_tensor={share}\n" f"\tplain_text={plain_text}\n\tptype={ptype}\n)"

    def __hash__(self):
        return hash(self.share)

    @property
    def share(self):
        """Returns underlying share"""
        return self._tensor.share

    @share.setter
    def share(self, value):
        """Sets share to value"""
        self._tensor.share = value

    @property
    def encoder(self):
        """Returns underlying encoder"""
        return self._tensor.encoder

    @encoder.setter
    def encoder(self, value):
        """Sets encoder to value"""
        self._tensor.encoder = value

    @staticmethod
    def rand(*sizes, device=None):
        """
        Returns a tensor with elements uniformly sampled in [0, 1). The uniform
        random samples are generated by generating random bits using fixed-point
        encoding and converting the result to an ArithmeticSharedTensor.
        """
        rand = MPCTensor([])
        encoder = FixedPointEncoder()
        rand._tensor = BinarySharedTensor.rand(*sizes, bits=encoder._precision_bits, device=device)
        rand._tensor.encoder = encoder
        rand.ptype = Ptype.binary
        return rand.to(Ptype.arithmetic, bits=encoder._precision_bits)

    # Comparators
    def _ltz(self):
        """Returns 1 for elements that are < 0 and 0 otherwise"""
        shift = torch.iinfo(torch.long).bits - 1
        precision = 0 if self.encoder.scale == 1 else None

        result = self._to_ptype(Ptype.binary)
        result.share >>= shift
        result = result._to_ptype(Ptype.arithmetic, precision=precision, bits=1)
        result.encoder._scale = 1
        return result

    def eq(self, y):
        """Returns self == y"""
        if comm.get().get_world_size() == 2:
            return (self - y)._eqz_2PC()

        return 1 - self.ne(y)

    def ne(self, y):
        """Returns self != y"""
        if comm.get().get_world_size() == 2:
            return 1 - self.eq(y)

        difference = self - y
        difference.share = torch_stack([difference.share, -(difference.share)])
        return difference._ltz().sum(0)

    def _eqz_2PC(self):
        """Returns self == 0"""
        # Create BinarySharedTensors from shares
        x0 = MPCTensor(self.share, src=0, ptype=Ptype.binary)
        x1 = MPCTensor(-self.share, src=1, ptype=Ptype.binary)

        # Perform equality testing using binary shares
        x0._tensor = x0._tensor.eq(x1._tensor)
        x0.encoder = self.encoder

        # Convert to Arithmetic sharing
        result = x0.to(Ptype.arithmetic, bits=1)
        result.encoder._scale = 1

        return result

    def div(self, y):
        r"""Divides each element of :attr:`self` with the scalar :attr:`y` or
        each element of the tensor :attr:`y` and returns a new resulting tensor.

        For `y` a scalar:

        .. math::
            \text{out}_i = \frac{\text{self}_i}{\text{y}}

        For `y` a tensor:

        .. math::
            \text{out}_i = \frac{\text{self}_i}{\text{y}_i}

        Note for :attr:`y` a tensor, the shapes of :attr:`self` and :attr:`y` must be
        `broadcastable`_.

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics"""  # noqa: B950
        result = self.clone()
        if isinstance(y, CrypTensor):
            result.share = torch.broadcast_tensors(result.share, y.share)[0].clone()
        elif is_tensor(y):
            result.share = torch.broadcast_tensors(result.share, y)[0].clone()

        if isinstance(y, MPCTensor):
            return result.mul(y.reciprocal())
        result._tensor.div_(y)
        return result


UNARY_FUNCTIONS = [
    "avg_pool2d",
    "square",
    "neg",
]

BINARY_FUNCTIONS = [
    "add",
    "sub",
    "mul",
    "matmul",
    "conv1d",
    "conv2d",
    "conv_transpose1d",
    "conv_transpose2d",
]


def _add_unary_passthrough_function(name):
    def unary_wrapper_function(self, *args, **kwargs):
        result = self.shallow_copy()
        result._tensor = getattr(result._tensor, name)(*args, **kwargs)
        return result

    setattr(MPCTensor, name, unary_wrapper_function)


def _add_binary_passthrough_function(name):
    def binary_wrapper_function(self, value, *args, **kwargs):
        result = self.shallow_copy()
        if isinstance(value, MPCTensor):
            value = value._tensor
        result._tensor = getattr(result._tensor, name)(value, *args, **kwargs)
        return result

    setattr(MPCTensor, name, binary_wrapper_function)


for func_name in UNARY_FUNCTIONS:
    _add_unary_passthrough_function(func_name)

for func_name in BINARY_FUNCTIONS:
    _add_binary_passthrough_function(func_name)
