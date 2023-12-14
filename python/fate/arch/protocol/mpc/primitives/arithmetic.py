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

import logging

# dependencies:
import torch

import fate.arch.protocol.mpc.communicator as comm
from fate.arch.context import Context
from fate.arch.protocol.mpc.common.rng import generate_random_ring_element
from fate.arch.protocol.mpc.common.tensor_types import is_float_tensor, is_int_tensor, is_tensor
from fate.arch.protocol.mpc.common.util import torch_stack
from fate.arch.protocol.mpc.config import cfg
from fate.arch.protocol.mpc.cryptensor import CrypTensor
from fate.arch.protocol.mpc.cuda import CUDALongTensor
from fate.arch.protocol.mpc.encoder import FixedPointEncoder
from fate.arch.protocol.mpc.functions import regular
from . import beaver, replicated  # noqa: F401

logger = logging.getLogger(__name__)

SENTINEL = -1


# MPC tensor where shares additive-sharings.
class ArithmeticSharedTensor(object):
    """
    Encrypted tensor object that uses additive sharing to perform computations.

    Additive shares are computed by splitting each value of the input tensor
    into n separate random values that add to the input tensor, where n is
    the number of parties present in the protocol (world_size).
    """

    # constructors:
    def __init__(
        self,
        ctx,
        tensor=None,
        size=None,
        broadcast_size=False,
        precision=None,
        src=0,
        device=None,
    ):
        """
        Creates the shared tensor from the input `tensor` provided by party `src`.

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

        if not isinstance(ctx, Context):
            logger.warning("ctx must be a Context object")
        self._ctx = ctx
        # do nothing if source is sentinel:
        if src == SENTINEL:
            return

        # assertions on inputs:
        assert (
            isinstance(src, int) and src >= 0 and src < comm.get().get_world_size()
        ), "specified source party does not exist"
        if self.rank == src:
            assert tensor is not None, "source must provide a data tensor"
            if hasattr(tensor, "src"):
                assert tensor.src == src, "source of data tensor must match source of encryption"
        if not broadcast_size:
            assert tensor is not None or size is not None, "must specify tensor or size, or set broadcast_size"

        # if device is unspecified, try and get it from tensor:
        if device is None and tensor is not None and hasattr(tensor, "device"):
            device = tensor.device

        # encode the input tensor:
        self.encoder = FixedPointEncoder(precision_bits=precision)
        if tensor is not None:
            if is_int_tensor(tensor) and precision != 0:
                tensor = tensor.float()
            tensor = self.encoder.encode(tensor)
            tensor = tensor.to(device=device)
            size = tensor.size()

        # if other parties do not know tensor's size, broadcast the size:
        if broadcast_size:
            size = comm.get().broadcast_obj(src, size)

        # generate pseudo-random zero sharing (PRZS) and add source's tensor:
        self.share = ArithmeticSharedTensor.PRZS(ctx, size, device=device).share
        if self.rank == src:
            self.share += tensor

    @staticmethod
    def new(*args, **kwargs):
        """
        Creates a new ArithmeticSharedTensor, passing all args and kwargs into the constructor.
        """
        return ArithmeticSharedTensor(*args, **kwargs)

    @property
    def device(self):
        """Return the `torch.device` of the underlying _tensor"""
        return self._tensor.device

    @property
    def is_cuda(self):
        """Return True if the underlying _tensor is stored on GPU, False otherwise"""
        return self._tensor.is_cuda

    def to(self, *args, **kwargs):
        """Call `torch.Tensor.to` on the underlying _tensor"""
        self._tensor = self._tensor.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        """Call `torch.Tensor.cuda` on the underlying _tensor"""
        self._tensor = CUDALongTensor(self._tensor.cuda(*args, **kwargs))
        return self

    def cpu(self, *args, **kwargs):
        """Call `torch.Tensor.cpu` on the underlying _tensor"""
        self._tensor = self._tensor.cpu(*args, **kwargs)
        return self

    @property
    def share(self):
        """Returns underlying _tensor"""
        return self._tensor

    @share.setter
    def share(self, value):
        """Sets _tensor to value"""
        self._tensor = value

    @staticmethod
    def from_shares(ctx, share, precision=None, device=None):
        """Generate an ArithmeticSharedTensor from a share from each party"""
        result = ArithmeticSharedTensor(ctx, src=SENTINEL)
        share = share.to(device) if device is not None else share
        result.share = CUDALongTensor(share) if share.is_cuda else share
        result.encoder = FixedPointEncoder(precision_bits=precision)
        return result

    @staticmethod
    def PRZS(ctx, *size, device=None):
        """
        Generate a Pseudo-random Sharing of Zero (using arithmetic shares)

        This function does so by generating `n` numbers across `n` parties with
        each number being held by exactly 2 parties. One of these parties adds
        this number while the other subtracts this number.
        """
        from fate.arch.protocol.mpc import generators

        tensor = ArithmeticSharedTensor(ctx, src=SENTINEL)
        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        g0 = generators["prev"][device]
        g1 = generators["next"][device]
        current_share = generate_random_ring_element(ctx, *size, generator=g0, device=device)
        next_share = generate_random_ring_element(ctx, *size, generator=g1, device=device)
        tensor.share = current_share - next_share
        return tensor

    @staticmethod
    def PRSS(ctx, *size, device=None):
        """
        Generates a Pseudo-random Secret Share from a set of random arithmetic shares
        """
        share = generate_random_ring_element(ctx, *size, device=device)
        tensor = ArithmeticSharedTensor.from_shares(share=share)
        return tensor

    @property
    def rank(self):
        return comm.get().get_rank()

    def shallow_copy(self):
        """Create a shallow copy"""
        result = ArithmeticSharedTensor(ctx=self._ctx, src=SENTINEL)
        result.encoder = self.encoder
        result._tensor = self._tensor
        return result

    def clone(self):
        result = ArithmeticSharedTensor(ctx=self._ctx, src=SENTINEL)
        result.encoder = self.encoder
        result._tensor = self._tensor.clone()
        return result

    def copy_(self, other):
        """Copies other tensor into this tensor."""
        self.share.copy_(other.share)
        self.encoder = other.encoder

    def __repr__(self):
        return f"ArithmeticSharedTensor({self.share})"

    def __bool__(self):
        """Override bool operator since encrypted tensors cannot evaluate"""
        raise RuntimeError("Cannot evaluate ArithmeticSharedTensors to boolean values")

    def __nonzero__(self):
        """__bool__ for backwards compatibility with Python 2"""
        raise RuntimeError("Cannot evaluate ArithmeticSharedTensors to boolean values")

    def __setitem__(self, index, value):
        """Set tensor values by index"""
        if isinstance(value, (int, float)) or is_tensor(value):
            value = ArithmeticSharedTensor(value)
        assert isinstance(value, ArithmeticSharedTensor), "Unsupported input type %s for __setitem__" % type(value)
        self.share.__setitem__(index, value.share)

    def pad(self, pad, mode="constant", value=0):
        """
        Pads the input tensor with values provided in `value`.
        """
        assert mode == "constant", "Padding with mode %s is currently unsupported" % mode

        result = self.shallow_copy()
        if isinstance(value, (int, float)):
            value = self.encoder.encode(value).item()
            if result.rank == 0:
                result.share = torch.nn.functional.pad(result.share, pad, mode=mode, value=value)
            else:
                result.share = torch.nn.functional.pad(result.share, pad, mode=mode, value=0)
        elif isinstance(value, ArithmeticSharedTensor):
            assert value.dim() == 0, "Private values used for padding must be 0-dimensional"
            value = value.share.item()
            result.share = torch.nn.functional.pad(result.share, pad, mode=mode, value=value)
        else:
            raise TypeError("Cannot pad ArithmeticSharedTensor with a %s value" % type(value))

        return result

    @staticmethod
    def stack(tensors, *args, **kwargs):
        """Perform tensor stacking"""
        for i, tensor in enumerate(tensors):
            if is_tensor(tensor):
                tensors[i] = ArithmeticSharedTensor(tensor)
            assert isinstance(tensors[i], ArithmeticSharedTensor), "Can't stack %s with ArithmeticSharedTensor" % type(
                tensor
            )

        result = tensors[0].shallow_copy()
        result.share = torch_stack([tensor.share for tensor in tensors], *args, **kwargs)
        return result

    @staticmethod
    def reveal_batch(tensor_or_list, dst=None):
        """Get (batched) plaintext without any downscaling"""
        if isinstance(tensor_or_list, ArithmeticSharedTensor):
            return tensor_or_list.reveal(dst=dst)

        assert isinstance(tensor_or_list, list), f"Invalid input type into reveal {type(tensor_or_list)}"
        shares = [tensor.share for tensor in tensor_or_list]
        if dst is None:
            return comm.get().all_reduce(shares, batched=True)
        else:
            return comm.get().reduce(shares, dst, batched=True)

    def reveal(self, dst=None, group=None):
        """Decrypts the tensor without any downscaling."""
        tensor = self.share.clone()
        if dst is None:
            return comm.get().all_reduce(tensor, group=group)
        else:
            return comm.get().reduce(tensor, dst, group=group)

    def get_plain_text(self, dst=None, group=None):
        """Decrypts the tensor."""
        # Edge case where share becomes 0 sized (e.g. result of split)
        if self.nelement() < 1:
            return torch.empty(self.share.size())
        return self.encoder.decode(self.reveal(dst=dst, group=group))

    def encode_(self, new_encoder):
        """Rescales the input to a new encoding in-place"""
        if self.encoder.scale == new_encoder.scale:
            return self
        elif self.encoder.scale < new_encoder.scale:
            scale_factor = new_encoder.scale // self.encoder.scale
            self.share *= scale_factor
        else:
            scale_factor = self.encoder.scale // new_encoder.scale
            self = self.div_(scale_factor)
        self.encoder = new_encoder
        return self

    def encode(self, new_encoder):
        """Rescales the input to a new encoding"""
        return self.clone().encode_(new_encoder)

    def encode_as_(self, other):
        """Rescales self to have the same encoding as other"""
        return self.encode_(other.encoder)

    def encode_as(self, other):
        return self.encode(other.encoder)

    def _arithmetic_function_(self, y, op, *args, **kwargs):
        return self._arithmetic_function(y, op, inplace=True, *args, **kwargs)

    def _arithmetic_function(self, y, op, inplace=False, *args, **kwargs):  # noqa:C901
        assert op in [
            "add",
            "sub",
            "mul",
            "matmul",
            "rmatmul",
            "conv1d",
            "conv2d",
            "conv_transpose1d",
            "conv_transpose2d",
        ], f"Provided op `{op}` is not a supported arithmetic function"

        additive_func = op in ["add", "sub"]
        public = isinstance(y, (int, float)) or is_tensor(y)
        private = isinstance(y, ArithmeticSharedTensor)

        if inplace:
            result = self
            if additive_func or (op == "mul" and public):
                op += "_"
        else:
            result = self.clone()

        if public:
            y = result.encoder.encode(y, device=self.device)

            if additive_func:  # ['add', 'sub']
                if result.rank == 0:
                    result.share = getattr(result.share, op)(y)
                else:
                    result.share = torch.broadcast_tensors(result.share, y)[0]
            elif op == "mul_":  # ['mul_']
                result.share = result.share.mul_(y)
            else:  # ['mul', 'matmul', 'convNd', 'conv_transposeNd']
                if op == "rmatmul":
                    result.share = torch.matmul(y, result.share, *args, **kwargs)
                else:
                    result.share = getattr(torch, op)(result.share, y, *args, **kwargs)
        elif private:
            if additive_func:  # ['add', 'sub', 'add_', 'sub_']
                # Re-encode if necessary:
                if self.encoder.scale > y.encoder.scale:
                    y.encode_as_(result)
                elif self.encoder.scale < y.encoder.scale:
                    result.encode_as_(y)
                result.share = getattr(result.share, op)(y.share)
            else:  # ['mul', 'matmul', 'convNd', 'conv_transposeNd']
                protocol = globals()[cfg.safety.mpc.protocol]
                tmp = getattr(protocol, op)(self._ctx, result, y, *args, **kwargs)
                result.share = tmp.share
        else:
            raise TypeError("Cannot %s %s with %s" % (op, type(y), type(self)))

        # Scale by encoder scale if necessary
        if not additive_func:
            if public:  # scale by self.encoder.scale
                if self.encoder.scale > 1:
                    return result.div_(result.encoder.scale)
                else:
                    result.encoder = self.encoder
            else:  # scale by larger of self.encoder.scale and y.encoder.scale
                if self.encoder.scale > 1 and y.encoder.scale > 1:
                    return result.div_(result.encoder.scale)
                elif self.encoder.scale > 1:
                    result.encoder = self.encoder
                else:
                    result.encoder = y.encoder

        return result

    def add(self, y):
        """Perform element-wise addition"""
        return self._arithmetic_function(y, "add")

    def add_(self, y):
        """Perform element-wise addition"""
        return self._arithmetic_function_(y, "add")

    def sub(self, y):
        """Perform element-wise subtraction"""
        return self._arithmetic_function(y, "sub")

    def sub_(self, y):
        """Perform element-wise subtraction"""
        return self._arithmetic_function_(y, "sub")

    def mul(self, y):
        """Perform element-wise multiplication"""
        if isinstance(y, int):
            result = self.clone()
            result.share = self.share * y
            return result
        return self._arithmetic_function(y, "mul")

    def mul_(self, y):
        """Perform element-wise multiplication"""
        if isinstance(y, int) or is_int_tensor(y):
            self.share *= y
            return self
        return self._arithmetic_function_(y, "mul")

    def div(self, y):
        """Divide by a given tensor"""
        result = self.clone()
        if isinstance(y, CrypTensor):
            result.share = torch.broadcast_tensors(result.share, y.share)[0].clone()
        elif is_tensor(y):
            result.share = torch.broadcast_tensors(result.share, y)[0].clone()
        return result.div_(y)

    def div_(self, y):
        """Divide two tensors element-wise"""
        # TODO: Add test coverage for this code path (next 4 lines)
        if isinstance(y, float) and int(y) == y:
            y = int(y)
        if is_float_tensor(y) and y.frac().eq(0).all():
            y = y.long()

        if isinstance(y, int) or is_int_tensor(y):
            validate = cfg.safety.mpc.debug.validation_mode

            if validate:
                tolerance = 1.0
                tensor = self.get_plain_text()

            # Truncate protocol for dividing by public integers:
            if comm.get().get_world_size() > 2:
                protocol = globals()[cfg.safety.mpc.protocol]
                protocol.truncate(self, y)
            else:
                self.share = self.share.div_(y, rounding_mode="trunc")

            # Validate
            if validate:
                if not torch.lt(torch.abs(self.get_plain_text() * y - tensor), tolerance).all():
                    raise ValueError("Final result of division is incorrect.")

            return self

        # Otherwise multiply by reciprocal
        if isinstance(y, float):
            y = torch.tensor([y], dtype=torch.float, device=self.device)

        assert is_float_tensor(y), "Unsupported type for div_: %s" % type(y)
        return self.mul_(y.reciprocal())

    def matmul(self, y):
        """Perform matrix multiplication using some tensor"""
        return self._arithmetic_function(y, "matmul")

    def rmatmul(self, y):
        """Perform right matrix multiplication using some tensor"""
        return self._arithmetic_function(y, "rmatmul")

    def conv1d(self, kernel, **kwargs):
        """Perform a 1D convolution using the given kernel"""
        return self._arithmetic_function(kernel, "conv1d", **kwargs)

    def conv2d(self, kernel, **kwargs):
        """Perform a 2D convolution using the given kernel"""
        return self._arithmetic_function(kernel, "conv2d", **kwargs)

    def conv_transpose1d(self, kernel, **kwargs):
        """Perform a 1D transpose convolution (deconvolution) using the given kernel"""
        return self._arithmetic_function(kernel, "conv_transpose1d", **kwargs)

    def conv_transpose2d(self, kernel, **kwargs):
        """Perform a 2D transpose convolution (deconvolution) using the given kernel"""
        return self._arithmetic_function(kernel, "conv_transpose2d", **kwargs)

    def index_add(self, dim, index, tensor):
        """Perform out-of-place index_add: Accumulate the elements of tensor into the
        self tensor by adding to the indices in the order given in index."""
        result = self.clone()
        return result.index_add_(dim, index, tensor)

    def index_add_(self, dim, index, tensor):
        """Perform in-place index_add: Accumulate the elements of tensor into the
        self tensor by adding to the indices in the order given in index."""
        public = isinstance(tensor, (int, float)) or is_tensor(tensor)
        private = isinstance(tensor, ArithmeticSharedTensor)
        if public:
            enc_tensor = self.encoder.encode(tensor)
            if self.rank == 0:
                self._tensor.index_add_(dim, index, enc_tensor)
        elif private:
            self._tensor.index_add_(dim, index, tensor._tensor)
        else:
            raise TypeError("index_add second tensor of unsupported type")
        return self

    def scatter_add(self, dim, index, other):
        """Adds all values from the tensor other into self at the indices
        specified in the index tensor in a similar fashion as scatter_(). For
        each value in other, it is added to an index in self which is specified
        by its index in other for dimension != dim and by the corresponding
        value in index for dimension = dim.
        """
        return self.clone().scatter_add_(dim, index, other)

    def scatter_add_(self, dim, index, other):
        """Adds all values from the tensor other into self at the indices
        specified in the index tensor in a similar fashion as scatter_(). For
        each value in other, it is added to an index in self which is specified
        by its index in other for dimension != dim and by the corresponding
        value in index for dimension = dim.
        """
        public = isinstance(other, (int, float)) or is_tensor(other)
        private = isinstance(other, ArithmeticSharedTensor)
        if public:
            if self.rank == 0:
                self.share.scatter_add_(dim, index, self.encoder.encode(other))
        elif private:
            self.share.scatter_add_(dim, index, other.share)
        else:
            raise TypeError("scatter_add second tensor of unsupported type")
        return self

    def avg_pool2d(self, kernel_size, stride=None, padding=0, ceil_mode=False):
        """Perform an average pooling on each 2D matrix of the given tensor

        Args:
            kernel_size (int or tuple): pooling kernel size.
        """
        # TODO: Add check for whether ceil_mode would change size of output and allow ceil_mode when it wouldn't
        if ceil_mode:
            raise NotImplementedError("CrypTen does not support `ceil_mode` for `avg_pool2d`")

        z = self._sum_pool2d(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)
        if isinstance(kernel_size, (int, float)):
            pool_size = kernel_size**2
        else:
            pool_size = kernel_size[0] * kernel_size[1]
        return z / pool_size

    def _sum_pool2d(self, kernel_size, stride=None, padding=0, ceil_mode=False):
        """Perform a sum pooling on each 2D matrix of the given tensor"""
        result = self.shallow_copy()

        result.share = torch.nn.functional.avg_pool2d(
            self.share,
            kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            divisor_override=1,
        )
        return result

    # negation and reciprocal:
    def neg_(self):
        """Negate the tensor's values"""
        self.share.neg_()
        return self

    def neg(self):
        """Negate the tensor's values"""
        return self.clone().neg_()

    def square_(self):
        protocol = globals()[cfg.safety.mpc.protocol]
        self.share = protocol.square(self._ctx, self).div_(self.encoder.scale).share
        return self

    def square(self):
        return self.clone().square_()

    def where(self, condition, y):
        """Selects elements from self or y based on condition

        Args:
            condition (torch.bool or ArithmeticSharedTensor): when True
                yield self, otherwise yield y.
            y (torch.tensor or ArithmeticSharedTensor): values selected at
                indices where condition is False.

        Returns: ArithmeticSharedTensor or torch.tensor
        """
        if is_tensor(condition):
            condition = condition.float()
            y_masked = y * (1 - condition)
        else:
            # encrypted tensor must be first operand
            y_masked = (1 - condition) * y

        return self * condition + y_masked

    def scatter_(self, dim, index, src):
        """Writes all values from the tensor `src` into `self` at the indices
        specified in the `index` tensor. For each value in `src`, its output index
        is specified by its index in `src` for `dimension != dim` and by the
        corresponding value in `index` for `dimension = dim`.
        """
        if is_tensor(src):
            src = ArithmeticSharedTensor(src)
        assert isinstance(src, ArithmeticSharedTensor), "Unrecognized scatter src type: %s" % type(src)
        self.share.scatter_(dim, index, src.share)
        return self

    def scatter(self, dim, index, src):
        """Writes all values from the tensor `src` into `self` at the indices
        specified in the `index` tensor. For each value in `src`, its output index
        is specified by its index in `src` for `dimension != dim` and by the
        corresponding value in `index` for `dimension = dim`.
        """
        result = self.clone()
        return result.scatter_(dim, index, src)

    # overload operators:
    __add__ = add
    __iadd__ = add_
    __radd__ = __add__
    __sub__ = sub
    __isub__ = sub_
    __mul__ = mul
    __imul__ = mul_
    __rmul__ = __mul__
    __div__ = div
    __truediv__ = div
    __itruediv__ = div_
    __neg__ = neg

    def __rsub__(self, tensor):
        """Subtracts self from tensor."""
        return -self + tensor

    @property
    def data(self):
        return self._tensor.data

    @data.setter
    def data(self, value):
        self._tensor.set_(value)


# Register regular functions
for func in regular.__all__:
    if not hasattr(ArithmeticSharedTensor, func):
        setattr(ArithmeticSharedTensor, func, getattr(regular, func))
