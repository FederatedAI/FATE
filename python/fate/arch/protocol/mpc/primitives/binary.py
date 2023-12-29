#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# dependencies:
import torch

import fate.arch.protocol.mpc.communicator as comm
from fate.arch.protocol.mpc.common.rng import generate_kbit_random_tensor
from fate.arch.protocol.mpc.common.tensor_types import is_tensor
from fate.arch.protocol.mpc.common.util import torch_cat, torch_stack
from fate.arch.protocol.mpc.cuda import CUDALongTensor
from fate.arch.protocol.mpc.encoder import FixedPointEncoder
from fate.arch.protocol.mpc.functions import regular
from . import beaver, circuit

SENTINEL = -1


# MPC tensor where shares are XOR-sharings.
class BinarySharedTensor(object):
    """
    Encrypted tensor object that uses binary sharing to perform computations.

    Binary shares are computed by splitting each value of the input tensor
    into n separate random values that xor together to the input tensor value,
    where n is the number of parties present in the protocol (world_size).
    """

    def __init__(self, tensor=None, size=None, broadcast_size=False, src=0, device=None):
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

        # assume zero bits of precision unless encoder is set outside of init:
        self.encoder = FixedPointEncoder(precision_bits=0)
        if tensor is not None:
            tensor = self.encoder.encode(tensor)
            tensor = tensor.to(device=device)
            size = tensor.size()

        # if other parties do not know tensor's size, broadcast the size:
        if broadcast_size:
            size = comm.get().broadcast_obj(size, src)

        # generate pseudo-random zero sharing (PRZS) and add source's tensor:
        self.share = BinarySharedTensor.PRZS(size, device=device).share
        if self.rank == src:
            self.share ^= tensor

    @staticmethod
    def new(*args, **kwargs):
        """
        Creates a new BinarySharedTensor, passing all args and kwargs into the constructor.
        """
        return BinarySharedTensor(*args, **kwargs)

    @staticmethod
    def from_shares(share, precision=None, src=0, device=None):
        """Generate a BinarySharedTensor from a share from each party"""
        result = BinarySharedTensor(src=SENTINEL)
        share = share.to(device) if device is not None else share
        result.share = CUDALongTensor(share) if share.is_cuda else share
        result.encoder = FixedPointEncoder(precision_bits=precision)
        return result

    @staticmethod
    def PRZS(*size, device=None):
        """
        Generate a Pseudo-random Sharing of Zero (using arithmetic shares)

        This function does so by generating `n` numbers across `n` parties with
        each number being held by exactly 2 parties. Therefore, each party holds
        two numbers. A zero sharing is found by having each party xor their two
        numbers together.
        """
        from fate.arch.protocol.mpc import generators

        tensor = BinarySharedTensor(src=SENTINEL)
        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        g0 = generators["prev"][device]
        g1 = generators["next"][device]
        current_share = generate_kbit_random_tensor(*size, device=device, generator=g0)
        next_share = generate_kbit_random_tensor(*size, device=device, generator=g1)
        tensor.share = current_share ^ next_share
        return tensor

    @staticmethod
    def rand(*size, bits=64, device=None):
        """
        Generate a uniform random samples with a given size.
        """
        tensor = BinarySharedTensor(src=SENTINEL)
        if isinstance(size[0], (torch.Size, tuple)):
            size = size[0]
        tensor.share = generate_kbit_random_tensor(size, bitlength=bits, device=device)
        return tensor

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
    def rank(self):
        return comm.get().get_rank()

    @property
    def share(self):
        """Returns underlying _tensor"""
        return self._tensor

    @share.setter
    def share(self, value):
        """Sets _tensor to value"""
        self._tensor = value

    def shallow_copy(self):
        """Create a shallow copy"""
        result = BinarySharedTensor(src=SENTINEL)
        result.encoder = self.encoder
        result._tensor = self._tensor
        return result

    def clone(self):
        result = BinarySharedTensor(src=SENTINEL)
        result.encoder = self.encoder
        result._tensor = self._tensor.clone()
        return result

    def copy_(self, other):
        """Copies other tensor into this tensor."""
        self.share.copy_(other.share)
        self.encoder = other.encoder

    def __repr__(self):
        return f"BinarySharedTensor({self.share})"

    def __bool__(self):
        """Override bool operator since encrypted tensors cannot evaluate"""
        raise RuntimeError("Cannot evaluate BinarySharedTensors to boolean values")

    def __nonzero__(self):
        """__bool__ for backwards compatibility with Python 2"""
        raise RuntimeError("Cannot evaluate BinarySharedTensors to boolean values")

    def __ixor__(self, y):
        """Bitwise XOR operator (element-wise) in place"""
        if is_tensor(y) or isinstance(y, int):
            if self.rank == 0:
                self.share ^= y
        elif isinstance(y, BinarySharedTensor):
            self.share ^= y.share
        else:
            raise TypeError("Cannot XOR %s with %s." % (type(y), type(self)))
        return self

    def __xor__(self, y):
        """Bitwise XOR operator (element-wise)"""
        result = self.clone()
        if isinstance(y, BinarySharedTensor):
            broadcast_tensors = torch.broadcast_tensors(result.share, y.share)
            result.share = broadcast_tensors[0].clone()
        elif is_tensor(y):
            broadcast_tensors = torch.broadcast_tensors(result.share, y)
            result.share = broadcast_tensors[0].clone()
        return result.__ixor__(y)

    def __iand__(self, y):
        """Bitwise AND operator (element-wise) in place"""
        if is_tensor(y) or isinstance(y, int):
            self.share &= y
        elif isinstance(y, BinarySharedTensor):
            self.share.set_(beaver.AND(self, y).share.data)
        else:
            raise TypeError("Cannot AND %s with %s." % (type(y), type(self)))
        return self

    def __and__(self, y):
        """Bitwise AND operator (element-wise)"""
        result = self.clone()
        # TODO: Remove explicit broadcasts to allow smaller beaver triples
        if isinstance(y, BinarySharedTensor):
            broadcast_tensors = torch.broadcast_tensors(result.share, y.share)
            result.share = broadcast_tensors[0].clone()
        elif is_tensor(y):
            broadcast_tensors = torch.broadcast_tensors(result.share, y)
            result.share = broadcast_tensors[0].clone()
        return result.__iand__(y)

    def __ior__(self, y):
        """Bitwise OR operator (element-wise) in place"""
        xor_result = self ^ y
        return self.__iand__(y).__ixor__(xor_result)

    def __or__(self, y):
        """Bitwise OR operator (element-wise)"""
        return self.__and__(y) ^ self ^ y

    def __invert__(self):
        """Bitwise NOT operator (element-wise)"""
        result = self.clone()
        if result.rank == 0:
            result.share ^= -1
        return result

    def lshift_(self, value):
        """Left shift elements by `value` bits"""
        assert isinstance(value, int), "lshift must take an integer argument."
        self.share <<= value
        return self

    def lshift(self, value):
        """Left shift elements by `value` bits"""
        return self.clone().lshift_(value)

    def rshift_(self, value):
        """Right shift elements by `value` bits"""
        assert isinstance(value, int), "rshift must take an integer argument."
        self.share >>= value
        return self

    def rshift(self, value):
        """Right shift elements by `value` bits"""
        return self.clone().rshift_(value)

    # Circuits
    def add(self, y):
        """Compute [self] + [y] for xor-sharing"""
        return circuit.add(self, y)

    def eq(self, y):
        return circuit.eq(self, y)

    def ne(self, y):
        return self.eq(y) ^ 1

    def lt(self, y):
        return circuit.lt(self, y)

    def le(self, y):
        return circuit.le(self, y)

    def gt(self, y):
        return circuit.gt(self, y)

    def ge(self, y):
        return circuit.ge(self, y)

    def __setitem__(self, index, value):
        """Set tensor values by index"""
        if is_tensor(value) or isinstance(value, list):
            value = BinarySharedTensor(value)
        assert isinstance(value, BinarySharedTensor), "Unsupported input type %s for __setitem__" % type(value)
        self.share.__setitem__(index, value.share)

    @staticmethod
    def stack(seq, *args, **kwargs):
        """Stacks a list of tensors along a given dimension"""
        assert isinstance(seq, list), "Stack input must be a list"
        assert isinstance(seq[0], BinarySharedTensor), "Sequence must contain BinarySharedTensors"
        result = seq[0].shallow_copy()
        result.share = torch_stack([BinarySharedTensor.share for BinarySharedTensor in seq], *args, **kwargs)
        return result

    def sum(self, dim=None):
        """Add all tensors along a given dimension using a log-reduction"""
        if dim is None:
            x = self.flatten()
        else:
            x = self.transpose(0, dim)

        # Add all BinarySharedTensors
        while x.size(0) > 1:
            extra = None
            if x.size(0) % 2 == 1:
                extra = x[0]
                x = x[1:]
            x0 = x[: (x.size(0) // 2)]
            x1 = x[(x.size(0) // 2) :]
            x = x0 + x1
            if extra is not None:
                x.share = torch_cat([x.share, extra.share.unsqueeze(0)])

        if dim is None:
            x = x.squeeze()
        else:
            x = x.transpose(0, dim).squeeze(dim)
        return x

    def cumsum(self, *args, **kwargs):
        raise NotImplementedError("BinarySharedTensor cumsum not implemented")

    def trace(self, *args, **kwargs):
        raise NotImplementedError("BinarySharedTensor trace not implemented")

    @staticmethod
    def reveal_batch(tensor_or_list, dst=None):
        """Get (batched) plaintext without any downscaling"""
        if isinstance(tensor_or_list, BinarySharedTensor):
            return tensor_or_list.reveal(dst=dst)

        assert isinstance(tensor_or_list, list), f"Invalid input type into reveal {type(tensor_or_list)}"
        shares = [tensor.share for tensor in tensor_or_list]
        op = torch.distributed.ReduceOp.BXOR
        if dst is None:
            return comm.get().all_reduce(shares, op=op, batched=True)
        else:
            return comm.get().reduce(shares, dst, op=op, batched=True)

    def reveal(self, dst=None):
        """Get plaintext without any downscaling"""
        op = torch.distributed.ReduceOp.BXOR
        if dst is None:
            return comm.get().all_reduce(self.share, op=op)
        else:
            return comm.get().reduce(self.share, dst, op=op)

    def get_plain_text(self, dst=None):
        """Decrypts the tensor."""
        # Edge case where share becomes 0 sized (e.g. result of split)
        if self.nelement() < 1:
            return torch.empty(self.share.size())
        return self.encoder.decode(self.reveal(dst=dst))

    def where(self, condition, y):
        """Selects elements from self or y based on condition

        Args:
            condition (torch.bool or BinarySharedTensor): when True yield self,
                otherwise yield y. Note condition is not bitwise.
            y (torch.tensor or BinarySharedTensor): selected when condition is
                False.

        Returns: BinarySharedTensor or torch.tensor.
        """
        if is_tensor(condition):
            condition = condition.long()
            is_binary = ((condition == 1) | (condition == 0)).all()
            assert is_binary, "condition values must be 0 or 1"
            # -1 mult expands 0 into binary 00...00 and 1 into 11...11
            condition_expanded = -condition
            y_masked = y & (~condition_expanded)
        elif isinstance(condition, BinarySharedTensor):
            condition_expanded = condition.clone()
            # -1 mult expands binary while & 1 isolates first bit
            condition_expanded.share = -(condition_expanded.share & 1)
            # encrypted tensor must be first operand
            y_masked = (~condition_expanded) & y
        else:
            msg = f"condition {condition} must be torch.bool, or BinarySharedTensor"
            raise ValueError(msg)

        return (self & condition_expanded) ^ y_masked

    def scatter_(self, dim, index, src):
        """Writes all values from the tensor `src` into `self` at the indices
        specified in the `index` tensor. For each value in `src`, its output index
        is specified by its index in `src` for `dimension != dim` and by the
        corresponding value in `index` for `dimension = dim`.
        """
        if is_tensor(src):
            src = BinarySharedTensor(src)
        assert isinstance(src, BinarySharedTensor), "Unrecognized scatter src type: %s" % type(src)
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

    # Bitwise operators
    __add__ = add
    __eq__ = eq
    __ne__ = ne
    __lt__ = lt
    __le__ = le
    __gt__ = gt
    __ge__ = ge
    __lshift__ = lshift
    __rshift__ = rshift

    # In-place bitwise operators
    __ilshift__ = lshift_
    __irshift__ = rshift_

    # Reversed boolean operations
    __radd__ = __add__
    __rxor__ = __xor__
    __rand__ = __and__
    __ror__ = __or__


# Register regular functions
skip_funcs = ["trace", "sum", "cumsum", "pad"]  # skip additive functions and pad
for func in regular.__all__:
    if func in skip_funcs:
        continue
    setattr(BinarySharedTensor, func, getattr(regular, func))
