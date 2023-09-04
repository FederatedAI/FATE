#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
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

import functools
import typing
from typing import List, Optional, Tuple, TypeVar, cast

import torch
from fate.arch.abc import CTableABC
from fate.arch.context import Context

_HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """Register a torch function override for DStorage"""

    @functools.wraps(torch_function)
    def decorator(func):
        _HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class DTensor:
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in _HANDLED_FUNCTIONS or not all(issubclass(t, (torch.Tensor, DTensor)) for t in types):
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)

    @property
    def T(self):
        return torch.transpose(self, 0, 1)

    def elem_type(self) -> Optional[str]:
        return self.shardings._type

    def __init__(self, shardings: "Shardings") -> None:
        self.shardings = shardings

    def __add__(self, other):
        try:
            return torch.add(self, other)
        except Exception as e:
            raise RuntimeError(f"Failed to add {self} and {other}") from e

    def __radd__(self, other):
        return torch.add(other, self)

    def __sub__(self, other):
        return torch.sub(self, other)

    def __rsub__(self, other):
        return torch.rsub(self, other)

    def __mul__(self, other):
        return torch.mul(self, other)

    def __rmul__(self, other):
        return torch.mul(other, self)

    def __truediv__(self, other):
        return torch.div(self, other)

    def __rtruediv__(self, other):
        return torch.div(other, self)

    def __matmul__(self, other):
        return torch.matmul(self, other)

    def __rmatmul__(self, other):
        return torch.matmul(other, self)

    def encrypt(self, encryptor):
        return torch.encrypt_f(self, encryptor)

    def encrypt_encoded(self, encryptor):
        return torch.encrypt_encoded_f(self, encryptor)

    def decrypt_encoded(self, decryptor):
        return torch.decrypt_encoded_f(self, decryptor)

    def encode(self, encoder):
        return torch.encode_f(self, encoder)

    def decode(self, decoder):
        return torch.decode_f(self, decoder)

    def decrypt(self, decryptor):
        return torch.decrypt_f(self, decryptor)

    def exp(self):
        return torch.exp(self)

    def log(self):
        return torch.log(self)

    def square(self):
        return torch.square(self)

    def sigmoid(self):
        return torch.sigmoid(self)

    @property
    def shape(self):
        return self.shardings.shape

    @property
    def dtype(self):
        return self.shardings.dtype

    @property
    def device(self):
        return self.shardings.device

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, DTensor) and self.shardings == __o.shardings

    def __str__(self) -> str:
        return f"<DTensor(shardings={self.shardings})>"

    @classmethod
    def from_sharding_table(
        cls,
        data: CTableABC,
        shapes: Optional[List[torch.Size]],
        axis=0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        return DTensor(Shardings(data, shapes, axis, dtype, device))

    @classmethod
    def from_sharding_list(cls, ctx: Context, data: List[torch.Tensor], num_partitions=16, axis=0):
        dtype = data[0].dtype
        device = data[0].device
        shapes = []
        for t in data:
            shapes.append(t.shape)
            assert dtype == t.dtype
            assert device == t.device

        for shape in shapes[1:]:
            for i, (s1, s2) in enumerate(zip(shapes[0], shape)):
                if i == axis:
                    continue
                else:
                    assert s1 == s2
        return cls.from_sharding_table(
            ctx.computing.parallelize(data, partition=num_partitions, include_key=False), shapes, axis, dtype, device
        )


T1 = TypeVar("T1")
T2 = TypeVar("T2")


class Shardings:
    def __init__(
        self,
        data: CTableABC[int, torch.Tensor],
        shapes: Optional[List[torch.Size]] = None,
        axis: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        type: Optional[str] = None,
    ):
        self._data = data
        self._type = type

        if shapes is None:
            shards_shape = sorted(self._data.map(lambda k, s: (k, s.shape)).collect())
            _shapes = []
            for i, (k, s) in enumerate(shards_shape):
                assert i == k
                _shapes.append(s)
        else:
            _shapes = shapes
        self._shapes = _ShardingShapes(_shapes, axis)

        if dtype is None or device is None:
            first_shard = self._data.first()[1]
            shard_dtype = cast(torch.dtype, first_shard.dtype)
            shard_device = cast(torch.device, first_shard.device)
            if dtype is not None:
                assert dtype == shard_dtype
            if device is not None:
                assert device == shard_device
            self._dtype = shard_dtype
            self._device = shard_device
        else:
            self._dtype = dtype
            self._device = device

    @property
    def shapes(self):
        return self._shapes

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self.shapes.merge_shapes()

    def with_dtype(self, dtype: torch.dtype):
        self._dtype = dtype
        return self

    @property
    def device(self):
        return self._device

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, Shardings)
            and self.device == __o.device
            and self.dtype == __o.dtype
            and self.shapes == __o.shapes
            and all(self._data.join(__o._data, lambda s1, s2: torch.allclose(s1, s2)).collect())
        )

    def __str__(self) -> str:
        return f"Sharding<shapes={self._shapes}, dtype={self._dtype}, device={self._device}>"

    def merge(self):
        shardings = [pair[1] for pair in sorted(self._data.collect())]
        return torch.cat(shardings, self.shapes.axis)

    def map_shard(
        self,
        func: typing.Callable[[torch.Tensor], torch.Tensor],
        shapes: Optional[List[torch.Size]] = None,
        axis: Optional[int] = None,
        dtype_promote_to: Optional[torch.dtype] = None,
        type: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if dtype is None:
            if dtype_promote_to is not None:
                dtype = torch.promote_types(self.dtype, dtype_promote_to)
            else:
                dtype = self._dtype
        if shapes is None:
            shapes = self.shapes.shapes
        if axis is None:
            axis = self.shapes.axis
        if type is None:
            type = self._type
        return Shardings(self._data.mapValues(func), shapes, axis, dtype, self._device, type)

    def map_reduce_shard(
        self,
        mapper_func: typing.Callable[[torch.Tensor], T1],
        reducer_func: typing.Callable[[T1, T1], T1],
    ) -> T1:
        """
        expect local output
        """
        return self._data.mapValues(mapper_func).reduce(reducer_func)

    def map_reduce_shard_with_stride(
        self,
        stride_mapper_func: typing.Callable[[int, int, torch.Tensor], T1],
        reducer_func: typing.Callable[[T1, T1], T1],
    ) -> T1:
        """
        map with stride
        """
        strides = self.shapes.strides()
        axis = self.shapes.axis

        def _stride_mapper(func: typing.Callable[[int, int, torch.Tensor], T1]):
            def _wrap(i: int, t: torch.Tensor) -> Tuple[int, T1]:
                stride = strides[i]
                size = t.shape[axis]
                return (i, func(stride, size, t))

            return _wrap

        return self._data.map(_stride_mapper(stride_mapper_func)).reduce(reducer_func)

    def join_shard(
        self,
        other: "Shardings",
        func: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        out_dtype: typing.Optional[torch.dtype] = None,
        out_shapes: typing.Optional[List[torch.Size]] = None,
        out_axis: typing.Optional[int] = None,
        dtype_promote_to: Optional[torch.dtype] = None,
    ):
        if out_dtype is None:
            out_dtype = torch.promote_types(self._dtype, other._dtype)
        if dtype_promote_to is not None:
            out_dtype = torch.promote_types(out_dtype, dtype_promote_to)
        if out_shapes is None or out_axis is None:
            shapes = self.shapes.bc_shapes(other.shapes)
            out_shapes = shapes.shapes
            out_axis = shapes.axis
        return Shardings(self._data.join(other._data, func), out_shapes, out_axis, out_dtype, self._device)

    def join_reduce_shard(
        self,
        other: "Shardings",
        mapper_func: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        reduce_func: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        return self._data.join(other._data, mapper_func).reduce(reduce_func)


class _ShardingShapes:
    def __init__(self, shapes: List[torch.Size], axis: int) -> None:
        self.shapes = shapes
        self.axis = axis

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, _ShardingShapes) and self.axis == __o.axis and len(self.shapes) == len(__o.shapes):
            for s1, s2 in zip(self.shapes, __o.shapes):
                if s1 != s2:
                    return False
        return True

    def __str__(self) -> str:
        return f"<ShardingShape(shapes={self.shapes}, axis={self.axis})>"

    def __repr__(self):
        return self.__str__()

    def bc_shapes(self, other: "_ShardingShapes") -> "_ShardingShapes":
        if isinstance(other, _ShardingShapes):
            assert len(self.shapes) == len(other.shapes), f"sharding num mismatch: {self.shapes} vs {other.shapes}"
            _bc_shapes = []
            for s1, s2 in zip(self.shapes, other.shapes):
                _bc_shapes.append(torch.broadcast_shapes(s1, s2))

            self_axis = len(_bc_shapes[0]) - len(self.shapes[0]) + self.axis
            other_axis = len(_bc_shapes[0]) - len(other.shapes[0]) + other.axis
            assert self_axis == other_axis, f"sharding axis mismatch: {self_axis} vs {other_axis}"
            return _ShardingShapes(_bc_shapes, self_axis)
        elif isinstance(other, torch.Size):
            _bc_shapes = []
            for s in self.shapes:
                _bc_shapes.append(torch.broadcast_shapes(s, other))
                # assert other shape in distributed axis is 1
                other_align_axis = len(other) - len(s) + self.axis
                if other_align_axis >= 0:
                    assert other[other_align_axis] == 1, f"shape in distributed axis should be 1: {self} vs {other}"
            self_axis = len(_bc_shapes[0]) - len(self.shapes[0]) + self.axis

            return _ShardingShapes(_bc_shapes, self_axis)
        else:
            raise NotImplementedError(f"type `{other}`")

    def merge_shapes(self):
        _shape = list(self.shapes[0])
        for s in self.shapes[1:]:
            for i in range(len(_shape)):
                if i == self.axis:
                    _shape[i] += s[i]
                else:
                    assert _shape[i] == s[i]
        return torch.Size(_shape)

    def strides(self):
        _stride = [0]
        agg = 0
        for s in self.shapes[:-1]:
            agg += s[self.axis]
            _stride.append(agg)
        return _stride

    def squeeze(self, dims: Tuple[int], keepdim=False):
        _shapes = []
        for s in self.shapes:
            _s = []
            for i in range(len(s)):
                if i in dims:
                    if keepdim:
                        _s.append(1)
                else:
                    _s.append(s[i])
            _shapes.append(torch.Size(_s))
        return _shapes
