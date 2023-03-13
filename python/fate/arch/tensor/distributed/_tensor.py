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
from typing import List, Optional, cast

import torch
from fate.arch.computing import CTableABC
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

    def __init__(self, shardings: "Shardings") -> None:
        self.shardings = shardings

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

    def to_local(self) -> torch.Tensor:
        storages = [pair[1] for pair in sorted(self.blocks.collect())]
        return torch.cat(storages, self._d_axis.axis)

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
        shapes = [t.shape for t in data]
        # TODO: validate according to axis
        dtype = data[0].dtype
        device = data[0].device
        return cls.from_sharding_table(
            ctx.computing.parallelize(data, partition=num_partitions, include_key=False), shapes, axis, dtype, device
        )

    @classmethod
    def from_storages(cls, ctx, storages: List[torch.Tensor], d_axis=0, partitions=4):
        d_type = storages[0].dtype
        device = storages[0].device
        shape_size = storages[0].shape
        for storage in storages[1:]:
            if storage.dtype != d_type:
                raise RuntimeError(f"requires same dtype")
            if storage.device != device:
                raise RuntimeError(f"requires same device")
        blocks = ctx.computing.parallelize(enumerate(storages), partition=partitions, include_key=True)
        return DTensor(blocks, shape_size, DAxis(d_axis, partitions), d_type, device)


class Shardings:
    def __init__(
        self,
        data: CTableABC,
        shapes: Optional[List[torch.Size]] = None,
        axis: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        self._data = data
        self._axis = axis

        if shapes is None:
            shards_shape = sorted(self._data.map(lambda k, s: (k, s.shape)).collect())
            self._shapes = []
            for i, (k, s) in enumerate(shards_shape):
                assert i == k
                self._shapes.append(s)
        else:
            self._shapes = shapes

        if dtype is None or device is None:
            first_shard = self._data.first()
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
    def shape(self):
        return self._shapes[0]

    @property
    def dtype(self):
        return self._dtype

    def with_dtype(self, dtype: torch.dtype):
        self._dtype = dtype
        return self

    @property
    def device(self):
        return self._device

    def __eq__(self, __o: object) -> bool:
        if (
            isinstance(__o, Shardings)
            and self.device == __o.device
            and self.dtype == __o.dtype
            and len(self._shapes) == len(__o._shapes)
        ):
            for s1, s2 in zip(self._shapes, __o._shapes):
                if s1 != s2:
                    return False
            return all(self._data.join(__o._data, lambda s1, s2: torch.allclose(s1, s2)).collect())
        return False

    def __str__(self) -> str:
        return f"Sharding<shapes={self._shapes}, dtype={self._dtype}, device={self._device}>"

    def map_shard(
        self, func: typing.Callable[[torch.Tensor], torch.Tensor], dtype_promote_to: Optional[torch.dtype] = None
    ):
        if dtype_promote_to is not None:
            dtype = torch.promote_types(self.dtype, dtype_promote_to)
        else:
            dtype = self._dtype
        return Shardings(self._data.mapValues(func), self._shapes, self._axis, dtype, self._device)

    def join_shard(
        self,
        other: "Shardings",
        func: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        out_dtype: typing.Optional[torch.dtype] = None,
        dtype_promote_to: Optional[torch.dtype] = None,
    ):
        if out_dtype is None:
            out_dtype = torch.promote_types(self._dtype, other._dtype)
        if dtype_promote_to is not None:
            out_dtype = torch.promote_types(out_dtype, dtype_promote_to)
        return Shardings(self._data.join(other._data, func), self._shapes, self._axis, out_dtype, self._device)
