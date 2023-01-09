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
from typing import List, Union

import torch
from fate.interface import Context

from .types import DStorage, LStorage, Shape, dtype


def tensor(t: torch.Tensor):
    from .storage.local.device.cpu.plain import _TorchStorage

    storage = _TorchStorage(dtype.from_torch_dtype(t.dtype), Shape(t.shape), t)
    return Tensor(storage)


def randn(shape, dtype):
    torch_tensor = torch.randn(shape, dtype=dtype.to_torch_dtype())
    return tensor(torch_tensor)


def distributed_tensor(ctx: Context, tensors: List[torch.Tensor], d_axis=0, partitions=3):
    from .storage.local.device.cpu.plain import _TorchStorage

    storages = [_TorchStorage(dtype.from_torch_dtype(t.dtype), Shape(t.shape), t) for t in tensors]
    storage = DStorage.from_storages(ctx, storages, d_axis, partitions)
    return Tensor(storage)


def _inject_op_sinature1(func):
    method = func.__name__

    def _wrap(input):
        from .ops._ops import dispatch_signature1

        return dispatch_signature1(method, input, [], {})

    return _wrap


def _inject_op_sinature2(func):
    method = func.__name__

    def _wrap(input, other):
        from .ops._ops import dispatch_signature2

        return dispatch_signature2(method, input, other, [], {})

    return _wrap


class Tensor:
    def __init__(self, storage: Union[DStorage, LStorage]) -> None:
        self._storage = storage

    @property
    def is_distributed(self):
        from .types import DStorage

        return isinstance(self._storage, DStorage)

    def to(self, party, name: str):
        return party.put(name, self)

    @property
    def dtype(self):
        return self._storage.dtype

    @property
    def storage(self):
        return self._storage

    @property
    def device(self):
        return self._storage.device

    @property
    def T(self):
        return Tensor(self._storage.transpose())

    @property
    def shape(self):
        return self._storage.shape

    def to_local(self):
        if isinstance(self._storage, DStorage):
            return Tensor(self._storage.to_local())
        return self

    def tolist(self):
        if isinstance(self._storage, DStorage):
            return self._storage.to_local().tolist()
        return self._storage.tolist()

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Tensor) and self._storage == __o._storage

    def __str__(self) -> str:
        return f"Tensor(storage={self.storage})"

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return self.rsub(other)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __div__(self, other):
        return self.div(other)

    def __truediv__(self, other):
        return self.truediv(other)

    def __matmul__(self, other):
        from .ops import matmul

        return matmul(self, other)

    def __getitem__(self, key):
        from .ops import slice

        return slice(self, key)

    """and others"""

    @_inject_op_sinature2
    def add(self, other) -> "Tensor":
        ...

    @_inject_op_sinature2
    def sub(self, other) -> "Tensor":
        ...

    @_inject_op_sinature2
    def rsub(self, other) -> "Tensor":
        ...

    @_inject_op_sinature2
    def mul(self, other) -> "Tensor":
        ...

    @_inject_op_sinature2
    def div(self, other) -> "Tensor":
        ...

    @_inject_op_sinature2
    def truediv(self, other) -> "Tensor":
        ...

    def mean(self, *args, **kwargs) -> "Tensor":
        return Tensor(self.storage.mean(*args, **kwargs))

    def std(self, *args, **kwargs) -> "Tensor":
        return Tensor(self.storage.std(*args, **kwargs))

    def max(self, *args, **kwargs) -> "Tensor":
        return Tensor(self.storage.max(*args, **kwargs))

    def min(self, *args, **kwargs) -> "Tensor":
        return Tensor(self.storage.min(*args, **kwargs))
