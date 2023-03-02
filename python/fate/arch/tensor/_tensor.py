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

from ..storage._protocol import LStorage
from .distributed import DStorage


class Tensor:
    def __init__(self, storage: Union[DStorage, LStorage]) -> None:
        self._storage = storage

    @property
    def is_distributed(self):
        from .distributed import DStorage

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
        from .distributed import DStorage

        if isinstance(self._storage, DStorage):
            return Tensor(self._storage.to_local())
        return self

    def tolist(self):
        from .distributed import DStorage

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
        from fate.arch.tensor._ops import add

        return add(self, other)

    def __radd__(self, other):
        from fate.arch.tensor._ops import add

        return add(other, self)

    def __sub__(self, other):
        from fate.arch.tensor._ops import sub

        return sub(self, other)

    def __rsub__(self, other):
        from fate.arch.tensor._ops import sub

        return sub(other, self)

    def __mul__(self, other):
        from fate.arch.tensor._ops import mul

        return mul(self, other)

    def __rmul__(self, other):
        from fate.arch.tensor._ops import mul

        return mul(other, self)

    def __div__(self, other):
        from fate.arch.tensor._ops import div

        return div(self, other)

    def __truediv__(self, other):
        from fate.arch.tensor._ops import truediv

        return truediv(self, other)

    def __matmul__(self, other):
        from fate.arch.tensor._ops import matmul

        return matmul(self, other)

    def __getitem__(self, key):
        from fate.arch.tensor._ops import slice

        return slice(self, key)

    def mean(self, *args, **kwargs) -> "Tensor":
        from fate.arch.tensor._ops import mean

        return mean(*args, **kwargs)

    def sum(self, *args, **kwargs) -> "Tensor":
        from fate.arch.tensor._ops import sum

        return sum(*args, **kwargs)

    def std(self, *args, **kwargs) -> "Tensor":
        from fate.arch.tensor._ops import std

        return std(*args, **kwargs)

    def max(self, *args, **kwargs) -> "Tensor":
        from fate.arch.tensor._ops import max

        return max(*args, **kwargs)

    def min(self, *args, **kwargs) -> "Tensor":
        from fate.arch.tensor._ops import min

        return min(*args, **kwargs)
