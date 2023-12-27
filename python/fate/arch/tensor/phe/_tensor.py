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

import torch

from fate.arch.trace import auto_trace

_HANDLED_FUNCTIONS = {}
_PHE_TENSOR_ENCODED_HANDLED_FUNCTIONS = {}


class PHETensorEncoded:
    def __init__(self, coder, shape: torch.Size, data, dtype, device) -> None:
        self.coder = coder
        self.data = data
        self.shape = shape
        self.dtype = dtype
        self.device = device

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in _PHE_TENSOR_ENCODED_HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, PHETensorEncoded)) for t in types
        ):
            return NotImplemented
        return _PHE_TENSOR_ENCODED_HANDLED_FUNCTIONS[func](*args, **kwargs)


class PHETensor:
    def __init__(self, pk, evaluator, coder, shape: torch.Size, data, dtype, device) -> None:
        self._pk = pk
        self._evaluator = evaluator
        self._coder = coder
        self._data = data
        self._shape = shape
        self._dtype = dtype
        if isinstance(device, torch.device):
            from fate.arch import unify

            self._device = unify.device.from_torch_device(device)
        else:
            self._device = device

    def type(self, dtype):
        return self.with_template(self._data, dtype)

    def __repr__(self) -> str:
        return f"<PHETensor shape={self.shape}, dtype={self.dtype}, data={self._data}>"

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, item):
        from ._ops import slice_f

        if isinstance(item, int):
            return slice_f(self, item)
        else:
            raise NotImplementedError(f"item {item} not supported")

    def with_template(self, data, dtype=None, shape=None):
        if dtype is None:
            dtype = self._dtype
        if shape is None:
            shape = self._shape
        return PHETensor(self._pk, self._evaluator, self._coder, shape, data, dtype, self._device)

    @property
    def pk(self):
        return self._pk

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def coder(self):
        return self._coder

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def device(self):
        return self._device

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in _HANDLED_FUNCTIONS or not all(issubclass(t, (torch.Tensor, PHETensor)) for t in types):
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)

    """implement arth magic"""

    def __add__(self, other):
        from ._ops import add

        return add(self, other)

    def __radd__(self, other):
        from ._ops import add

        return add(other, self)

    def __sub__(self, other):
        from ._ops import sub

        return sub(self, other)

    def __rsub__(self, other):
        from ._ops import rsub

        return rsub(self, other)

    def __mul__(self, other):
        from ._ops import mul

        return mul(self, other)

    def __rmul__(self, other):
        from ._ops import mul

        return mul(other, self)

    def __matmul__(self, other):
        from ._ops import matmul

        return matmul(self, other)

    def __rmatmul__(self, other):
        from ._ops import rmatmul_f

        return rmatmul_f(self, other)


def implements(torch_function):
    """Register a torch function override for PHETensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        _HANDLED_FUNCTIONS[torch_function] = auto_trace(func)
        return func

    return decorator


def implements_encoded(torch_function):
    """Register a torch function override for PHEEncodedTensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        _PHE_TENSOR_ENCODED_HANDLED_FUNCTIONS[torch_function] = auto_trace(func)
        return func

    return decorator
