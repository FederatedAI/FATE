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
from typing import Any, Callable, List

import torch
from fate.arch.tensor._exception import OpsDispatchUnsupportedError
from fate.arch.tensor.types import LStorage, Shape, dtype
from fate.arch.unify import device


class _RustPaillierStorage(LStorage):
    device = device.CPU

    def __init__(self, dtype: dtype, shape: Shape, data) -> None:
        self.dtype = dtype
        self.shape = shape
        self.data = data

    def tolist(self):
        return self.data.tolist()

    def to_local(self) -> "_RustPaillierStorage":
        return self

    def transpose(self):
        return _RustPaillierStorage(self.dtype, self.shape.transpose(), self.data.T)

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, _RustPaillierStorage)
            and (self.dtype == __o.dtype)
            and (isinstance(self.data, torch.Tensor))
            and (isinstance(__o.data, torch.Tensor))
            and torch.equal(self.data, __o.data)
        )

    def __str__(self) -> str:
        if isinstance(self.data, torch.Tensor):
            return f"_RustPaillierStorage({self.device}, {self.dtype}, {self.shape},\n<Inner data={self.data}, dtype={self.data.dtype}>)"
        return f"_RustPaillierStorage({self.device}, {self.dtype}, {self.shape},\n{self.data})"

    def __repr__(self) -> str:
        return self.__str__()

    def cat(self, others: List["_RustPaillierStorage"], axis):
        device = self.device
        d_type = self.dtype
        tensors = [self.data]
        for storage in others:
            if not isinstance(storage, _RustPaillierStorage) or storage.dtype != d_type or storage.device != device:
                raise RuntimeError(f"not supported type: {storage}")
        tensors.extend([storage.data for storage in others])
        cat_tensor = torch.cat(tensors, axis)
        return _RustPaillierStorage(d_type, Shape(cat_tensor.shape), cat_tensor)


def _ops_dispatch_signature_1_local_cpu_paillier(
    method, args, kwargs
) -> Callable[[_RustPaillierStorage], _RustPaillierStorage]:
    raise OpsDispatchUnsupportedError(method, False, device.CPU, dtype.paillier)


def _ops_dispatch_signature_2_local_cpu_paillier(
    method,
    args,
    kwargs,
) -> Callable[[Any, Any], _RustPaillierStorage]:

    # TODO: implement ops directly in C/Rust side
    def _wrap(a, b, **kwargs) -> _RustPaillierStorage:
        import operator

        a, b = _maybe_unwrap_storage(a), _maybe_unwrap_storage(b)
        func = getattr(operator, method)
        output = func(a, b)
        return _RustPaillierStorage(dtype.paillier, Shape(output.shape), output)

    return _wrap


def _ops_dispatch_signature_3_local_cpu_paillier(
    method,
    args,
    kwargs,
) -> Callable[[_RustPaillierStorage], _RustPaillierStorage]:
    raise OpsDispatchUnsupportedError(method, False, device.CPU, dtype.paillier)


def _maybe_unwrap_storage(s):
    from .plain import _TorchStorage

    if isinstance(s, (_RustPaillierStorage, _TorchStorage)):
        return s.data
    else:
        return s
