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
from typing import Callable, List

import torch
from fate.arch.unify import device

from ..._dtype import dtype
from ..._protocol import LStorage
from ..._shape import Shape


class _TorchStorage(LStorage):
    device = device.CPU

    def __init__(self, dtype: dtype, shape: Shape, data) -> None:
        self.dtype = dtype
        self.shape = shape
        self.data = data

    def tolist(self):
        return self.data.tolist()

    def to_local(self) -> "_TorchStorage":
        return self

    def transpose(self):
        return _TorchStorage(self.dtype, self.shape.transpose(), self.data.T)

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, _TorchStorage)
            and (self.dtype == __o.dtype)
            and (isinstance(self.data, torch.Tensor))
            and (isinstance(__o.data, torch.Tensor))
            and torch.equal(self.data, __o.data)
        )

    def __str__(self) -> str:
        if isinstance(self.data, torch.Tensor):
            return f"_CPUStorage({self.device}, {self.dtype}, {self.shape},\n<Inner data={self.data}, dtype={self.data.dtype}>)"
        return f"_CPUStorage({self.device}, {self.dtype}, {self.shape},\n{self.data})"

    def __repr__(self) -> str:
        return self.__str__()

    def cat(self, others: List["_TorchStorage"], axis):
        device = self.device
        d_type = self.dtype
        tensors = [self.data]
        for storage in others:
            if not isinstance(storage, _TorchStorage) or storage.dtype != d_type or storage.device != device:
                raise RuntimeError(f"not supported type: {storage}")
        tensors.extend([storage.data for storage in others])
        cat_tensor = torch.cat(tensors, axis)
        return _TorchStorage(d_type, Shape(cat_tensor.shape), cat_tensor)

    ### ops dispatch, use staticmethod here
    @staticmethod
    def unary(method, args, kwargs):
        if _has_custom_unary(method):
            return _ops_cpu_plain_unary_custom(method, args, kwargs)
        else:
            return _ops_cpu_plain_unary_buildin(method, args, kwargs)

    @staticmethod
    def binary(method, args, kwargs):
        if _has_custom_binary(method):
            return _ops_cpu_plain_binary_custom(method, args, kwargs)
        else:
            return _ops_cpu_plain_binary_buildin(method, args, kwargs)

    def max(self, *args, **kwargs):
        return _ops_cpu_plain_unary_custom("max", args, kwargs)(self)

    def min(self, *args, **kwargs):
        return _ops_cpu_plain_unary_custom("min", args, kwargs)(self)


def _ops_cpu_plain_unary_buildin(method, args, kwargs) -> Callable[[_TorchStorage], _TorchStorage]:
    if (
        func := {
            "exp": torch.exp,
            "log": torch.log,
            "neg": torch.neg,
            "reciprocal": torch.reciprocal,
            "square": torch.square,
            "abs": torch.abs,
            "sum": torch.sum,
            "sqrt": torch.sqrt,
            "var": torch.var,
            "std": torch.std,
            "mean": torch.mean,
        }.get(method)
    ) is not None:

        def _wrap(storage: _TorchStorage) -> _TorchStorage:
            output = func(storage.data, *args, **kwargs)
            output_dtype = dtype.from_torch_dtype(output.dtype)
            output_shape = Shape(output.shape)
            return _TorchStorage(output_dtype, output_shape, output)

        return _wrap
    raise NotImplementedError(f"method `{method}` not found in torch unary buildin, consider to add custom extending")
