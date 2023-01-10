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
from fate.arch.tensor.types import LStorage, Shape, dtype
from fate.arch.unify import device


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

    def mean(self, *args, **kwargs):
        return _ops_cpu_plain_unary_buildin("mean", args, kwargs)(self)

    def sum(self, *args, **kwargs):
        return _ops_cpu_plain_unary_buildin("sum", args, kwargs)(self)

    def var(self, *args, **kwargs):
        return _ops_cpu_plain_unary_buildin("var", args, kwargs)(self)

    def std(self, *args, **kwargs):
        return _ops_cpu_plain_unary_buildin("std", args, kwargs)(self)

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


def _has_custom_unary(method):
    return method in {"slice", "max", "min"}


def _ops_cpu_plain_unary_custom(method, args, kwargs) -> Callable[[_TorchStorage], _TorchStorage]:
    if method == "slice":

        def _slice(storage: _TorchStorage):
            output = storage.data[args[0]]
            output_dtype = dtype.from_torch_dtype(output.dtype)
            output_shape = Shape(output.shape)
            return _TorchStorage(output_dtype, output_shape, output)

        return _slice

    if method == "max":

        def _max(storage: _TorchStorage):
            dim = None
            if len(args) > 0:
                dim = args[0]
            if "dim" in kwargs:
                dim = kwargs["dim"]
            if dim is None:
                output = torch.as_tensor(storage.data.max(*args, **kwargs))
            else:
                output = storage.data.max(*args, **kwargs).values
            output_dtype = dtype.from_torch_dtype(output.dtype)
            output_shape = Shape(output.shape)
            return _TorchStorage(output_dtype, output_shape, output)

        return _max

    if method == "min":

        def _min(storage: _TorchStorage):
            dim = None
            if len(args) > 0:
                dim = args[0]
            if "dim" in kwargs:
                dim = kwargs["dim"]
            if dim is None:
                output = torch.as_tensor(storage.data.min(*args, **kwargs))
            else:
                output = storage.data.min(*args, **kwargs).values
            output_dtype = dtype.from_torch_dtype(output.dtype)
            output_shape = Shape(output.shape)
            return _TorchStorage(output_dtype, output_shape, output)

        return _min

    raise NotImplementedError(f"method `{method}` not found in torch unary custom, consider to add custom extending")


def _ops_cpu_plain_binary_buildin(method, args, kwargs) -> Callable[[Any, Any], _TorchStorage]:
    if (
        func := {
            "add": torch.add,
            "sub": torch.sub,
            "mul": torch.mul,
            "div": torch.div,
            "pow": torch.pow,
            "remainder": torch.remainder,
            "matmul": torch.matmul,
            "true_divide": torch.true_divide,
            "truediv": torch.true_divide,
            "maximum": torch.maximum,
            "minimum": torch.minimum,
        }.get(method)
    ) is not None:

        def _wrap(a, b) -> _TorchStorage:
            output = func(_maybe_unwrap_storage(a), _maybe_unwrap_storage(b), *args, **kwargs)
            output_dtype = dtype.from_torch_dtype(output.dtype)
            output_shape = Shape(output.shape)
            return _TorchStorage(output_dtype, output_shape, output)

        return _wrap
    raise NotImplementedError(f"method `{method}` not found in torch binary buildin, consider to add custom extending")


def _has_custom_binary(method):
    return False


def _ops_cpu_plain_binary_custom(method, args, kwargs) -> Callable[[_TorchStorage], _TorchStorage]:
    raise NotImplementedError(f"method `{method}` not found in torch buildin, consider to add custom extending")


def _maybe_unwrap_storage(s):
    if isinstance(s, _TorchStorage):
        return s.data
    else:
        return s


# def _ops_dispatch_signature_3_local_cpu_plain(
#     method,
#     args,
#     kwargs,
# ) -> Callable[[_CPUStorage], _CPUStorage]:
#     def _wrap(storage: _CPUStorage) -> _CPUStorage:
#         func = getattr(torch, method)
#         output = func(storage.data, *args, **kwargs)
#         output_dtype = dtype.from_torch_dtype(output.dtype)
#         output_shape = Shape(output.shape)
#         return _CPUStorage(output_dtype, output_shape, output)

#     return _wrap
