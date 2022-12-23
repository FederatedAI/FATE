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
        output = torch.mean(self.data, *args, **kwargs)
        output_dtype = dtype.from_torch_dtype(output.dtype)
        output_shape = Shape(output.shape)
        return _TorchStorage(output_dtype, output_shape, output)

    def sum(self, *args, **kwargs):
        output = torch.sum(self.data, *args, **kwargs)
        output_dtype = dtype.from_torch_dtype(output.dtype)
        output_shape = Shape(output.shape)
        return _TorchStorage(output_dtype, output_shape, output)


def _ops_cpu_plain_unary_buildin(method, args, kwargs) -> Callable[[_TorchStorage], _TorchStorage]:
    if method in {"exp", "log", "neg", "reciprocal", "square", "abs", "sum", "sqrt", "var", "std"}:
        func = getattr(torch, method)

        def _wrap(storage: _TorchStorage) -> _TorchStorage:
            output = func(storage.data, *args, **kwargs)
            output_dtype = dtype.from_torch_dtype(output.dtype)
            output_shape = Shape(output.shape)
            return _TorchStorage(output_dtype, output_shape, output)

        return _wrap
    raise NotImplementedError(f"method `{method}` not found in torch unary buildin, consider to add custom extending")


def _has_custom_unary(method):
    return method in {"slice"}


def _ops_cpu_plain_unary_custom(method, args, kwargs) -> Callable[[_TorchStorage], _TorchStorage]:
    if method == "slice":
        args[0]

        def _slice(storage: _TorchStorage):
            output = storage.data[args[0]]
            output_dtype = dtype.from_torch_dtype(output.dtype)
            output_shape = Shape(output.shape)
            return _TorchStorage(output_dtype, output_shape, output)

        return _slice

    raise NotImplementedError(f"method `{method}` not found in torch unary custom, consider to add custom extending")


def _ops_cpu_plain_binary_buildin(method, args, kwargs) -> Callable[[Any, Any], _TorchStorage]:
    if method in {"add", "sub", "mul", "div", "pow", "remainder", "matmul", "true_divide"}:
        func = getattr(torch, method)

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
