import operator
from typing import Any, Callable, List

import torch

from ..._base import Shape, StorageBase, device, dtype
from ..._exception import OpsDispatchUnsupportedError


class _CPUStorage(StorageBase):
    device = device.CPU

    def __init__(self, dtype: dtype, shape: Shape, data) -> None:
        self.dtype = dtype
        self.shape = shape
        self.data = data

    def transpose(self):
        return _CPUStorage(self.dtype, self.shape.transpose(), self.data.T)

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, _CPUStorage)
            and (self.dtype == __o.dtype)
            and (isinstance(self.data, torch.Tensor))
            and (isinstance(__o.data, torch.Tensor))
            and torch.equal(self.data, __o.data)
        )

    def __str__(self) -> str:
        return f"_CPUStorage({self.device}, {self.dtype}, {self.shape},\n{self.data})"

    def __repr__(self) -> str:
        return self.__str__()

    def cat(self, others: List["_CPUStorage"], axis):
        device = self.device
        d_type = self.dtype
        tensors = [self.data]
        for storage in others:
            if (
                not isinstance(storage, _CPUStorage)
                or storage.dtype != d_type
                or storage.device != device
            ):
                raise RuntimeError(f"not supported type: {storage}")
        tensors.extend([storage.data for storage in others])
        cat_tensor = torch.cat(tensors, axis)
        return _CPUStorage(d_type, Shape(cat_tensor.shape), cat_tensor)


def _ops_dispatch_signature_1_local_cpu_unknown(
    method,
    dtype: dtype,
    args,
    kwargs,
) -> Callable[[_CPUStorage], _CPUStorage]:
    if dtype.is_basic():
        return _ops_dispatch_signature_1_local_cpu_plain(method, args, kwargs)
    elif dtype.is_paillier():
        return _ops_dispatch_signature_1_local_cpu_paillier(method, args, kwargs)
    raise OpsDispatchUnsupportedError(method, False, device.CPU, dtype)


def _ops_dispatch_signature_2_local_cpu_unknown(
    method, dtype: dtype, args, kwargs
) -> Callable[[Any, Any], _CPUStorage]:
    if dtype.is_basic():
        return _ops_dispatch_signature_2_local_cpu_plain(method, args, kwargs)
    elif dtype.is_paillier():
        return _ops_dispatch_signature_2_local_cpu_paillier(method, args, kwargs)
    raise OpsDispatchUnsupportedError(method, False, device.CPU, dtype)


def _ops_dispatch_signature_3_local_cpu_unknown(
    method, dtype: dtype, args, kwargs
) -> Callable[[_CPUStorage], _CPUStorage]:
    if dtype.is_basic():
        return _ops_dispatch_signature_3_local_cpu_plain(method, args, kwargs)
    elif dtype.is_paillier():
        return _ops_dispatch_signature_3_local_cpu_paillier(method, args, kwargs)
    raise OpsDispatchUnsupportedError(method, False, device.CPU, dtype)


def _maybe_unwrap_storage(s):
    if isinstance(s, _CPUStorage):
        return s.data
    else:
        return s


def _get_method(method):
    # TODO: fix method list
    if hasattr(torch, method):
        return getattr(torch, method)
    else:
        from ._ops import custom_ops

        if method in custom_ops:
            return custom_ops[method]
        else:
            raise NotImplementedError(f"method `{method}` not found")


def _ops_dispatch_signature_1_local_cpu_plain(
    method, args, kwargs
) -> Callable[[_CPUStorage], _CPUStorage]:
    def _wrap(storage: _CPUStorage) -> _CPUStorage:
        func = _get_method(method)
        output = func(_maybe_unwrap_storage(storage), *args, **kwargs)
        output_dtype = dtype.from_torch_dtype(output.dtype)
        output_shape = Shape(output.shape)
        return _CPUStorage(output_dtype, output_shape, output)

    return _wrap


def _ops_dispatch_signature_1_local_cpu_paillier(
    method, args, kwargs
) -> Callable[[_CPUStorage], _CPUStorage]:
    raise OpsDispatchUnsupportedError(method, False, device.CPU, dtype.paillier)


def _ops_dispatch_signature_2_local_cpu_plain(
    method,
    args,
    kwargs,
) -> Callable[[Any, Any], _CPUStorage]:
    def _wrap(a, b) -> _CPUStorage:
        func = _get_method(method)
        output = func(
            _maybe_unwrap_storage(a), _maybe_unwrap_storage(b), *args, **kwargs
        )
        output_dtype = dtype.from_torch_dtype(output.dtype)
        output_shape = Shape(output.shape)
        return _CPUStorage(output_dtype, output_shape, output)

    return _wrap


def _ops_dispatch_signature_2_local_cpu_paillier(
    method,
    args,
    kwargs,
) -> Callable[[Any, Any], _CPUStorage]:

    # TODO: implement ops directly in C/Rust side
    def _wrap(a, b, **kwargs) -> _CPUStorage:
        a, b = _maybe_unwrap_storage(a), _maybe_unwrap_storage(b)
        func = getattr(operator, method)
        output = func(a, b)
        return _CPUStorage(dtype.paillier, Shape(output.shape), output)

    return _wrap


def _ops_dispatch_signature_3_local_cpu_plain(
    method,
    args,
    kwargs,
) -> Callable[[_CPUStorage], _CPUStorage]:
    def _wrap(storage: _CPUStorage) -> _CPUStorage:
        func = getattr(torch, method)
        output = func(storage.data, *args, **kwargs)
        output_dtype = dtype.from_torch_dtype(output.dtype)
        output_shape = Shape(output.shape)
        return _CPUStorage(output_dtype, output_shape, output)

    return _wrap


def _ops_dispatch_signature_3_local_cpu_paillier(
    method,
    args,
    kwargs,
) -> Callable[[_CPUStorage], _CPUStorage]:
    raise OpsDispatchUnsupportedError(method, False, device.CPU, dtype.paillier)
