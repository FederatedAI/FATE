from typing import Callable, List, Optional

import torch

from ..._base import _StorageOpsHandler, dtype
from .._register import _device_register

from ..._base import StorageBase
from ..._base import dtype
from ..._base import device


class _CPUStorageOpsHandler(_StorageOpsHandler):
    @classmethod
    def get_storage_op(cls, method: str, dtypes: List[Optional[dtype]]) -> Callable:
        # do nothing, just pass to torch
        if _is_bisic_types(dtypes):
            return _basic_ops(method)
        else:
            return _extend_ops.dispatch(method)


def _is_bisic_types(dtypes: List[Optional[dtype]]):
    for d_type in dtypes:
        if d_type is not None and not d_type.is_basic():
            return False
    return True


def _basic_ops(method: str):
    def _wrap(*args):
        func = getattr(torch, method)
        func_args = []

        def _recursive(in_list, out_list):
            for a in in_list:
                if isinstance(a, list):
                    out = []
                    _recursive(a, out)
                    out_list.append(out)
                elif isinstance(a, _CPUStorage):
                    out_list.append(a.data)

        _recursive(args, func_args)
        t = func(*func_args)
        output_dtype = dtype.from_torch_dtype(t.dtype)
        return _CPUStorage(output_dtype, t)

    return _wrap


class _CPUStorage(StorageBase):
    device = device.CPU

    def __init__(self, dtype: dtype, data) -> None:
        self.dtype = dtype
        self.data = data

    def __str__(self) -> str:
        return f"_CPUStorage(dtype={self.dtype}, data={self.data})"

    def __repr__(self) -> str:
        return self.__str__()


class _extend_ops:
    @classmethod
    def dispatch(cls, method: str) -> Callable:
        if method in {"add", "sub", "mul", "div"}:
            return _binary(method)
        raise ValueError(f"method: {method} not found")


def _binary(method):
    def _wrap(x, y):
        from ..cpu import _CPUStorage
        import operator

        if x.dtype == dtype.phe:
            return _CPUStorage(dtype.phe, getattr(operator, method)(x.data, y.data))

    return _wrap


_device_register.register(device.CPU, _CPUStorageOpsHandler)
