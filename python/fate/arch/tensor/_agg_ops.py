from typing import overload

from ._ops import dispatch_signature3
from ._storage_ops import (
    _ops_dispatch_signature1_local_unknown_unknown,
    _ops_dispatch_signature2_local_unknown_unknown,
    _ops_dispatch_signature3_local_unknown_unknown,
)
from ._tensor import Tensor

# TODO: parameter `keepdim` maybe a bit complex in distributed version, fix me later


@overload
def sum(a: Tensor, *, dtype=None) -> Tensor:
    ...


@overload
def sum(a: Tensor, dim, keepdim=False, *, dtype=None) -> Tensor:
    ...


def sum(a: Tensor, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]

    def _mapper(device, dtype):
        return _ops_dispatch_signature3_local_unknown_unknown(
            "sum", device, dtype, args, kwargs
        )

    def _reducer(device, dtype):
        if dim and dim != a.shape.d_axis:
            return None
        return _ops_dispatch_signature2_local_unknown_unknown(
            "add", device, dtype, [], {}
        )

    return dispatch_signature3("sum", a, _mapper, _reducer, None, args, kwargs)


def mean(a: Tensor, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]

    def _mapper(device, dtype):
        return _ops_dispatch_signature3_local_unknown_unknown(
            "mean", device, dtype, args, kwargs
        )

    def _reducer(device, dtype):
        if dim and dim != a.shape.d_axis:
            return None
        return _ops_dispatch_signature2_local_unknown_unknown(
            "add", device, dtype, [], {}
        )

    def _post_func(device, dtype):
        def _warp(s):
            truediv = _ops_dispatch_signature2_local_unknown_unknown(
                "truediv", device, dtype, [], {}
            )
            return truediv(s, s.shape[0])

        return _post_func

    return dispatch_signature3("mean", a, _mapper, _reducer, _post_func, args, kwargs)


def max():
    ...


def min():
    ...
