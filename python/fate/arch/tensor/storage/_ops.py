from typing import Any, Callable

from ..types import DStorage, Storage
from .local.device import (
    _ops_dispatch_signature1_local_unknown_unknown,
    _ops_dispatch_signature2_local_unknown_unknown,
)


# signature1: elemwise (tensor) -> tensor
def _ops_dispatch_signature1_unknown_unknown_unknown(
    method, distributed, device, dtype, args, kwargs
) -> Callable[[Storage], Storage]:
    if distributed:
        return _ops_dispatch_signature1_distributed_unknown_unknown(method, device, dtype, args, kwargs)
    else:

        return _ops_dispatch_signature1_local_unknown_unknown(method, device, dtype, args, kwargs)


def _ops_dispatch_signature1_distributed_unknown_unknown(
    method, device, dtype, args, kwargs
) -> Callable[[Storage], Storage]:

    local_ops = _ops_dispatch_signature1_local_unknown_unknown(method, device, dtype, args, kwargs)

    def _wrap(storage: Storage) -> DStorage:
        return DStorage.elemwise_unary_op(
            storage, local_ops, dtype
        )  # FIXME: infer output dtype is hard without additional table call

    return _wrap


# signature2: elemwise (tensor, tensor) -> tensor
def _ops_dispatch_signature2_unknown_unknown_unknown(
    method, distributed, device, dtype, args, kwargs
) -> Callable[[Any, Any], Storage]:
    if distributed:
        return _ops_dispatch_signature2_distributed_unknown_unknown(method, device, dtype, args, kwargs)
    else:

        return _ops_dispatch_signature2_local_unknown_unknown(method, device, dtype, args, kwargs)


def _ops_dispatch_signature2_distributed_unknown_unknown(
    method, device, dtype, args, kwargs
) -> Callable[[Any, Any], Storage]:
    local_ops = _ops_dispatch_signature2_local_unknown_unknown(method, device, dtype, args, kwargs)

    def _wrap(storage1, storage2, **kwargs) -> DStorage:
        if isinstance(storage1, DStorage) and isinstance(storage2, DStorage):
            return DStorage.elemwise_binary_op(storage1, storage2, local_ops, dtype, **kwargs)
        else:
            # then storage2 should be broadcast
            return DStorage.elemwise_bc_op(storage1, storage2, local_ops, dtype, **kwargs)

    return _wrap
