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
