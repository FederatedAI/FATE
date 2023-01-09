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
from functools import wraps

from .._exception import OpDispatchInvalidDevice, OpsDispatchBadSignatureError
from .._tensor import Tensor
from ..types import Shape


def auto_unary_op(func):
    method = func.__name__

    @wraps(func)
    def wrapper(x, *args, **kwargs):
        return dispatch_signature1(method, x, args, kwargs)

    return wrapper


def auto_binary_op(func):
    method = func.__name__

    @wraps(func)
    def wrapper(x, y, *args, **kwargs):
        return dispatch_signature2(method, x, y, args, kwargs)

    return wrapper


def _maybe_get_storage(tensor):
    if isinstance(tensor, Tensor):
        return tensor.storage
    else:
        return tensor


def _get_dispatch_info(tensors):
    _is_distributed = False
    _device = None
    _dtype = None
    for tensor in tensors:
        if isinstance(tensor, Tensor):
            # set distributed or local
            _is_distributed = _is_distributed or tensor.is_distributed

            # set device
            if _device is None:
                _device = tensor.device
            elif _device != tensor.device:
                raise OpDispatchInvalidDevice(f"device mismatch: {_device} and {tensor.device}")

            # set dtypes
            if _dtype is None:
                _dtype = tensor.dtype
            else:
                _dtype = _dtype.type_promoted(tensor.dtype)
    return _is_distributed, _device, _dtype


def dispatch_signature1(method, tensor, args, kwargs):
    if not isinstance(tensor, Tensor):
        raise OpsDispatchBadSignatureError(f"required exactly one tensor input, got {tensor}")
    from ..storage._ops import _ops_dispatch_signature1_unknown_unknown_unknown

    storage_op = _ops_dispatch_signature1_unknown_unknown_unknown(
        method=method,
        distributed=tensor.is_distributed,
        device=tensor.device,
        dtype=tensor.dtype,
        args=args,
        kwargs=kwargs,
    )
    storage = storage_op(_maybe_get_storage(tensor))
    return Tensor(storage)


def dispatch_signature2(method, tensor, other, args, kwargs, bc_shape_validate=True):
    if not isinstance(tensor, Tensor) and not isinstance(other, Tensor):
        raise OpsDispatchBadSignatureError(f"atleast one tensor input, got {tensor} and {other}")
    from ..storage._ops import _ops_dispatch_signature2_unknown_unknown_unknown

    if bc_shape_validate:
        if isinstance(tensor, Tensor) and isinstance(other, Tensor):
            if Shape.broadcast_shape([tensor.shape, other.shape], raise_exception=False) is None:
                raise RuntimeError(f"shape broadcast failed: {tensor.shape} and {other.shape}")
    _is_distributed, _device, _dtype = _get_dispatch_info([tensor, other])
    storage_op = _ops_dispatch_signature2_unknown_unknown_unknown(
        method, _is_distributed, _device, _dtype, args, kwargs
    )
    storage = storage_op(_maybe_get_storage(tensor), _maybe_get_storage(other))
    return Tensor(storage)
