from functools import wraps

from ._base import DStorage, Shape
from ._exception import OpDispatchInvalidDevice, OpsDispatchBadSignatureError
from ._storage_ops import (
    _ops_dispatch_signature1_unknown_unknown_unknown,
    _ops_dispatch_signature2_unknown_unknown_unknown,
    _ops_dispatch_signature3_unknown_unknown_unknown,
)
from ._tensor import Tensor


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


# def auto_unary_agg_op(func):
#     method = func.__name__

#     @wraps(func)
#     def wrapper(x, *args, **kwargs):
#         return dispatch_signature3(method, x, args, kwargs)

#     return wrapper


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
            _is_distributed = _is_distributed or isinstance(tensor.storage, DStorage)

            # set device
            if _device is None:
                _device = tensor.device
            elif _device != tensor.device:
                raise OpDispatchInvalidDevice(
                    f"device mismatch: {_device} and {tensor.device}"
                )

            # set dtypes
            if _dtype is None:
                _dtype = tensor.dtype
            else:
                _dtype = _dtype.type_promoted(tensor.dtype)
    return _is_distributed, _device, _dtype


def dispatch_signature1(method, tensor, args, kwargs):
    if not isinstance(tensor, Tensor):
        raise OpsDispatchBadSignatureError(
            f"required exactly one tensor input, got {tensor}"
        )
    storage_op = _ops_dispatch_signature1_unknown_unknown_unknown(
        method=method,
        distributed=isinstance(tensor.storage, DStorage),
        device=tensor.device,
        dtype=tensor.dtype,
        args=args,
        kwargs=kwargs,
    )
    storage = storage_op(_maybe_get_storage(tensor))
    return Tensor(storage)


def dispatch_signature2(method, tensor, other, args, kwargs, bc_shape_validate=True):
    if not isinstance(tensor, Tensor) and not isinstance(other, Tensor):
        raise OpsDispatchBadSignatureError(
            f"atleast one tensor input, got {tensor} and {other}"
        )
    if bc_shape_validate:
        if isinstance(tensor, Tensor) and isinstance(other, Tensor):
            if (
                Shape.broadcast_shape(
                    [tensor.shape, other.shape], raise_exception=False
                )
                is None
            ):
                raise RuntimeError(
                    f"shape broadcast failed: {tensor.shape} and {other.shape}"
                )
    _is_distributed, _device, _dtype = _get_dispatch_info([tensor, other])
    storage_op = _ops_dispatch_signature2_unknown_unknown_unknown(
        method, _is_distributed, _device, _dtype, args, kwargs
    )
    storage = storage_op(_maybe_get_storage(tensor), _maybe_get_storage(other))
    return Tensor(storage)


def dispatch_signature3(method, tensor, mapper, reducer, post_func, args, kwargs):
    if not isinstance(tensor, Tensor):
        raise OpsDispatchBadSignatureError(
            f"required exactly one tensor input, got {tensor}"
        )
    _device = tensor.device
    _dtype = tensor.dtype
    storage_op = _ops_dispatch_signature3_unknown_unknown_unknown(
        method=method,
        distributed=isinstance(tensor.storage, DStorage),
        device=_device,
        dtype=_dtype,
        mapper=mapper(_device, _dtype),
        reducer=reducer(_device, _dtype),
        post_func=None if post_func is None else post_func(_device, _dtype),
        args=args,
        kwargs=kwargs,
    )
    storage = storage_op(_maybe_get_storage(tensor))
    return Tensor(storage)


# def dispatch(method, signature_type, distributed, args, kwargs):
#     if distributed:
#         return signature_type_dispatch_distributed(method, signature_type, args, kwargs)
#     else:
#         return signature_type_dispatch.dispatch(method, signature_type, args, kwargs)


# class signature_type_dispatch:
#     _signatures: Dict[str, "signature"] = {}

#     @classmethod
#     def dispatch(cls, method, signature_type, args, kwargs) -> Callable:
#         return cls._signatures[signature_type].dispatch(method, args, kwargs)


# class signature:
#     def __init__(self, func) -> None:
#         self._devices: Dict[device, Any] = {}
#         self._func = func

#     def get_device(self, args, kwargs) -> device:
#         return self._func(args, kwargs)

#     def dispatch(self, method, args, kwargs) -> Callable:
#         device = self.get_device(args, kwargs)
#         return self._devices[device](method)

#     def __call__(
#         self, device: device
#     ) -> Callable[[Callable[[str], Callable]], Callable[[str], Callable]]:
#         def _wrap(func: Callable[[str], Callable]):
#             self._devices[device] = func
#             return func

#         return _wrap


# unary_signature = signature()


# @unary_signature(device.CPU)
# def add(method: str) -> Callable:
#     ...


# def signature_type_dispatch_distributed(method, signature_type) -> Callable:
#     ...


# def _apply_buildin_storage_ops1(method, input: StorageBase, *, **kwargs):
#     ...

# def _apply_buildin_ops2(method, input, other, *, **kwargs):
#     ...


# def _apply_buildin_ops(method, *args):
#     # dispatch based on distributed or not
#     storage_args = []
#     dtypes = []
#     has_distributed_tensor = False
#     for arg in args:
#         if isinstance(arg, Tensor):
#             storage_args.append(arg.storage)
#             dtypes.append(arg.dtype)
#             if isinstance(arg.storage, DStorage):
#                 has_distributed_tensor = True
#         else:
#             storage_args.append(arg)
#             dtypes.append(None)

#     if has_distributed_tensor:
#         return _apply_distributed_buildin_ops(method, dtypes, storage_args)
#     else:
#         return _apply_basic_buildin_ops(method, dtypes, storage_args)


# def _apply_basic_buildin_ops(method, dtypes, storage_args):
#     from .device import _device_register

#     device = _get_device(storage_args)
#     storage_op = _device_register.get_device_ops_handler(device).get_storage_op(
#         method, dtypes
#     )
#     output_storage = storage_op(*storage_args)
#     return Tensor(output_storage)


# def _apply_distributed_buildin_ops(method, dtypes, storage_args):
#     from .device import _device_register

#     device = _get_device(storage_args)
#     ops_hander = _device_register.get_device_ops_handler(device)
#     storage_op = ops_hander.get_storage_op(method, dtypes)
#     if method in {
#         "neg",
#         "sqrt",
#         "exp",
#         "log",
#         "neg",
#         "reciprocal",
#         "square",
#         "abs",
#         "sigmoid",
#     }:
#         assert len(storage_args) == 1 and isinstance(
#             storage_args[0], DStorage
#         ), f"can't apply {method} to given args: {storage_args}"
#         output_storage = storage_args[0].elemwise_unary_op(storage_op)
#         return Tensor(output_storage)

#     if method in {"add", "sub", "mul", "div", "truediv"}:
#         if len(storage_args) != 2:
#             raise ValueError(f"method `{method}` supported to be binary")
#         x, y = storage_args
#         if isinstance(x, DStorage) and isinstance(y, DStorage):
#             output_storage = x.elemwise_binary_op(y, storage_op)
#         else:
#             raise ValueError("")
#         return Tensor(output_storage)
#     raise ValueError(f"method `{method}` not supported")


# def _get_device(storage_args):
#     devices = []

#     def _recursive(arg_list):
#         for arg in arg_list:
#             if isinstance(arg, list):
#                 _recursive(arg)
#             elif isinstance(arg, StorageBase):
#                 devices.append(arg.device)

#     _recursive(storage_args)
#     if len(devices) == 0:
#         raise ValueError(f"dispatch failed, need at least one tensor in args")
#     device = devices[0]
#     for other_device in devices:
#         if device != other_device:
#             raise ValueError(f"tensor device not match")
#     return device

# def op(func):
#     return warp_func(func)


# class warp_func:
#     def __init__(self, func) -> None:
#         self._fallback_func = func
#         self._specialize_funcs = []

#     def __call__(self, *args: Any, **kwds: Any):
#         func = self.first_match_specialized(*args, **kwds)
#         return func(*args, **kwds)

#     def specialize(self, **kwargs):
#         def _warp(func):
#             self._specialize_funcs.append((kwargs, func))
#             return func

#         return _warp

#     def first_match_specialized(self, *args, **kwds)
#         ...
