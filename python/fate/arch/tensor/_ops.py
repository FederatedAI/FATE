from ._tensor import Tensor
from ._base import DStorage, StorageBase
from functools import wraps

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


def _unary_func(func):
    method = func.__name__

    @wraps(func)
    def wrapper(x):
        return _apply_buildin_ops(method, x)

    return wrapper


def _binary_func(func):
    method = func.__name__

    @wraps(func)
    def wrapper(x, y):
        return _apply_buildin_ops(method, x, y)

    return wrapper


def _apply_buildin_ops(method, *args):
    # dispatch based on distributed or not
    storage_args = []
    dtypes = []
    has_distributed_tensor = False
    for arg in args:
        if isinstance(arg, Tensor):
            storage_args.append(arg.storage)
            dtypes.append(arg.dtype)
            if isinstance(arg.storage, DStorage):
                has_distributed_tensor = True
        else:
            storage_args.append(arg)
            dtypes.append(None)

    if has_distributed_tensor:
        return _apply_distributed_buildin_ops(method, dtypes, storage_args)
    else:
        return _apply_basic_buildin_ops(method, dtypes, storage_args)


def _apply_basic_buildin_ops(method, dtypes, storage_args):
    from .device import _device_register

    device = _get_device(storage_args)
    storage_op = _device_register.get_device_ops_handler(device).get_storage_op(
        method, dtypes
    )
    output_storage = storage_op(*storage_args)
    return Tensor(output_storage)


def _apply_distributed_buildin_ops(method, dtypes, storage_args):
    from .device import _device_register

    device = _get_device(storage_args)
    ops_hander = _device_register.get_device_ops_handler(device)
    storage_op = ops_hander.get_storage_op(method, dtypes)
    if method in {"neg", "sqrt", "exp", "log", "neg", "reciprocal", "square", "abs"}:
        assert len(storage_args) == 1 and isinstance(
            storage_args[0], DStorage
        ), f"can't apply {method} to given args: {storage_args}"
        output_storage = storage_args[0].elemwise_unary_op(storage_op)
        return Tensor(output_storage)

    if method in {"add", "sub", "mul", "div"}:
        if len(storage_args) != 2:
            raise ValueError(f"method `{method}` supported to be binary")
        x, y = storage_args
        if isinstance(x, DStorage) and isinstance(y, DStorage):
            output_storage = x.elemwise_binary_op(y, storage_op)
        else:
            raise ValueError("")
        return Tensor(output_storage)
    raise ValueError(f"method `{method}` not supported")


def _get_device(storage_args):
    devices = []

    def _recursive(arg_list):
        for arg in arg_list:
            if isinstance(arg, list):
                _recursive(arg)
            elif isinstance(arg, StorageBase):
                devices.append(arg.device)

    _recursive(storage_args)
    if len(devices) == 0:
        raise ValueError(f"dispatch failed, need at least one tensor in args")
    device = devices[0]
    for other_device in devices:
        if device != other_device:
            raise ValueError(f"tensor device not match")
    return device
