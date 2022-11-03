from typing import Any, Callable

from .._base import StorageBase, _StorageOpsHandler, device


class _device_register:
    _device_handlers = {}

    @classmethod
    def register(cls, device: device, storage_ops_handler):
        cls._device_handlers[device] = storage_ops_handler

    @classmethod
    def get_device_ops_handler(cls, device: device) -> "_StorageOpsHandler":
        if device not in cls._device_handlers:
            raise NotImplementedError(f"lack of implemention for device {device}")
        return cls._device_handlers[device]


def _ops_dispatch_signature1_local_unknown_unknown(
    method,
    _device,
    dtype,
    args,
    kwargs,
) -> Callable[[StorageBase], StorageBase]:
    if _device == device.CPU:
        from .cpu._base import _ops_dispatch_signature_1_local_cpu_unknown

        return _ops_dispatch_signature_1_local_cpu_unknown(method, dtype, args, kwargs)
    raise ValueError()


def _ops_dispatch_signature2_local_unknown_unknown(
    method, _device, dtype, args, kwargs
) -> Callable[[Any, Any], StorageBase]:
    if _device == device.CPU:
        from .cpu._base import _ops_dispatch_signature_2_local_cpu_unknown

        return _ops_dispatch_signature_2_local_cpu_unknown(method, dtype, args, kwargs)
    raise ValueError()


def _ops_dispatch_signature3_local_unknown_unknown(
    method, _device, dtype, args, kwargs
) -> Callable[[StorageBase], StorageBase]:
    if _device == device.CPU:
        from .cpu._base import _ops_dispatch_signature_3_local_cpu_unknown

        return _ops_dispatch_signature_3_local_cpu_unknown(method, dtype, args, kwargs)
    raise ValueError()
