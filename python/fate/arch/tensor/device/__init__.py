from typing import Any, Callable

from .._base import Storage, device


def _ops_dispatch_signature1_local_unknown_unknown(
    method,
    _device,
    dtype,
    args,
    kwargs,
) -> Callable[[Storage], Storage]:
    if _device == device.CPU:
        from .cpu._base import _ops_dispatch_signature_1_local_cpu_unknown

        return _ops_dispatch_signature_1_local_cpu_unknown(method, dtype, args, kwargs)
    raise ValueError()


def _ops_dispatch_signature2_local_unknown_unknown(
    method, _device, dtype, args, kwargs
) -> Callable[[Any, Any], Storage]:
    if _device == device.CPU:
        from .cpu._base import _ops_dispatch_signature_2_local_cpu_unknown

        return _ops_dispatch_signature_2_local_cpu_unknown(method, dtype, args, kwargs)
    raise ValueError()
