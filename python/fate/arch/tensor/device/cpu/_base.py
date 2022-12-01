from typing import Any, Callable

import torch

from ..._base import LStorage, Shape, device, dtype
from ..._exception import OpsDispatchUnsupportedError


def _ops_dispatch_signature_1_local_cpu_unknown(
    method,
    dtype: dtype,
    args,
    kwargs,
) -> Callable[[LStorage], LStorage]:
    if dtype.is_basic():
        from .plain import _TorchStorage

        return _TorchStorage.unary(method, args, kwargs)
    elif dtype.is_paillier():
        from .paillier import _ops_dispatch_signature_1_local_cpu_paillier

        return _ops_dispatch_signature_1_local_cpu_paillier(method, args, kwargs)
    raise OpsDispatchUnsupportedError(method, False, device.CPU, dtype)


def _ops_dispatch_signature_2_local_cpu_unknown(method, dtype: dtype, args, kwargs) -> Callable[[Any, Any], LStorage]:
    if dtype.is_basic():
        from .plain import _TorchStorage

        return _TorchStorage.binary(method, args, kwargs)
    elif dtype.is_paillier():
        from .paillier import _ops_dispatch_signature_2_local_cpu_paillier

        return _ops_dispatch_signature_2_local_cpu_paillier(method, args, kwargs)
    raise OpsDispatchUnsupportedError(method, False, device.CPU, dtype)


# def _ops_dispatch_signature_3_local_cpu_unknown(
#     method, dtype: dtype, args, kwargs
# ) -> Callable[[_CPUStorage], _CPUStorage]:
#     if dtype.is_basic():
#         return _ops_dispatch_signature_3_local_cpu_plain(method, args, kwargs)
#     elif dtype.is_paillier():
#         return _ops_dispatch_signature_3_local_cpu_paillier(method, args, kwargs)
#     raise OpsDispatchUnsupportedError(method, False, device.CPU, dtype)
