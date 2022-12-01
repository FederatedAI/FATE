from typing import Any, Callable

from fate.arch.tensor.types import LStorage, dtype


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


def _ops_dispatch_signature_2_local_cpu_unknown(method, dtype: dtype, args, kwargs) -> Callable[[Any, Any], LStorage]:
    if dtype.is_basic():
        from .plain import _TorchStorage

        return _TorchStorage.binary(method, args, kwargs)
    elif dtype.is_paillier():
        from .paillier import _ops_dispatch_signature_2_local_cpu_paillier

        return _ops_dispatch_signature_2_local_cpu_paillier(method, args, kwargs)
