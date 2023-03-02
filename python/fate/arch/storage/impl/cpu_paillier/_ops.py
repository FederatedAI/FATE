import operator

from ..torch_based import _TorchStorage
from .paillier import _RustPaillierStorage


def _maybe_unwrap(a, b):
    if isinstance(a, (_RustPaillierStorage, _TorchStorage)):
        a = a.data
    if isinstance(b, (_RustPaillierStorage, _TorchStorage)):
        b = b.data
    return a, b


def _binary(a, b, f):
    if isinstance(a, _RustPaillierStorage):
        return _RustPaillierStorage(a.dtype, a.shape, f(*_maybe_unwrap(a, b)))
    elif isinstance(b, _RustPaillierStorage):
        return _RustPaillierStorage(b.dtype, b.shape, f(*_maybe_unwrap(a, b)))
    else:
        raise NotImplementedError()


def mul(a, b):
    return _binary(a, b, operator.mul)


def sub(a, b):
    return _binary(a, b, operator.sub)


def truediv(a, b):
    return _binary(a, b, operator.truediv)


def add(a, b):
    return _binary(a, b, operator.add)


def matmul(a, b):
    return _binary(a, b, operator.matmul)
