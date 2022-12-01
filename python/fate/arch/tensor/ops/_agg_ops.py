from typing import overload

from .._tensor import Tensor

# TODO: parameter `keepdim` maybe a bit complex in distributed version, fix me later


@overload
def sum(a: Tensor, *, dtype=None) -> Tensor:
    ...


@overload
def sum(a: Tensor, dim, keepdim=False, *, dtype=None) -> Tensor:
    ...


def sum(a: Tensor, *args, **kwargs):
    return Tensor(a.storage.sum(*args, **kwargs))


def mean(a: Tensor, *args, **kwargs):
    return Tensor(a.storage.mean(*args, **kwargs))


def std(a: Tensor, *args, **kwargs):
    return Tensor(a.storage.std(*args, **kwargs))


def var(a: Tensor, *args, **kwargs):
    return Tensor(a.storage.var(*args, **kwargs))


def max():
    ...


def min():
    ...
