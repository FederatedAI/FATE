from typing import overload

from ._tensor import Tensor

# TODO: parameter `keepdim` maybe a bit complex in distributed version, fix me later


@overload
def sum(a: Tensor, *, dtype=None) -> Tensor:
    ...


@overload
def sum(a: Tensor, dim, keepdim=False, *, dtype=None) -> Tensor:
    ...


def sum(a: Tensor, *args, **kwargs):
    return a.sum(*args, **kwargs)


def mean(a: Tensor, *args, **kwargs):
    return a.mean(*args, **kwargs)


def std(a: Tensor, *args, **kwargs):
    return a.std(*args, **kwargs)


def max():
    ...


def min():
    ...
