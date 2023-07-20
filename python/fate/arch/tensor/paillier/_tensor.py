import functools

import torch

_HANDLED_FUNCTIONS = {}


class PaillierTensor:
    def __init__(self, data, dtype) -> None:
        self._data = data
        self._dtype = dtype

    def __repr__(self) -> str:
        return f"<PaillierTensor shape={self.shape}, dtype={self.dtype}>"

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, item):
        if isinstance(item, int):
            return PaillierTensor(self._data.slice0(item), self._dtype)
        else:
            raise NotImplementedError(f"item {item} not supported")

    @property
    def shape(self):
        return torch.Size(self._data.shape)

    @property
    def dtype(self):
        return self._dtype

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in _HANDLED_FUNCTIONS or not all(issubclass(t, (torch.Tensor, PaillierTensor)) for t in types):
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)

    """implement arth magic"""

    def __add__(self, other):
        from ._ops import add

        return add(self, other)

    def __radd__(self, other):
        from ._ops import add

        return add(other, self)

    def __sub__(self, other):
        from ._ops import sub

        return sub(self, other)

    def __rsub__(self, other):
        from ._ops import rsub

        return rsub(self, other)

    def __mul__(self, other):
        from ._ops import mul

        return mul(self, other)

    def __rmul__(self, other):
        from ._ops import mul

        return mul(other, self)

    def __matmul__(self, other):
        from ._ops import matmul

        return matmul(self, other)

    def __rmatmul__(self, other):
        from ._ops import rmatmul_f

        return rmatmul_f(self, other)


def implements(torch_function):
    """Register a torch function override for PaillierTensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        _HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator
