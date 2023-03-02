import torch

from ..unify import device as D
from ._dtype import dtype
from ._shape import Shape


def from_torch(t: torch.Tensor, device=D.CPU):
    if device == D.CPU:
        from ..storage.impl.torch_based._storage import _TorchStorage

        return _TorchStorage(dtype.from_torch_dtype(t.dtype), Shape(t.shape), t)
    raise NotImplementedError()


def randn(shape, _dtype: dtype, device=D.CPU):
    if device == D.CPU:
        from ..storage.impl.torch_based._storage import _TorchStorage

        return _TorchStorage(
            dtype.from_torch_dtype(_dtype), Shape(shape), torch.randn(shape, dtype=_dtype.to_torch_dtype())
        )
    raise NotImplementedError()


def ones(shape, _dtype: dtype, device=D.CPU):
    if device == D.CPU:
        from ..storage.impl.torch_based._storage import _TorchStorage

        return _TorchStorage(
            dtype.from_torch_dtype(_dtype), Shape(shape), torch.ones(shape, dtype=_dtype.to_torch_dtype())
        )
    raise NotImplementedError()


def zeros(shape, _dtype: dtype, device=D.CPU):
    if device == D.CPU:
        from ..storage.impl.torch_based._storage import _TorchStorage

        return _TorchStorage(
            dtype.from_torch_dtype(_dtype), Shape(shape), torch.zeros(shape, dtype=_dtype.to_torch_dtype())
        )
    raise NotImplementedError()
