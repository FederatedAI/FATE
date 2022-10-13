import torch
from ._base import StorageBase, dtype, DStorage, device
from fate.interface import Context
from typing import List
from .abc.tensor import PHEDecryptorABC, PHEEncryptorABC, PHETensorABC


def tensor(t: torch.Tensor):
    from .device.cpu import _CPUStorage

    storage = _CPUStorage(dtype.from_torch_dtype(t.dtype), t)
    return Tensor(storage)


def distributed_tensor(ctx: Context, tensors: List[torch.Tensor], partitions=3):
    from .device.cpu import _CPUStorage

    storages = [_CPUStorage(dtype.from_torch_dtype(t.dtype), t) for t in tensors]
    d_type = storages[0].dtype
    device = storages[0].device
    blocks = ctx.computing.parallelize(
        enumerate(storages), partition=partitions, include_key=True
    )
    storage = DStorage(blocks, 0, d_type, device)
    return Tensor(storage)


def _inject_op(func):
    method = func.__name__

    def _wrap(*args):
        from ._ops import _apply_buildin_ops

        return _apply_buildin_ops(method, *args)

    return _wrap


class Tensor:
    def __init__(self, storage: StorageBase) -> None:
        self._storage = storage

    @property
    def dtype(self):
        return self._storage.dtype

    @property
    def storage(self):
        return self._storage

    @property
    def device(self):
        return self._storage.device

    def to_local(self):
        if isinstance(self._storage, DStorage):
            return Tensor(self._storage.to_local())
        return self

    def __str__(self) -> str:
        return f"Tensor(device={self.device}, storage={self.storage})"

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return self.mul(other)

    def __div__(self, other):
        return self.div(other)

    """and others"""

    @_inject_op
    def add(self, other) -> "Tensor":
        ...

    @_inject_op
    def sub(self, other) -> "Tensor":
        ...

    @_inject_op
    def mul(self, other) -> "Tensor":
        ...

    @_inject_op
    def div(self, other) -> "Tensor":
        ...
