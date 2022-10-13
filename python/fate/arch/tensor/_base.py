import abc
from typing import Callable, List, Optional, Protocol
from enum import Enum
from ..unify import device
import torch

from enum import Enum, auto


class dtype(Enum):
    def __init__(self, is_floating_point, is_signed, index) -> None:
        self.is_floating_point = is_floating_point
        self.is_signed = is_signed
        self.index = index

    int64 = (False, True, 1)
    int32 = (False, True, 2)
    float64 = (True, True, 3)
    float32 = (True, True, 4)
    phe = (True, True, 5)  # partially homomorphic encryption
    #
    def is_basic(self):
        return (
            self == dtype.float32
            or self == dtype.float64
            or self == dtype.int32
            or self == dtype.int64
        )

    @classmethod
    def from_torch_dtype(cls, t_type):
        if t_type == torch.int32:
            return dtype.int32
        if t_type == torch.int64:
            return dtype.int64
        if t_type == torch.float64:
            return dtype.float64
        if t_type == torch.float32:
            return dtype.float32
        raise TypeError(f"unsupported type: {t_type}")


class StorageBase(metaclass=abc.ABCMeta):
    device: device
    dtype: dtype

    def to_local(self):
        return self


class _StorageOpsHandler(Protocol):
    @classmethod
    def get_storage_op(cls, method: str, dtypes: List[Optional["dtype"]]) -> Callable:
        ...


class DStorage(StorageBase):
    def __init__(self, blocks, d_axis, dtype, device) -> None:
        self.blocks = blocks
        self.d_axis = d_axis
        self._dtype = dtype
        self._device = device

    def elemwise_unary_op(self, func: Callable[[StorageBase], StorageBase], output_dtype=None):
        output_block = self.blocks.mapValues(func)
        if output_dtype is None:
            output_dtype = self._dtype
        return DStorage(output_block, self.d_axis, output_dtype, self._device)

    def elemwise_binary_op(
        self, other: "DStorage", func: Callable[[StorageBase, StorageBase], StorageBase]
    ):
        output_blocks = self.blocks.join(other.blocks, func)
        output_dtype = self._dtype  #
        return DStorage(output_blocks, self.d_axis, output_dtype, self._device)

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def __str__(self) -> str:
        return (
            f"DStorage(blocks={self.blocks}, dtype={self.dtype}, d_axis={self.d_axis})"
        )

    def num_blocks(self):
        return self.blocks.count()

    def collect(self):
        return [pair[1] for pair in sorted(self.blocks.collect())]

    def to_local(self):
        # TODO: this is device specificated, fix me
        storages = self.collect()
        tensors = [storage.data for storage in storages]
        cat_tensor = torch.cat(tensors, self.d_axis)
        from .device.cpu import _CPUStorage

        return _CPUStorage(dtype.from_torch_dtype(cat_tensor.dtype), cat_tensor)
