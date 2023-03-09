#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import functools
from typing import List

import torch

_HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """Register a torch function override for DStorage"""

    @functools.wraps(torch_function)
    def decorator(func):
        _HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class DAxis:
    def __init__(self, axis: int, partitions) -> None:
        self.axis = axis
        self.partitions = partitions

    def __str__(self) -> str:
        return f"DAxis<axis={self.axis}, partitions={self.partitions}>"


class DTensor:
    def __init__(
        self, blocks, shape: torch.Size, d_axis: DAxis, dtype: torch.dtype, device: torch.device, transposed=False
    ) -> None:
        self.blocks = blocks
        self._shape = shape
        self._dtype = dtype
        self._device = device
        self._d_axis = d_axis
        # TODO: fix me
        self.transposed = transposed

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in _HANDLED_FUNCTIONS or not all(issubclass(t, (torch.Tensor, DTensor)) for t in types):
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)

    @property
    def shape(self):
        return self._shape

    @property
    def d_axis(self) -> DAxis:
        return self._d_axis

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def transpose(self) -> "DTensor":
        return DTensor(self.blocks, self.shape.transpose(), self.dtype, self.device, not self.transposed)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, DTensor) and self._dtype == __o.dtype and self._device == __o.device:
            return torch.allclose(self.to_local(), __o.to_local())
        else:
            return False

    def __str__(self) -> str:
        return f"DStorage({self.device}, {self.dtype}, {self.shape})"

    def collect(self) -> List[torch.Tensor]:
        return [pair[1] for pair in sorted(self.blocks.collect())]

    def to_local(self) -> torch.Tensor:
        storages = self.collect()
        return torch.cat(storages, self._d_axis.axis)

    @classmethod
    def from_storages(cls, ctx, storages: List[torch.Tensor], d_axis=0, partitions=4):
        d_type = storages[0].dtype
        device = storages[0].device
        shape_size = storages[0].shape
        for storage in storages[1:]:
            if storage.dtype != d_type:
                raise RuntimeError(f"requires same dtype")
            if storage.device != device:
                raise RuntimeError(f"requires same device")
        blocks = ctx.computing.parallelize(enumerate(storages), partition=partitions, include_key=True)
        return DTensor(blocks, shape_size, DAxis(d_axis, partitions), d_type, device)

    # @classmethod
    # def elemwise_bc_op(
    #     cls,
    #     a: "DStorage",
    #     b: "DStorage",
    #     func: Callable[[LStorage, LStorage], LStorage],
    #     output_dtype=None,
    #     shape=None,
    #     **kwargs,
    # ):
    #     # TODO: remove this
    #     def _apply_transpose(func, lf, rf):
    #         def _wrap(lblk, rblk):
    #             if lf:
    #                 lblk = lblk.transpose()
    #             if rf:
    #                 rblk = rblk.transpose()
    #             return func(lblk, rblk)

    #         return _wrap

    #     if isinstance(a, DStorage) and not isinstance(b, DStorage):
    #         func = _apply_transpose(func, a.transposed, False)
    #         output_blocks = a.blocks.mapValues(lambda x: func(x, b, **kwargs))
    #     elif isinstance(b, DStorage) and not isinstance(a, DStorage):
    #         func = _apply_transpose(func, False, b.transposed)
    #         output_blocks = b.blocks.mapValues(lambda x: func(a, x, **kwargs))
    #     else:
    #         raise RuntimeError("exactly one DStorage required")
    #     if output_dtype is None:
    #         output_dtype = a._dtype
    #     if shape is None:
    #         shape = a.shape
    #     return DStorage(output_blocks, shape, output_dtype, a._device)
