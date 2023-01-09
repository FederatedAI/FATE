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
from typing import Callable, List, Optional

from fate.arch.unify import device

from ._dtype import dtype
from ._lstorage import LStorage
from ._shape import DAxis, Shape


class DStorage:
    def __init__(self, blocks, shape: Shape, dtype: dtype, device: device, transposed=False) -> None:
        self.blocks = blocks
        self._shape = shape
        self._dtype = dtype
        self._device = device
        self.transposed = transposed

    @property
    def shape(self):
        return self._shape

    @property
    def d_axis(self) -> DAxis:
        if self._shape.d_axis is None:
            raise ValueError(f"DStorage should not have none daxis")
        return self._shape.d_axis

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def transpose(self) -> "DStorage":
        return DStorage(self.blocks, self.shape.transpose(), self.dtype, self.device, not self.transposed)

    def sum(self, *args, **kwargs):
        from ..storage.distributed.agg import sum

        return sum(self, *args, **kwargs)

    def max(self, *args, **kwargs):
        from ..storage.distributed.agg import max

        return max(self, *args, **kwargs)

    def min(self, *args, **kwargs):
        from ..storage.distributed.agg import min

        return min(self, *args, **kwargs)

    def mean(self, *args, **kwargs):
        from ..storage.distributed.agg import mean

        return mean(self, *args, **kwargs)

    def std(self, *args, **kwargs):
        from ..storage.distributed.agg import std

        return std(self, *args, **kwargs)

    def var(self, *args, **kwargs):
        from ..storage.distributed.agg import var

        return var(self, *args, **kwargs)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, DStorage) and self._dtype == __o.dtype and self._device == __o.device:
            return self.to_local() == __o.to_local()
        else:
            return False

    def __str__(self) -> str:
        return f"DStorage({self.device}, {self.dtype}, {self.shape})"

    def num_blocks(self):
        return self.blocks.count()

    def collect(self) -> List[LStorage]:
        return [pair[1] for pair in sorted(self.blocks.collect())]

    def to_local(self):
        storages = self.collect()
        return storages[0].cat(storages[1:], self.shape.d_axis.axis)

    @classmethod
    def from_storages(cls, ctx, storages: List[LStorage], d_axis=0, partitions=4):
        d_type = storages[0].dtype
        device = storages[0].device
        shape_size = storages[0].shape.size
        if storages[0].shape.d_axis is not None:
            raise RuntimeError(f"can't create DStorage from list of DStorage")
        if isinstance(shape_size, int):
            shape_size = (shape_size,)
        shape_len = len(shape_size)
        if d_axis > shape_len or d_axis < 0:
            raise RuntimeError(f"d_axis out of bound")
        for storage in storages[1:]:
            if storage.dtype != d_type:
                raise RuntimeError(f"requires same dtype")
            if storage.device != device:
                raise RuntimeError(f"requires same device")
            if storage.shape.d_axis is not None:
                raise RuntimeError(f"can't create DStorage from list of DStorage")
            if len(storage.shape.size) != shape_len:
                raise RuntimeError(f"requires same shape len")
            for i in range(shape_len):
                if i == d_axis:
                    shape_size = (
                        *shape_size[:d_axis],
                        shape_size[d_axis] + storage.shape.size[d_axis],
                        *shape_size[(d_axis + 1) :],
                    )
                else:
                    if shape_size[i] != storage.shape.size[i]:
                        raise RuntimeError(f"requires same shape except d_axis")
        blocks = ctx.computing.parallelize(enumerate(storages), partition=partitions, include_key=True)
        d_axis_cls = DAxis(d_axis, [s.shape.size[d_axis] for s in storages])
        return DStorage(blocks, Shape(shape_size, d_axis_cls), d_type, device)

    @classmethod
    def unary_op(
        cls,
        a: "DStorage",
        mapper: Callable[[LStorage], LStorage],
        output_shape: Optional[Shape] = None,
        output_dtype=None,
    ):
        def _apply_transpose(func, flag):
            def _wrap(blk):
                if flag:
                    blk = blk.transpose()
                return func(blk)

            return _wrap

        mapper = _apply_transpose(mapper, a.transposed)
        output_block = a.blocks.mapValues(mapper)
        if output_dtype is None:
            output_dtype = a._dtype
        if output_shape is None:
            output_shape = a.shape
        return DStorage(output_block, output_shape, output_dtype, a._device)

    @classmethod
    def elemwise_unary_op(
        cls,
        a,
        mapper: Callable[[LStorage], LStorage],
        output_dtype=None,
    ):
        def _apply_transpose(func, flag):
            def _wrap(blk):
                if flag:
                    blk = blk.transpose()
                return func(blk)

            return _wrap

        mapper = _apply_transpose(mapper, a.transposed)
        output_block = a.blocks.mapValues(mapper)
        if output_dtype is None:
            output_dtype = a._dtype
        return DStorage(output_block, a.shape, output_dtype, a._device)

    @classmethod
    def agg_unary_op(
        cls,
        a: "DStorage",
        mapper: Callable[[LStorage], LStorage],
        reducer,
        post_func,
        output_dtype=None,
    ):
        if output_dtype is None:
            output_dtype = a._dtype
        output_block = a.blocks.mapValues(mapper)
        if reducer is not None:
            output_block = output_block.reduce(reducer)

            if post_func is not None:
                output_block = post_func(output_block)
            return output_block
        else:
            return DStorage(output_block, a.shape, output_dtype, a._device)

    @classmethod
    def elemwise_binary_op(
        cls,
        a: "DStorage",
        b: "DStorage",
        binary_mapper: Callable[[LStorage, LStorage], LStorage],
        output_dtype=None,
    ):
        def _apply_transpose(func, lf, rf):
            def _wrap(lblk, rblk):
                if lf:
                    lblk = lblk.transpose()
                if rf:
                    rblk = rblk.transpose()
                return func(lblk, rblk)

            return _wrap

        binary_mapper = _apply_transpose(binary_mapper, a.transposed, b.transposed)
        output_blocks = a.blocks.join(b.blocks, binary_mapper)
        if output_dtype is None:
            output_dtype = a._dtype
        return DStorage(output_blocks, a.shape, output_dtype, a._device)

    @classmethod
    def elemwise_bc_op(
        cls,
        a: "DStorage",
        b: "DStorage",
        func: Callable[[LStorage, LStorage], LStorage],
        output_dtype=None,
        shape=None,
        **kwargs,
    ):
        def _apply_transpose(func, lf, rf):
            def _wrap(lblk, rblk):
                if lf:
                    lblk = lblk.transpose()
                if rf:
                    rblk = rblk.transpose()
                return func(lblk, rblk)

            return _wrap

        if isinstance(a, DStorage) and not isinstance(b, DStorage):
            func = _apply_transpose(func, a.transposed, False)
            output_blocks = a.blocks.mapValues(lambda x: func(x, b, **kwargs))
        elif isinstance(b, DStorage) and not isinstance(a, DStorage):
            func = _apply_transpose(func, False, b.transposed)
            output_blocks = b.blocks.mapValues(lambda x: func(a, x, **kwargs))
        else:
            raise RuntimeError("exactly one DStorage required")
        if output_dtype is None:
            output_dtype = a._dtype
        if shape is None:
            shape = a.shape
        return DStorage(output_blocks, shape, output_dtype, a._device)

    def local_ops_helper(self):
        from ..storage._helper import local_ops_helper

        return local_ops_helper(self.device, self.dtype)
