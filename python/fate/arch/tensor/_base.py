from enum import Enum
from functools import reduce
from typing import Callable, List, Optional, Protocol, Union, overload

import torch

from ..unify import device


class dtype(Enum):
    def __init__(self, is_floating_point, is_signed, rank) -> None:
        self.is_floating_point = is_floating_point
        self.is_signed = is_signed
        self.rank = rank

    int32 = (False, True, 1)
    int64 = (False, True, 2)
    float32 = (True, True, 3)
    float64 = (True, True, 4)
    paillier = (True, True, 5)  # partially homomorphic encryption
    #
    def is_basic(self):
        return self == dtype.float32 or self == dtype.float64 or self == dtype.int32 or self == dtype.int64

    def is_paillier(self):
        return self == dtype.paillier

    def type_promoted(self, other: "dtype") -> "dtype":
        if self.rank < other.rank:
            return other
        else:
            return self

    def to_torch_dtype(self):
        if self == dtype.int32:
            return torch.int32
        if self == dtype.int64:
            return torch.int64
        if self == dtype.float64:
            return torch.float64
        if self == dtype.float32:
            return torch.float32
        raise TypeError(f"unsupported type: {self}")

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


class Shape:
    def __init__(self, size, d_axis=None) -> None:
        if isinstance(size, int):
            size = (size,)
        self.size = size
        self.d_axis = d_axis

    def transpose(self) -> "Shape":
        if len(self.size) != 2:
            raise RuntimeError(f"transpose of size {self.size} no supported")
        size = self.size[::-1]

        if self.d_axis is not None:
            d_axis = len(self.size) - 1 - self.d_axis
        else:
            d_axis = None
        return Shape(size, d_axis)

    def is_d_axis(self, axis: int):
        if self.d_axis is None:
            return False
        gap = abs(self.d_axis - axis)
        return gap == 0 or gap == len(self.size)

    def __len__(self):
        return len(self.size)

    def prod(self):
        return reduce(lambda x, y: x * y, self.size)

    def __str__(self) -> str:
        return f"Shape<size={self.size}, d_axis={self.d_axis}>"

    def __repr__(self) -> str:
        return self.__str__()

    def slice(self, key):
        if isinstance(key, int):
            ...

    @overload
    def __getitem__(self, key: int) -> int:
        ...

    @overload
    def __getitem__(self, key: slice) -> "Shape":
        ...

    def __getitem__(self, key):
        if isinstance(key, int):
            if -len(self.size) + 1 < key < len(self.size):
                return self.size[key]
            else:
                raise ValueError("out of bound")
        elif isinstance(key, slice):
            out = self.size[key]
            out_d_axis = None
            if self.d_axis is not None:
                d_axis_mask = [False] * len(self.size)
                d_axis_mask[self.d_axis] = True
                out_d_axis = None
                for i, v in enumerate(d_axis_mask[key]):
                    if v:
                        out_d_axis = i
            return Shape(out, out_d_axis)
        else:
            raise NotImplementedError(f"key type {type(key)}")

    @classmethod
    def broadcast_shape(cls, shapes: List["Shape"], raise_exception=True):
        max_len = 0
        for shape in shapes:
            if isinstance(shape.size, int):
                if max_len < 1:
                    max_len = 1
            elif isinstance(shape.size, tuple) or isinstance(shape.size, list):
                s = len(shape.size)
                if max_len < s:
                    max_len = s
        result = [1] * max_len
        d_axis = None
        shapes = [Shape((s.size,), s.d_axis) if isinstance(s.size, int) else s for s in shapes]
        for shape in shapes:
            if isinstance(shape.size, tuple) or isinstance(shape.size, list):
                if shape.d_axis is not None:
                    aligned_d_axis = max_len - len(shape.size) + shape.d_axis
                    if d_axis is None:
                        d_axis = aligned_d_axis
                    elif d_axis != aligned_d_axis:
                        if raise_exception:
                            raise RuntimeError("d_axis mismatch: d_axis should be equal after shape broadcast")
                        else:
                            return None
                for i in range(-1, -1 - len(shape.size), -1):
                    if shape.size[i] < 0:
                        if raise_exception:
                            raise RuntimeError(
                                "Trying to create tensor with negative dimension ({}): ({})".format(
                                    shape.size[i], shape.size[i]
                                )
                            )
                        else:
                            return None
                    if shape.size[i] == 1 or shape.size[i] == result[i]:
                        continue
                    if result[i] != 1:
                        if raise_exception:
                            raise RuntimeError("Shape mismatch: objects cannot be broadcast to a single shape")
                        else:
                            return None
                    result[i] = shape.size[i]
            else:
                if raise_exception:
                    raise RuntimeError(
                        "Input shapes should be of type ints, a tuple of ints, or a list of ints, got ",
                        shape,
                    )
                else:
                    return None
        # check d_axis
        # TODO: we may split local tensor into parts and distributed in future
        if d_axis is not None:
            for shape in shapes:
                if shape.d_axis is not None:
                    continue
                p = d_axis - (len(result) - len(shape.size))
                if p >= 0 and shape.size[p] != 1:
                    raise RuntimeError("Can't broadcast along distributed axis for Local Storage ")
        return Shape(result, d_axis)


class LStorage(Protocol):
    device: device
    dtype: dtype
    shape: "Shape"

    def tolist(self):
        ...

    def transpose(self) -> "LStorage":
        ...

    def to_local(self) -> "LStorage":
        ...


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
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def transpose(self) -> "DStorage":
        return DStorage(self.blocks, self.shape.transpose(), self.dtype, self.device, not self.transposed)

    def sum(self, *args, **kwargs):
        from .storage.agg import sum

        return sum(self, *args, **kwargs)

    def max(self, *args, **kwargs):
        from .storage.agg import max

        return max(self, *args, **kwargs)

    def mean(self, *args, **kwargs):
        from .storage.agg import mean

        return mean(self, *args, **kwargs)

    def std(self, *args, **kwargs):
        from .storage.agg import std

        return std(self, *args, **kwargs)

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
        return storages[0].cat(storages[1:], self.shape.d_axis)

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
        return DStorage(blocks, Shape(shape_size, d_axis), d_type, device)

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
        return DStorage(output_blocks, a.shape, output_dtype, a._device)


Storage = Union[LStorage, DStorage]
