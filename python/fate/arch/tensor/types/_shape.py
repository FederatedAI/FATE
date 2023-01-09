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
from functools import reduce
from typing import List, Optional, overload


class DAxis:
    def __init__(self, axis: int, partitions) -> None:
        self.axis = axis
        self.partitions = partitions

    def __str__(self) -> str:
        return f"DAxis<axis={self.axis}, partitions={self.partitions}>"


class Shape:
    def __init__(self, size, d_axis: Optional[DAxis] = None) -> None:
        if isinstance(size, int):
            size = (size,)
        self.size = size
        self.d_axis = d_axis

    def transpose(self) -> "Shape":
        if len(self.size) != 2:
            raise RuntimeError(f"transpose of size {self.size} no supported")
        size = self.size[::-1]

        if self.d_axis is not None:
            d_axis = DAxis(len(self.size) - 1 - self.d_axis.axis, self.d_axis.partitions)
        else:
            d_axis = None
        return Shape(size, d_axis)

    def is_d_axis(self, axis: int):
        if self.d_axis is None:
            return False
        gap = abs(self.d_axis.axis - axis)
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
            raise NotImplementedError(f"key {key}")
        if isinstance(key, list):
            if self.d_axis is None:
                raise NotImplementedError(f"key {key}")

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
                d_axis_mask[self.d_axis.axis] = True
                out_d_axis = None
                for i, v in enumerate(d_axis_mask[key]):
                    if v:
                        out_d_axis = DAxis(i, self.d_axis.partitions)
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
                    aligned_d_axis = DAxis(max_len - len(shape.size) + shape.d_axis.axis, shape.d_axis.partitions)
                    if d_axis is None:
                        d_axis = aligned_d_axis
                    elif d_axis.axis != aligned_d_axis.axis:
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
                p = d_axis.axis - (len(result) - len(shape.size))
                if p >= 0 and shape.size[p] != 1:
                    raise RuntimeError("Can't broadcast along distributed axis for Local Storage ")
        return Shape(result, d_axis)
