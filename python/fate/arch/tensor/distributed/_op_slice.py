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

import torch
from fate.arch.tensor import _custom_ops

from ._tensor import DTensor, implements


@implements(_custom_ops.slice_f)
def slice_f(input: DTensor, key):
    # 1: int slice key means slice 0 dimention
    if isinstance(key, int):
        if 0 <= key < input.shape[0]:
            # 1.1: slice output in one of shardings
            if input.shardings.shapes.axis == 0:
                return input.shardings.map_reduce_shard_with_stride(
                    stride_mapper_func=lambda stride, _, s: [s[key - stride]]
                    if stride <= key < stride + s.shape[0]
                    else [],
                    reducer_func=lambda x, y: [*x, *y],
                )[0]
            # 1.2: slice output is distributed
            else:
                return DTensor(
                    input.shardings.map_shard(lambda s: s[key], shapes=input.shardings.shapes.squeeze((0,)))
                )

        else:
            raise IndexError(f"index {key} is out of bounds for dimension 0 with size {input.shape[0]}")

    # 2: list slice key
    if isinstance(key, list):
        for k in key:
            if k < 0 or k >= input.shape[0]:
                raise IndexError(f"index {k} is out of bounds for dimension 0 with size {input.shape[0]}")

        if input.shardings.shapes.axis == 0:
            outputs = input.shardings.map_reduce_shard_with_stride(
                stride_mapper_func=lambda stride, _, s: [
                    (i, s[k - stride]) for i, k in enumerate(key) if stride <= k < stride + s.shape[0]
                ],
                reducer_func=lambda x, y: [*x, *y],
            )
            return torch.cat([v for _, v in sorted(outputs)])
        else:
            return DTensor(input.shardings.map_shard(lambda s: s[key], shapes=input.shardings.shapes.squeeze((0,))))

    # 3: slice key
    if isinstance(key, slice):
        start, stop, step = key.indices(input.shape[0])
        indices = list(range(start, stop, step))
        return slice_f(input, indices)

    # 4: tuple key for multi-dimensional slicing
    if isinstance(key, tuple):
        raise NotImplementedError("tuple key {key}")
        # result = input
        # for dim, k in enumerate(key):
        #     if isinstance(k, (int, list, slice)):
        #         ...
        #     else:
        #         raise NotImplementedError(f"slice_f on {key}")
        # return result

    raise NotImplementedError(f"slice_f on {key}")
