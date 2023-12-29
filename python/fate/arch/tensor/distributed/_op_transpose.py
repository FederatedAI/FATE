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

from ._tensor import DTensor, Shardings, _ShardingShapes, implements


@implements(torch.transpose)
def transpose(input: DTensor, dim0, dim1):
    shapes = transpose_shape(input.shardings.shapes, dim0, dim1)
    return DTensor(
        input.shardings.map_shard(lambda x: x.transpose(dim0, dim1).detach(), shapes=shapes.shapes, axis=shapes.axis)
    )

    # TODO: lazy transpose

    # if dim0 and dim1 are not in distributed axis:
    # 1. just transpose local tensor in each partition
    # 2. shapes should be modified.
    # if input.shardings.shapes.axis not in (dim0, dim1):
    #     return DTensor(
    #         input.shardings.map_shard(lambda x: x.transpose(dim0, dim1), shapes=shapes.shapes, axis=shapes.axis)
    #     )
    # # if dim0 and dim1 are in distributed axis:
    # # 1. local tensor in each partition should not be transposed.
    # # 2. only shapes and distributed axis should be modified.
    # else:
    #     return DTensor(
    #         Shardings(
    #             data=input.shardings._data,
    #             shapes=shapes.shapes,
    #             axis=shapes.axis,
    #             dtype=input.shardings._dtype,
    #             device=input.shardings._device,
    #         )
    #     )


def transpose_shape(shape: _ShardingShapes, dim0, dim1):
    # transpose shapes
    shapes = []
    for s in shape.shapes:
        s = list(s)
        s[dim0], s[dim1] = s[dim1], s[dim0]
        shapes.append(torch.Size(s))
    # transpose axis
    axis = shape.axis
    if dim0 == axis:
        axis = dim1
    elif dim1 == axis:
        axis = dim0
    return _ShardingShapes(shapes=shapes, axis=axis)
