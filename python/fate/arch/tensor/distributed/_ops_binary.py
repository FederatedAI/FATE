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

from ._tensor import DTensor, implements


@implements(torch.add)
def add(input, other):
    return _binary(input, other, torch.add)


@implements(torch.sub)
def sub(input, other):
    return _binary(input, other, torch.sub)


@implements(torch.rsub)
def rsub(input, other):
    return _binary(input, other, torch.rsub)


@implements(torch.mul)
def mul(input, other):
    return _binary(input, other, torch.mul)


def _create_meta_tensor(x):
    if isinstance(x, (torch.Tensor, DTensor)):
        return torch.zeros(*x.shape, device=torch.device("meta"), dtype=x.dtype)
    else:
        return torch.tensor(x, device=torch.device("meta"))


@implements(torch.div)
def div(input, other, *, rounding_mode=None):
    _x = _create_meta_tensor(input)
    _y = _create_meta_tensor(other)
    _z = torch.div(_x, _y, rounding_mode=rounding_mode)

    return _binary(input, other, lambda x, y: torch.div(x, y, rounding_mode=rounding_mode), dtype_promote_to=_z.dtype)


def _binary(input, other, op, swap_operad=False, dtype_promote_to=None):
    # swap input and output if input is not DTensor
    if not isinstance(input, DTensor):
        return _binary(other, input, op, swap_operad=not swap_operad, dtype_promote_to=dtype_promote_to)

    if isinstance(other, DTensor):
        if swap_operad:
            return DTensor(other.shardings.join_shard(input.shardings, op, dtype_promote_to=dtype_promote_to))
        else:
            return DTensor(input.shardings.join_shard(other.shardings, op, dtype_promote_to=dtype_promote_to))

    # other is local tensor, broadcast to partitions
    else:
        if isinstance(other, torch.Tensor):
            shapes = input.shardings.shapes.bc_shapes(other.shape)
        else:
            # other is scalar
            shapes = input.shardings.shapes.bc_shapes(torch.Size([]))

        if swap_operad:
            return DTensor(
                input.shardings.map_shard(
                    lambda x: op(other, x),
                    dtype_promote_to=dtype_promote_to,
                    shapes=shapes.shapes,
                    axis=shapes.axis,
                )
            )

        else:
            return DTensor(
                input.shardings.map_shard(
                    lambda x: op(x, other),
                    dtype_promote_to=dtype_promote_to,
                    shapes=shapes.shapes,
                    axis=shapes.axis,
                )
            )
