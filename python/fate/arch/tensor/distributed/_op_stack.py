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

import typing

import torch
import logging

from ._tensor import DTensor, implements

logger = logging.getLogger(__name__)


@implements(torch.stack)
def stack(
    tensors: typing.Union[typing.Tuple[DTensor, ...], typing.List[DTensor]],
    dim: int = 0,
    *,
    out: DTensor = None,
):
    logger.warning("stack DTensors may be slow")
    raise ValueError("stack DTensors may be slow")
    if out is not None:
        raise NotImplementedError("stack does not support out")

    out = tensors[0]
    for tensor in tensors[1:]:
        # TODO: check shapes
        out = DTensor(
            out.shardings.join_shard(
                tensor.shardings,
                func=lambda x, y: torch.stack([x, y], dim=dim),
            )
        )
    return out
