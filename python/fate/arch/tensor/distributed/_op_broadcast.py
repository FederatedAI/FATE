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
import logging

from ._tensor import DTensor, implements

logger = logging.getLogger(__name__)


@implements(torch.broadcast_tensors)
def broadcast_tensors(*input: DTensor):
    for t in input[1:]:
        if isinstance(t, DTensor):
            if t.shardings != input[0].shardings:
                raise RuntimeError("broadcast_tensors expects all inputs to be broadcastable to the first input")
        if torch.broadcast_shapes(input[0].shape, t.shape) != input[0].shape:
            raise RuntimeError("broadcast_tensors expects all inputs to be broadcastable to the first input")
    return input
