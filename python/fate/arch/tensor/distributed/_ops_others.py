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


@implements(_custom_ops.to_local_f)
def to_local_f(input: DTensor):
    return input.shardings.merge()


@implements(_custom_ops.encode_as_int_f)
def encode_as_int_f(input: DTensor, precision):
    return DTensor(input.shardings.map_shard(lambda x: (x * 2**precision).type(torch.int64), dtype=torch.int64))
