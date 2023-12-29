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


@implements(torch.exp)
def exp(input: DTensor):
    return DTensor(input.shardings.map_shard(torch.exp, dtype_promote_to=torch.float32))


@implements(torch.log)
def log(input: DTensor):
    return DTensor(input.shardings.map_shard(torch.log, dtype_promote_to=torch.float32))


@implements(torch.square)
def square(input: DTensor):
    return DTensor(input.shardings.map_shard(torch.square))


@implements(torch.sigmoid)
def sigmoid(input: DTensor):
    return DTensor(input.shardings.map_shard(torch.sigmoid))


@implements(torch.neg)
def neg(input: DTensor):
    return DTensor(input.shardings.map_shard(torch.neg))
