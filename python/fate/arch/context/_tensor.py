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

from ..tensor import Tensor as FPTensor
from ..unify import device


class TensorKit:
    def __init__(self, computing, device: device) -> None:
        self.computing = computing
        self.device = device

    def random_tensor(self, shape, num_partition=1) -> FPTensor:
        from fate.arch.tensor.impl.tensor.distributed import FPTensorDistributed

        parts = []
        first_dim_approx = shape[0] // num_partition
        last_part_first_dim = shape[0] - (num_partition - 1) * first_dim_approx
        assert first_dim_approx > 0
        for i in range(num_partition):
            if i == num_partition - 1:
                parts.append(
                    torch.rand(
                        (
                            last_part_first_dim,
                            *shape[1:],
                        )
                    )
                )
            else:
                parts.append(torch.rand((first_dim_approx, *shape[1:])))
        return FPTensor(
            FPTensorDistributed(self.computing.parallelize(parts, include_key=False, partition=num_partition)),
        )

    def create_tensor(self, tensor: torch.Tensor) -> "FPTensor":
        return FPTensor(tensor)
