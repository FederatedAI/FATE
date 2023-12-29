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
from enum import Enum


class device(Enum):
    def __init__(self, type: str, index) -> None:
        self.type = type
        self.index = index

    CPU = ("CPU", 1)
    CUDA = ("CUDA", 2)

    @classmethod
    def from_torch_device(cls, tensor_device):
        if tensor_device.type == "cpu":
            return device.CPU
        elif tensor_device.type == "cuda":
            return device.CUDA
        else:
            raise ValueError(f"device type {tensor_device.type} not supported")

    def to_torch_device(self):
        import torch

        if self.type == "CPU":
            return torch.device("cpu")
        elif self.type == "CUDA":
            return torch.device("cuda", self.index)
        else:
            raise ValueError(f"device type {self.type} not supported")
