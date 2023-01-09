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

import torch


class dtype(Enum):
    def __init__(self, is_floating_point, is_signed, rank) -> None:
        self.is_floating_point = is_floating_point
        self.is_signed = is_signed
        self.rank = rank

    int32 = (False, True, 1)
    int64 = (False, True, 2)
    float32 = (True, True, 3)
    float64 = (True, True, 4)
    paillier = (True, True, 5)  # partially homomorphic encryption
    #
    def is_basic(self):
        return self == dtype.float32 or self == dtype.float64 or self == dtype.int32 or self == dtype.int64

    def is_paillier(self):
        return self == dtype.paillier

    def type_promoted(self, other: "dtype") -> "dtype":
        if self.rank < other.rank:
            return other
        else:
            return self

    def to_torch_dtype(self):
        if self == dtype.int32:
            return torch.int32
        if self == dtype.int64:
            return torch.int64
        if self == dtype.float64:
            return torch.float64
        if self == dtype.float32:
            return torch.float32
        raise TypeError(f"unsupported type: {self}")

    @classmethod
    def from_torch_dtype(cls, t_type):
        if t_type == torch.int32:
            return dtype.int32
        if t_type == torch.int64:
            return dtype.int64
        if t_type == torch.float64:
            return dtype.float64
        if t_type == torch.float32:
            return dtype.float32
        raise TypeError(f"unsupported type: {t_type}")
