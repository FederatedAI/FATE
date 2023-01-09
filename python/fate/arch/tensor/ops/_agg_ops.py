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
from typing import overload

from .._tensor import Tensor

# TODO: parameter `keepdim` maybe a bit complex in distributed version, fix me later


@overload
def sum(a: Tensor, *, dtype=None) -> Tensor:
    ...


@overload
def sum(a: Tensor, dim, keepdim=False, *, dtype=None) -> Tensor:
    ...


def sum(a: Tensor, *args, **kwargs):
    storage = a.storage
    if func := getattr(storage, "sum"):
        return Tensor(func(*args, **kwargs))
    else:
        raise NotImplementedError(f"sum not impl for tensor `{a}` with storage `{storage}`")


def mean(a: Tensor, *args, **kwargs):
    storage = a.storage
    if func := getattr(storage, "mean"):
        return Tensor(func(*args, **kwargs))
    else:
        raise NotImplementedError(f"mean not impl for tensor `{a}` with storage `{storage}`")


def std(a: Tensor, *args, **kwargs):
    storage = a.storage
    if func := getattr(storage, "std"):
        return Tensor(func(*args, **kwargs))
    else:
        raise NotImplementedError(f"std not impl for tensor `{a}` with storage `{storage}`")


def var(a: Tensor, *args, **kwargs):
    storage = a.storage
    if func := getattr(storage, "var"):
        return Tensor(func(*args, **kwargs))
    else:
        raise NotImplementedError(f"var not impl for tensor `{a}` with storage `{storage}`")


def max(a: Tensor, *args, **kwargs):
    storage = a.storage
    if func := getattr(storage, "max"):
        return Tensor(func(*args, **kwargs))
    else:
        raise NotImplementedError(f"max not impl for tensor `{a}` with storage `{storage}`")


def min(a: Tensor, *args, **kwargs):
    storage = a.storage
    if func := getattr(storage, "min"):
        return Tensor(func(*args, **kwargs))
    else:
        raise NotImplementedError(f"min not impl for tensor `{a}` with storage `{storage}`")
