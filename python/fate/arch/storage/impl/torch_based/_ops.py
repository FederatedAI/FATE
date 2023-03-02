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

from ..._dtype import dtype
from ..._shape import Shape
from ._storage import _TorchStorage


def from_torch(t: torch.Tensor):
    return _TorchStorage(dtype.from_torch_dtype(t.dtype), Shape(t.shape), t)


def randn(shape, _dtype: dtype):
    return _TorchStorage(
        dtype.from_torch_dtype(_dtype), Shape(shape), torch.randn(shape, dtype=_dtype.to_torch_dtype())
    )


def ones(shape, _dtype: dtype):
    return _TorchStorage(
        dtype.from_torch_dtype(_dtype), Shape(shape), torch.ones(shape, dtype=_dtype.to_torch_dtype())
    )


def zeros(shape, _dtype: dtype):
    return _TorchStorage(
        dtype.from_torch_dtype(_dtype), Shape(shape), torch.zeros(shape, dtype=_dtype.to_torch_dtype())
    )


def quantile(storage, q, epsilon):
    from fate_utils.quantile import quantile_f64_ix2

    return quantile_f64_ix2(storage.data.numpy(), q, epsilon)


def quantile_summary(storage, epsilon):
    from fate_utils.quantile import summary_f64_ix2

    return summary_f64_ix2(storage.data.numpy(), epsilon)


def mean(storage, *args, **kwargs):
    return _lift(torch.mean(storage.data, *args, **kwargs))


def sum(storage, *args, **kwargs):
    return _lift(torch.sum(storage.data, *args, **kwargs))


def var(storage, *args, **kwargs):
    return _lift(torch.var(storage.data, *args, **kwargs))


def std(storage, *args, **kwargs):
    return _lift(torch.std(storage.data, *args, **kwargs))


def square(storage, *args, **kwargs):
    return _lift(torch.square(storage.data, *args, **kwargs))


def sqrt(storage, *args, **kwargs):
    return _lift(torch.sqrt(storage.data, *args, **kwargs))


def exp(storage, *args, **kwargs):
    return _lift(torch.exp(storage.data, *args, **kwargs))


def max(storage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    if dim is None:
        output = torch.as_tensor(storage.data.max(*args, **kwargs))
    else:
        output = storage.data.max(*args, **kwargs).values
    return _lift(output)


def min(storage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    if dim is None:
        output = torch.as_tensor(storage.data.min(*args, **kwargs))
    else:
        output = storage.data.min(*args, **kwargs).values
    return _lift(output)


def add(a, b):
    return _lift(torch.add(_maybe_unwrap_storage(a), _maybe_unwrap_storage(b)))


def sub(a, b):
    return _lift(torch.sub(_maybe_unwrap_storage(a), _maybe_unwrap_storage(b)))


def mul(a, b):
    return _lift(torch.mul(_maybe_unwrap_storage(a), _maybe_unwrap_storage(b)))


def div(a, b):
    return _lift(torch.div(_maybe_unwrap_storage(a), _maybe_unwrap_storage(b)))


def maximum(a, b):
    return _lift(torch.maximum(_maybe_unwrap_storage(a), _maybe_unwrap_storage(b)))


def minimum(a, b):
    return _lift(torch.minimum(_maybe_unwrap_storage(a), _maybe_unwrap_storage(b)))


def truediv(a, b):
    return _lift(torch.true_divide(_maybe_unwrap_storage(a), _maybe_unwrap_storage(b)))


def slice(a, key):
    return _lift(_maybe_unwrap_storage(a)[key])


def matmul(a, b):
    return _lift(torch.matmul(_maybe_unwrap_storage(a), _maybe_unwrap_storage(b)))


def _lift(output) -> "_TorchStorage":
    output_dtype = dtype.from_torch_dtype(output.dtype)
    output_shape = Shape(output.shape)
    return _TorchStorage(output_dtype, output_shape, output)


def _maybe_unwrap_storage(s):
    if isinstance(s, _TorchStorage):
        return s.data
    else:
        return s
