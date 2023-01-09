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
from fate.arch.tensor.types import DStorage


def sum(storage: DStorage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    if dim is not None and not kwargs.get("keepdim", False):
        kwargs["keepdim"] = True
    local_ops = storage.local_ops_helper()
    output = DStorage.unary_op(storage, lambda x: local_ops.sum(x, *args, **kwargs))
    if dim is None or dim == storage.d_axis.axis:
        output = output.blocks.reduce(lambda x, y: local_ops.add(x, y))
    return output


def mean(storage: DStorage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    local_ops = storage.local_ops_helper()
    if dim is not None and dim != storage.d_axis.axis:
        count = storage.shape[dim]
        return DStorage.unary_op(storage, lambda x: local_ops.truediv(local_ops.sum(x, *args, **kwargs), count))
    else:
        output = DStorage.unary_op(storage, lambda x: local_ops.sum(x, *args, **kwargs))
        if dim is None:
            count = storage.shape.prod()
        else:
            count = storage.shape[dim]
        output = output.blocks.reduce(lambda x, y: local_ops.add(x, y))
        return local_ops.truediv(output, count)


def var(storage: DStorage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    unbiased = kwargs.get("unbiased", True)

    local_ops = storage.local_ops_helper()
    if dim is not None and dim != storage.d_axis.axis:
        return DStorage.unary_op(storage, lambda x: local_ops.var(x, dim=dim, unbiased=unbiased))

    else:
        if dim is None:
            n = storage.shape.prod()

            def _mapper(x):
                return (local_ops.sum(local_ops.square(x)), local_ops.sum(x))

        else:
            n = storage.shape[dim]

            def _mapper(x):
                return (local_ops.sum(local_ops.square(x), dim=dim), local_ops.sum(x, dim=dim))

        def _reducer(x, y):
            return (local_ops.add(x[0], y[0]), local_ops.add(x[1], y[1]))

        sq, s = storage.blocks.mapValues(_mapper).reduce(_reducer)
        output = local_ops.sub(local_ops.div(sq, n), local_ops.square(local_ops.div(s, n)))
        if unbiased:
            output = local_ops.mul(output, n / (n - 1))
        return output


def std(storage: DStorage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    unbiased = kwargs.get("unbiased", True)

    local_ops = storage.local_ops_helper()
    if dim is not None and dim != storage.d_axis.axis:
        return DStorage.unary_op(storage, lambda x: local_ops.std(x, dim=dim, unbiased=unbiased))

    else:
        if dim is None:
            n = storage.shape.prod()

            def _mapper(x):
                return (local_ops.sum(local_ops.square(x)), local_ops.sum(x))

        else:
            n = storage.shape[dim]

            def _mapper(x):
                return (local_ops.sum(local_ops.square(x), dim=dim), local_ops.sum(x, dim=dim))

        def _reducer(x, y):
            return (local_ops.add(x[0], y[0]), local_ops.add(x[1], y[1]))

        sq, s = storage.blocks.mapValues(_mapper).reduce(_reducer)
        output = local_ops.sub(local_ops.div(sq, n), local_ops.square(local_ops.div(s, n)))
        if unbiased:
            output = local_ops.mul(output, n / (n - 1))
        output = local_ops.sqrt(output)
        return output


def max(storage: DStorage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    local_ops = storage.local_ops_helper()
    if dim is None:

        def _mapper(x):
            return local_ops.max(x)

        return storage.blocks.mapValues(_mapper).reduce(lambda x, y: local_ops.maximum(x, y))
    else:
        if dim == storage.d_axis.axis:
            return storage.blocks.mapValues(lambda x: local_ops.max(x, dim=dim)).reduce(
                lambda x, y: local_ops.maximum(x, y)
            )
        else:
            return DStorage.unary_op(storage, lambda s: local_ops.max(s, *args, **kwargs))


def min(storage: DStorage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    local_ops = storage.local_ops_helper()
    if dim is None:

        def _mapper(x):
            return local_ops.min(x)

        return storage.blocks.mapValues(_mapper).reduce(lambda x, y: local_ops.minimum(x, y))
    else:
        if dim == storage.d_axis.axis:
            return storage.blocks.mapValues(lambda x: local_ops.min(x, dim=dim)).reduce(
                lambda x, y: local_ops.minimum(x, y)
            )
        else:
            return DStorage.unary_op(storage, lambda s: local_ops.min(s, *args, **kwargs))
